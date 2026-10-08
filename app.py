# -*- coding: utf-8 -*-
"""
Streamlit 骨架：區域／主題篩選 → 天數 → 地圖總覽（KMeans 分組＋路線建議）→ 選擇景點。
資料直接查 travel_hub_official.db，分群／排序直接呼叫既有的
kmeans_clustering.cluster_by_days() 與 route_optimization.greedy_route()，
不重寫演算法邏輯。

同一個景點如果被好幾篇不同文章介紹過，資料庫故意保留成好幾列（見資料清洗
階段的去重規則：改成 (address, source_url) 去重，不是同名去重），這裡把
它們合併成一張卡片，列出每篇文章各自的標題（超連結）＋摘要，讓使用者先
看摘要判斷哪篇對自己有幫助，再決定要點開哪篇原文。

地圖圖釘的 popup 走不同的精簡路線：不放連結（連結留給卡片，地圖是快速掃過
的情境），摘要也不是列出每篇文章各自的版本，而是用 summary_extraction.py
的 TF-IDF 排名法，把好幾篇文章的摘要句子合併成一個句子池再排名一次，抽出
跨文章有共識、比較能代表這個景點的句子——卡片跟地圖看到的資料還是同一份
（articles 清單），只是為不同情境做了不同程度的精簡，不是各自維護一份。
"""
import csv
import io
import json
import sqlite3
from datetime import date, datetime, timedelta

import folium
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from summary_extraction import extract_consensus_summary

from closed_day_rules import check_conflict, parse_closed_days
from hotel_recommendation import TOKYO_HUB_STATIONS, recommend_hotel_for_trip
from kmeans_clustering import DAILY_SPOT_WARNING_THRESHOLD, cluster_by_days
from route_optimization import LONG_LEG_WARNING_KM, greedy_route, haversine_km

DB_PATH = "travel_hub_official.db"

WEEKDAY_ZH = ["一", "二", "三", "四", "五", "六", "日"]

# 交通警示附的官方鐵路路線圖連結（提案表格四要求，之前只做了距離文字警示，這是補上
# 的連結）。專案一開始就決定不接 Google Distance Matrix 這類路線規劃API，所以這裡只
# 能給一個固定的官方路網圖，讓使用者自己對照查怎麼搭，不是幫他查好兩點間的路線。
# 東京地鐵（Tokyo Metro）官方網站繁體中文版路線圖頁面，選它是因為涵蓋東京市區最主要
# 的9條地鐵路線、且有官方認證的繁中版本（不是第三方轉譯），跟本專案繁中呈現一致。
TOKYO_METRO_MAP_URL = "https://www.tokyometro.jp/tcn/subwaymap/index.html"

# 分群用的天數色票，跟外觀提案定案的抹茶配色（day1 抹茶綠/day2 赤茶）延伸而來
DAY_COLORS = ["#6E8C46", "#C1663A", "#3B7D93", "#8B5FBF", "#B08900", "#4E86E1"]

# 還沒選天數/心願單時的瀏覽地圖：全部景點用同一種中性色，跟規劃模式的分天配色區分開
BROWSE_COLOR = "#5C6B73"

# 地圖底圖：原本用 CartoDB positron，後來 CARTO 改成需要 API 金鑰，底圖只剩
# 「API KEY REQUIRED」浮水印，改用 Esri 淺灰底圖（免金鑰，外觀跟原本的淺灰底接近）
BASEMAP_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
    "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)
BASEMAP_ATTR = "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ"

# 瀏覽模式地圖圖示：依景點類型顯示不同 emoji＋底色圓圈（提案項目7）。用 emoji 而不是
# Font Awesome 圖示字型，是因為瀏覽模式地圖是另外快取渲染的靜態 HTML，沒有掛
# Leaflet.awesome-markers 的圖示字型資源，emoji 靠瀏覽器原生字型顯示，不用額外載入
# 任何東西就能穩定顯示，也剛好每個 emoji 本身都跟類型意義對應（⛩️=神社、♨️=溫泉）。
SPOT_TYPE_ICONS = {
    "美食": ("🍜", "#C0392B"),
    "購物": ("🛍️", "#8E44AD"),
    "寺廟神社": ("⛩️", "#D35400"),
    "公園綠地": ("🌳", "#27AE60"),
    "觀覽景點": ("🖼️", "#2980B9"),
    "娛樂設施": ("🎡", "#D6336C"),
    "溫泉錢湯": ("♨️", "#16A0A6"),
    "複合商場": ("🏬", "#B7950B"),
    "未分類": ("📍", BROWSE_COLOR),
}

# 期間限定活動（iwafu）的分類配色：直接沿用 DAY_COLORS 裡已經在這個底色（抹茶/奶油）
# 上驗證過夠鮮明的 5 色（跳過 DAY_COLORS[0] 的抹茶綠，那是網站專屬 accent 色，不挪作他用），
# 不是重新調一組新顏色——原本用彩色圓點 emoji（🟠🟣🟡🔴🔵）當圖例，但 emoji 是系統內建
# 固定色，配這個底色怎麼調都不夠鮮明也不搭；改成直接把分類文字染色（不加任何圓點/徽章
# 形狀），跟側欄其他區塊「純文字＋emoji圖示」的扁平風格一致，不會憑空多一個突兀的色塊。
IWAFU_BADGE_COLORS = {
    "活動": "#3B7D93",  # 對應 DAY_COLORS 灰藍
    "展覽": "#8B5FBF",  # 對應 DAY_COLORS 紫
    "市集": "#D9A521",  # 比 DAY_COLORS 芥末黃（#B08900）調亮，使用者反饋太暗
    "美食": "#D97B4F",  # 比 DAY_COLORS 赤茶橘（#C1663A）調亮，使用者反饋太暗
    "動漫": "#4E86E1",  # 對應 DAY_COLORS 天藍
}
# Streamlit 的 widget key 若含中文字，轉成 CSS class 時會被壓縮成同樣的短橫線（例如
# 「美食」「購物」都變成 st-key-theme_--），沒辦法用 CSS 精準選到單一分類，所以額外
# 準備一組英文 slug 給 checkbox 的 key 用，讓下面的顏色 CSS 選得到正確的那一顆。
IWAFU_CATEGORY_SLUGS = {"活動": "event", "展覽": "expo", "市集": "market", "美食": "food", "動漫": "anime"}
# 側欄顯示用的替代名稱：iwafu 的活動類別「美食」跟上面景點類型的「美食」同名，兩個區塊
# 上下相鄰，使用者反映容易看混（一個是 472 個常設美食景點，一個是 5 個期間限定美食活動）。
# 只改側欄的顯示文字，資料庫的 category 值跟卡片/地圖上的類別色塊維持原樣——那兩處旁邊
# 就有 🎪 圖示標明是活動，不會混淆，不需要跟著加字。
IWAFU_CATEGORY_LABELS = {"美食": "限定美食"}

# 這幾組是同一個品牌的不同分店，原文本身就用同一段話介紹好幾間分店（不是description
# 抓取邏輯的bug，查證過原文結構就是如此，見docs/05§10），卡片上會出現一模一樣的介紹
# 文字——不加說明使用者容易誤以為是資料重複/抓錯，加一行小字讓使用者知道這是刻意的。
BRANCH_SHARED_DESC_NOTE = {
    349: "🏬 品牌分店，介紹文字沿用原文的共同介紹", 350: "🏬 品牌分店，介紹文字沿用原文的共同介紹",  # WEGO 原宿本店 / WEGO 1.3.5... 原宿店
    1592: "🏬 品牌分店，介紹文字沿用原文的共同介紹", 1593: "🏬 品牌分店，介紹文字沿用原文的共同介紹",  # 原宿1階店・表参道じゃんがら 2F店 / 西武池袋店
    2435: "🏬 品牌分店，介紹文字沿用原文的共同介紹", 2438: "🏬 品牌分店，介紹文字沿用原文的共同介紹",  # 東京ばな奈スタジオ 大丸東京店 / 羽田空港店
    2416: "🎪 定期市集活動，非常設店家，兩處會場共用介紹", 2417: "🎪 定期市集活動，非常設店家，兩處會場共用介紹",  # 大江戶古董市：東京國際論壇．地上廣場 / 代代木公園．欅並木
}


def spot_marker_html(spot_type, badge_color=None):
    """瀏覽模式地圖的圖釘 HTML：白底圓圈＋類型 emoji（外框用類型色），右上角視情況疊加
    一個星號徽章（有 iwafu 期間限定活動對應到這個景點時才顯示，顏色依活動主題區分）。
    用 folium.DivIcon 包這段 HTML，而不是 folium.Icon，因為要同時疊兩層（主圖示＋徽章），
    folium.Icon 的 AwesomeMarkers 圖示只能顯示單一圖示，做不到疊加徽章。

    底色原本用類型色實心填滿，但深色底會把 emoji 自己的顏色/線條吃掉、看不清楚符號
    （emoji 本身就是彩色圖案，不是純色圖示，深色底跟 emoji 的顏色搶視覺）——改成白底
    ＋類型色外框，emoji 在白底上不管什麼顏色都看得清楚，類型色改放到外框，一樣能分辨
    類型，只是不再用實心色塊。"""
    emoji, color = SPOT_TYPE_ICONS.get(spot_type or "未分類", SPOT_TYPE_ICONS["未分類"])
    badge_html = ""
    if badge_color:
        badge_html = (
            f'<div style="position:absolute; top:-3px; right:-3px; width:15px; height:15px; '
            f'border-radius:50%; background:{badge_color}; border:1.5px solid white; '
            f'display:flex; align-items:center; justify-content:center; font-size:9px; '
            f'line-height:1; color:white;">★</div>'
        )
    return (
        f'<div style="position:relative; width:32px; height:32px;">'
        f'<div style="width:28px; height:28px; border-radius:50%; background:#FFFFFF; '
        f'border:2.5px solid {color}; box-shadow:0 1px 3px rgba(0,0,0,0.35); display:flex; '
        f'align-items:center; justify-content:center; font-size:15px; line-height:1;">{emoji}</div>'
        f'{badge_html}</div>'
    )

# 地圖 popup 的共識摘要長度上限：popup max_width=280px，中文約 20 字一行。原本抓 5 行
# （100 字），使用者實測回饋 popup 整個太高、要捲動地圖才看得完，降到 3 行的量。
MAP_SUMMARY_MAX_CHARS = 60
# 營業時間在 popup 裡的長度上限（見 spot_fact_lines 的 hours_limit）
MAP_HOURS_MAX_CHARS = 40


def truncate_for_map(text):
    if text and len(text) > MAP_SUMMARY_MAX_CHARS:
        return text[:MAP_SUMMARY_MAX_CHARS] + "…"
    return text

SPOT_FIELDS = (
    "id, name, neighborhood, address, latitude, longitude, cleaned_closed_days, "
    "transit, business_hours_raw, summary, source_url, article_title, spot_type"
)

# 初始狀態刻意「什麼都沒選」：區域篩選空白（先看到全部景點，再自己縮小範圍）、
# 心願單空白、天數 0（代表「還沒開始規劃」，地圖/分群要天數 >0 才會跑），
# 讓使用者從一片空白自己建立心願單，不是一進來就先幫他決定好一組示範資料。

st.set_page_config(page_title="旅遊前置決策與景點分群系統", layout="wide", page_icon="🗾")

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: "Hiragino Sans", "Noto Sans TC", "Microsoft JhengHei", sans-serif;
    }
    h1, h2, h3 {
        font-family: "Zen Kaku Gothic New", "Hiragino Kaku Gothic ProN", "Noto Sans TC", sans-serif;
    }
    /* 景點卡片：白底＋陰影，跟頁面底色區隔開，視覺上更像獨立卡片、更有重點 */
    div[class*="st-key-spot_card_"] {
        background-color: #FFFFFF;
        box-shadow: 0 2px 10px rgba(43, 48, 33, 0.12);
        border-radius: 10px;
    }
    /* 心願單移除的✕按鈕：拿掉邊框跟底色，只留符號本身，內容置中 */
    div[class*="st-key-remove_"] button {
        border: none;
        background: transparent;
        box-shadow: none;
        padding: 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    div[class*="st-key-remove_"] button:hover {
        background: rgba(43, 48, 33, 0.08);
        border-radius: 4px;
    }
    /* 清空按鈕：真正的問題不是置中邏輯，是按鈕本身太窄——Streamlit 預設左右
       padding 12px，欄位只有約44px寬時，扣掉padding只剩約18px給文字，「清空」
       兩個字實際要28px，文字會溢出、視覺上看起來像被擠到一邊。縮小左右padding，
       讓內容框有足夠空間，Streamlit 原本的置中（button 預設就是 flex置中）就會
       正常生效，不需要額外強制 width/text-align。 */
    div[class*="st-key-clear_wishlist"] button {
        white-space: nowrap !important;
        padding: 4px 6px !important;
    }
    /* 上一頁/下一頁按鈕：跟清空按鈕同樣道理，欄位縮小後要縮小padding讓文字塞得下。
       景點分頁與活動分頁兩組按鈕都要，不然活動那組的文字會被擠成兩行。 */
    div[class*="st-key-page_prev"] button,
    div[class*="st-key-page_next"] button,
    div[class*="st-key-event_prev"] button,
    div[class*="st-key-event_next"] button {
        white-space: nowrap !important;
        padding: 4px 6px !important;
    }
    /* 已儲存計畫的載入/刪除按鈕：欄位比例 [3,1,1] 更窄，同樣的padding過寬問題 */
    div[class*="st-key-load_plan_"] button,
    div[class*="st-key-delete_plan_"] button {
        white-space: nowrap !important;
        padding: 4px 6px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 期間限定活動分類色點：一開始試過把文字整個染色，使用者回饋改用實心圓點放在文字前面——
# emoji 圓點（🟠🟣等）有光澤感，人眼會讀成立體，跟其他扁平風格不搭；改用 CSS 畫的純色圓
# （::before 偽元素，checkbox 的 label 是 markdown 渲染出來的純文字，塞不進真的 <span>，
# 只能用 CSS 在文字前面「畫」一個圓點），顏色一樣直接來自 IWAFU_BADGE_COLORS（跟地圖星號
# 徽章同一份資料）。CSS 選擇器用 IWAFU_CATEGORY_SLUGS 的英文 slug（中文 key 轉成 CSS class
# 時會被壓縮成同樣的短橫線，選不到單一分類，見該常數旁的註解）。
st.markdown(
    "<style>" + "".join(
        f'div[class*="st-key-theme_{slug}"] label p::before {{ content: ""; display: inline-block; '
        f'width: 10px; height: 10px; border-radius: 50%; background: {IWAFU_BADGE_COLORS[cat]}; '
        f'margin-right: 6px; vertical-align: middle; }}'
        for cat, slug in IWAFU_CATEGORY_SLUGS.items()
    ) + "</style>",
    unsafe_allow_html=True,
)


@st.cache_data
def load_grouped_spots():
    """把同一個 (name, address) 的多筆文章列合併成一個景點：地址／經緯度／公休／交通／
    營業時間取最小 id 那筆（同一景點這些事實類欄位理論上一致，取一筆當代表就好），
    summary／source_url／article_title 收集成 articles 清單保留每篇文章各自的內容。
    最小 id 當這個景點對外代表的 primary_id，心願單/KMeans分群/地圖標記都用這個 id，
    回傳 (by_primary_id, id_to_primary_id)。"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"SELECT {SPOT_FIELDS} FROM mitsugo_spots ORDER BY id")
    rows = cur.fetchall()
    conn.close()

    groups = {}
    row_key_by_id = {}
    for (sid, name, nb, address, lat, lon, closed, transit, hours, summary, url, article_title, spot_type) in rows:
        key = (name, address)
        row_key_by_id[sid] = key
        if key not in groups:
            groups[key] = {
                "primary_id": sid, "name": name, "neighborhood": nb, "address": address,
                "latitude": lat, "longitude": lon, "closed": closed, "transit": transit,
                "hours": hours, "spot_type": spot_type, "articles": [],
            }
        groups[key]["articles"].append((article_title, summary, url))

    by_primary = {g["primary_id"]: g for g in groups.values()}
    id_to_primary = {sid: groups[key]["primary_id"] for sid, key in row_key_by_id.items()}
    return by_primary, id_to_primary


@st.cache_data
def load_consensus_summaries(_by_primary):
    """每個景點的跨文章共識摘要要跑 TF-IDF（見 summary_extraction.extract_consensus_summary），
    瀏覽地圖一次要畫全部景點的 popup，不能每次 rerun 都重新抽取一次，所以在這裡一次算完＋
    快取。參數名加底線讓 st.cache_data 不要嘗試 hash 這個 dict（改用函式本身的快取，資料
    不變就不會重算）。"""
    return {
        pid: extract_consensus_summary([summary for _title, summary, _url in rec["articles"]])
        for pid, rec in _by_primary.items()
    }


def iwafu_theme_spot_counts(spot_categories):
    """每個活動類別「勾了會篩出幾個景點」。

    吃的是「每個景點有哪些活動類別」的集合（不是單一徽章類別）：一個景點可能同時有
    美食跟活動兩種類別的活動（例如六本木之丘展望台），兩個分類各算它一次，因為兩邊
    勾選確實都篩得到它。"""
    counts = {}
    for categories in spot_categories.values():
        for category in categories:
            counts[category] = counts.get(category, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


def iwafu_theme_label(category, event_count, spot_count):
    """側欄「期間限定活動」checkbox 的文字：場數跟景點數兩個都要寫。

    這顆勾選框同時影響兩個分頁——活動分頁篩出「這一類的活動」、景點分頁篩出「有這一類
    活動的景點」——而兩邊的數字落差很大（例如「活動」有 82 場，但只有 13 個景點掛得上）。
    只寫其中一個，不管寫哪個都會在另一頁對不上，使用者會以為資料掉了。

    順帶一提，這個落差本身就是有意義的資訊：大多數活動辦在飯店宴會廳、展演場館、商店街，
    那些地點本來就不在樂吃購的景點庫裡，所以配得到景點的只是少數。"""
    return f"{IWAFU_CATEGORY_LABELS.get(category, category)}（{event_count} 場・{spot_count} 景點）"


def format_event_period(start, end):
    """活動檔期顯示格式：同一年只寫一次年份（2026/07/01 – 08/31），跨年才兩邊都寫完整
    （2026/11/13 – 2027/02/14）。卡片是三欄窄版、地圖 popup 只有 280px，省下的字寬有感。"""
    if start.year == end.year:
        return f"{start:%Y/%m/%d} – {end:%m/%d}"
    return f"{start:%Y/%m/%d} – {end:%Y/%m/%d}"


def event_category_chip(category):
    """活動類別的色塊標籤（HTML）。顏色直接用 IWAFU_BADGE_COLORS，跟側欄圖例的分類
    文字染色、地圖圖釘右上角的星號徽章是同一組對應關係，使用者看顏色就能串起來。"""
    color = IWAFU_BADGE_COLORS.get(category, BROWSE_COLOR)
    return (f"<span style='background:{color};color:#FFFFFF;font-size:11px;"
            f"border-radius:4px;padding:1px 6px;'>{category}</span>")


def has_real_info(text):
    """判斷一個欄位是不是真的有內容——資料庫裡有兩種「看起來有值、其實等於沒有」的佔位符：
    公休欄位的「查無公休資訊，詳情請以商家公告為準」（725/888 筆，fill_missing_closed_days.py
    統一填的）、交通欄位的「暫無資料」（507/888 筆，爬蟲階段就這樣寫）。

    畫面上不顯示這種列：每張卡片、每個地圖 popup 都印一次同一句話，除了佔掉版面，反而把
    真正有資訊的那幾筆稀釋掉（長得跟廢話那行一模一樣，沒有辨識度）。公休的提醒不是刪掉，
    是移到規劃階段統整顯示一次（見下方 unknown_closed），出現在使用者真正需要的決策時點。"""
    return bool(text) and not text.startswith("查無") and not text.startswith("暫無")


def truncate_text(text, limit):
    """地圖 popup 專用的長度上限。popup 是浮在地圖上的小視窗，內容太長會蓋掉半張地圖、
    還要捲動才看得完（使用者實測回饋），所以營業時間、摘要這類長度不可控的欄位要截斷。"""
    if text and len(text) > limit:
        return text[:limit] + "…"
    return text


def parse_iwafu_period(period_str):
    """iwafu_events.period 是「YYYY.MM.DD ～ YYYY.MM.DD」這種乾淨格式（229筆全部有值，
    格式一致，爬蟲階段就已經是這樣），直接拆解成 (start_date, end_date)；格式不符就
    誠實回傳 None，不硬猜，呼叫端要自己過濾掉 None。"""
    if not period_str or "～" not in period_str:
        return None
    start_str, end_str = period_str.split("～", 1)
    try:
        start = datetime.strptime(start_str.strip(), "%Y.%m.%d").date()
        end = datetime.strptime(end_str.strip(), "%Y.%m.%d").date()
    except ValueError:
        return None
    return (start, end)


@st.cache_data
def load_iwafu_event_options(_id_to_primary):
    """全部「還沒結束」的活動（目前 175 場），每筆是一個 dict。側欄「推薦旅遊日期」的選單
    跟主畫面「期間限定活動」分頁共用這一份，不各撈各的。

    只保留 period 能正確解析、且 end >= 今天的活動：已經結束的活動對「規劃未來行程」沒有
    意義，選了只會推薦一個不可能成行的日期範圍；排除掉也順便降低少數 period 資料還沒被
    驗證過的殘留筆數造成的實際影響（見 iwafu_period_review.txt）。

    依檔期開始日排序，讓活動分頁預設就是「快要開始的排前面」。"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT title, category, period, venue, source_url, matched_spot_id "
        "FROM iwafu_events ORDER BY title"
    )
    rows = cur.fetchall()
    conn.close()
    today = date.today()
    options = []
    for title, category, period_str, venue, url, matched_spot_id in rows:
        parsed = parse_iwafu_period(period_str)
        if not parsed or parsed[1] < today:
            continue
        options.append({
            "title": title, "category": category, "period_str": period_str,
            "start": parsed[0], "end": parsed[1], "venue": venue, "url": url,
            # 有配對到景點的活動，順便把景點那邊的 primary_id 帶上，活動卡片才能顯示
            # 「地點：XXX」並提供直接加入心願單的按鈕（175 場裡有 41 場配得到）。
            "spot_id": _id_to_primary.get(matched_spot_id) if matched_spot_id else None,
            "label": iwafu_event_label(title, period_str, venue),
        })
    options.sort(key=lambda e: (e["start"], e["end"], e["title"]))
    return options


def iwafu_event_label(title, period_str, venue):
    """選單標籤要加上會場名稱：同一個活動可能在好幾個場地同時舉辦，iwafu 本身就是拆成
    好幾筆（例如「Hills 美食漢堡大獎賽 2026」分成六本木之丘／麻布台之丘／虎之門之丘三筆，
    名稱跟期間完全一樣）。只用「名稱＋期間」當標籤的話，選單裡會出現看起來一模一樣的
    選項，而且下面的 label→event 字典會把它們壓成同一筆、另外兩筆根本選不到。
    加上 venue 之後 175 筆全部唯一（venue 有 2 筆是空的，退回原本的寫法）。"""
    base = f"{title}｜{venue}" if venue else title
    return f"{base}（{period_str}）"


def find_full_overlap(periods):
    """periods：[(start,end), ...]。回傳全部區間都重疊的共同區間，沒有共同重疊回傳 None。"""
    if not periods:
        return None
    start = max(p[0] for p in periods)
    end = min(p[1] for p in periods)
    return (start, end) if start <= end else None


def find_pairwise_overlaps(events):
    """events：[(title, start, end), ...]，用在全體沒有共同重疊時，退而求其次列出
    「哪兩個活動之間」還有重疊，讓使用者自己判斷要犧牲哪個、或分兩趟去。"""
    overlaps = []
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            t1, s1, e1 = events[i]
            t2, s2, e2 = events[j]
            ov = find_full_overlap([(s1, e1), (s2, e2)])
            if ov:
                overlaps.append((t1, t2, ov[0], ov[1]))
    return overlaps


def describe_trip_vs_window(plan_days, plan_start, ov_start, ov_end, window_desc):
    """把「使用者目前設定的行程」跟「選到的活動期間」做比對，回傳 (level, 完整訊息)。

    `window_desc` 是活動期間那半句（例如「選的 3 個活動共同期間落在 2026/08/06 ～
    2026/08/16」），由呼叫端依「單一活動／多活動共同區間」組好傳進來。行程與活動期間
    合成同一句、只印一個提示框，不然同一個日期範圍會被講兩次。

    **這裡只回答「行程有沒有落在活動期間內」，不宣稱使用者參加得到什麼。** 能不能參加
    取決於使用者怎麼安排——同一天跑三個活動、還是分三天各跑一個，是他自己的決定，
    系統不知道也不該替他斷言。早期版本寫成「這些活動都參加得到」，是越界的推論。

    **判斷標準是「有沒有重疊」，不是「有沒有完整落在期間內」**：活動期間 08/06～08/16、
    行程 08/05 出發玩三天（05～07），08/06 與 08/07 兩天落在期間內就算數，不需要整趟
    都塞進活動期間。也因此不計算「建議出發日」——出發日不是硬性條件，算出一個範圍
    反而會讓使用者以為一定要在那個區間出發。

    日期一律印完整年月日：活動檔期會跨年（例如 2026.11.13 ～ 2027.02.14），只印月/日
    會看不出「02/11」是明年。"""
    if not plan_days:
        return "ok", f"{window_desc}，建議安排在這段期間前往。"
    trip_end_date = plan_start + timedelta(days=plan_days - 1)
    trip_txt = f"{plan_start:%Y/%m/%d}～{trip_end_date:%Y/%m/%d}"
    hit = find_full_overlap([(plan_start, trip_end_date), (ov_start, ov_end)])
    if hit:
        return "ok", f"你目前的行程是 {trip_txt}，{window_desc}，行程已落在這段期間內。"
    return "conflict", f"你目前的行程是 {trip_txt}，{window_desc}，建議安排在這段期間前往。"


@st.cache_data
def load_iwafu_spot_events(_id_to_primary):
    """{primary_id: [活動dict, ...]} 給已比對過的 iwafu 活動用（matched_spot_id），一次
    供三個地方共用：地圖圖釘右上角的星號徽章、側欄「期間限定活動」篩選、景點卡片的活動
    清單（活動名稱＋類別＋檔期＋原頁連結）。

    原本這支只撈 category 一個欄位（函式名是 load_iwafu_spot_badges），所以全站沒有任何
    地方看得到活動叫什麼、什麼時候辦——地圖 popup 只印得出「期間限定：展覽」這種類別，
    卡片則完全沒有活動資訊。改成連 title/period/source_url 一起撈。

    **只保留還沒結束（end >= 今天）的活動**，跟側欄「推薦旅遊日期」的
    load_iwafu_event_options() 同一個判斷標準：已經結束的活動對規劃未來行程沒有意義，
    顯示出來只會讓使用者白跑一趟。副作用是「有活動徽章的景點數」會比資料庫裡的配對數少
    （50 筆配對分屬 35 個景點，濾掉已結束的之後剩 31 個景點），側欄篩選數字跟著變少是
    預期行為，不是資料掉了。period 解析不出來的也一併排除，因為卡片要顯示檔期，沒有可信
    的起訖日期就不顯示，不硬猜。

    同一個景點可能對到好幾個活動（最多的有 5 個），所以值是 list 不是單一活動，依檔期
    開始日排序，卡片會全部列出、地圖 popup 只顯示第一個＋「等 N 個」。"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT matched_spot_id, title, category, period, source_url, venue "
        "FROM iwafu_events WHERE matched_spot_id IS NOT NULL"
    )
    rows = cur.fetchall()
    conn.close()
    today = date.today()
    events = {}
    for spot_id, title, category, period_str, url, venue in rows:
        primary_id = _id_to_primary.get(spot_id)
        if primary_id is None:
            continue
        parsed = parse_iwafu_period(period_str)
        if not parsed or parsed[1] < today:
            continue
        events.setdefault(primary_id, []).append({
            "title": title, "category": category, "url": url,
            "start": parsed[0], "end": parsed[1],
            # 跟側欄「推薦旅遊日期」選單用同一個標籤字串，卡片上的「加入日期計算」按鈕
            # 才能直接把這個活動塞進那個選單的已選清單（兩邊必須完全一致才對得起來）。
            "label": iwafu_event_label(title, period_str, venue),
        })
    for spot_events in events.values():
        spot_events.sort(key=lambda e: e["start"])
    return events


def ensure_saved_plans_table():
    """`saved_plans` 是全新的表，不動任何既有資料，用 CREATE TABLE IF NOT EXISTS
    直接在 app 啟動時建立即可，不用另外寫一支一次性遷移腳本（跟 neighborhood/
    spot_type 那種要幫既有 892 筆資料回填新欄位的情況不同，這裡沒有舊資料要處理）。
    `name` 設 UNIQUE，儲存時用同名覆蓋（見 save_plan_to_db），使用者可以把它當
    「命名存檔格」重複儲存同一個計畫名稱來更新內容。"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS saved_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            data TEXT NOT NULL
        )"""
    )
    conn.commit()
    conn.close()


def save_plan_to_db(name, wishlist_ids, day_count, start_date_value, event_labels=None):
    """存的是「使用者輸入」——心願單、天數、行程開始日期、想參加的期間限定活動，不存
    KMeans 分組結果或路線順序。理由：KMeans 用固定 random_state=42（見
    kmeans_clustering.py），同一份 wishlist_ids＋day_count 重新分組一定得到一樣的結果，
    存分組後的輸出反而是多餘的冗餘資料，載入計畫後讓分組現場重算即可，不用擔心跟儲存
    當下對不上。event_labels 存的是「推薦旅遊日期」選單的已選項目（活動標籤字串），
    舊計畫沒有這個欄位，載入時用 .get(..., []) 給預設值，不用另外寫遷移腳本。"""
    payload = json.dumps({
        "wishlist_ids": sorted(wishlist_ids),
        "day_count": day_count,
        "start_date": start_date_value.isoformat(),
        "event_labels": event_labels or [],
    })
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO saved_plans (name, created_at, data) VALUES (?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET created_at = excluded.created_at, data = excluded.data""",
        (name, f"{datetime.now():%Y-%m-%d %H:%M}", payload),
    )
    conn.commit()
    conn.close()


def load_saved_plans_from_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, name, created_at, data FROM saved_plans ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "created_at": r[2], "data": r[3]} for r in rows]


def delete_saved_plan_from_db(plan_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM saved_plans WHERE id = ?", (plan_id,))
    conn.commit()
    conn.close()


# 景點類型的顯示順序（不是照筆數排序，觀覽景點/娛樂設施/溫泉錢湯這種較少但明確的
# 類別排前面比較容易被注意到，未分類放最後）
SPOT_TYPE_ORDER = ["美食", "購物", "寺廟神社", "公園綠地", "觀覽景點", "娛樂設施", "溫泉錢湯", "複合商場", "未分類"]


def spot_type_counts(by_primary):
    counts = {}
    for rec in by_primary.values():
        t = rec.get("spot_type") or "未分類"
        counts[t] = counts.get(t, 0) + 1
    return [(t, counts[t]) for t in SPOT_TYPE_ORDER if t in counts]


def spot_fact_lines(rec, closed_text=None, include_transit=True, combine_hours_closed=False,
                    hours_limit=None):
    """景點的地址／交通／營業時間／公休（不含文章摘要——摘要現在跟各篇文章標題成對顯示，
    見 article_blocks()）。地圖 popup 跟景點卡片都呼叫這個函式，確保欄位定義一致。

    `hours_limit` 只有地圖 popup 會傳：營業時間長度很不可控（「商場11:00〜21:00；餐廳
    11:00〜23:00（部分店舖營業時間不同，請參考營業時間表）」這種要吃掉三行），在浮動的
    小視窗裡會把地圖蓋掉；卡片版面夠寬不用截斷。"""
    lines = []
    if rec.get("address"):
        lines.append(f"📍 {rec['address']}")
    if include_transit and has_real_info(rec.get("transit")):
        lines.append(f"🚉 {rec['transit']}")
    hours = rec.get("hours")
    if hours and hours_limit:
        hours = truncate_text(hours, hours_limit)
    closed = closed_text or rec.get("closed")
    show_closed = has_real_info(closed)
    if combine_hours_closed:
        if hours and show_closed:
            lines.append(f"🕒 {hours}｜🚪 {closed}")
        elif hours:
            lines.append(f"🕒 {hours}")
        elif show_closed:
            lines.append(f"🚪 {closed}")
    else:
        if hours:
            lines.append(f"🕒 {hours}")
        if show_closed:
            lines.append(f"🚪 {closed}")
    return lines


def day_neighborhood_summary(group):
    """給 Day 摘要列用：一天的景點群裡有哪些地名，去重＋保留出現順序。KMeans 是照經緯度
    分群、不是照地名分群，同一天完全可能橫跨 2-3 個地名，超過 2 個就顯示前兩個＋「等N個」，
    避免文字長度不可控。"""
    seen = []
    for s in group:
        nb = spot_records.get(s["id"], {}).get("neighborhood") or s.get("area")
        if nb and nb not in seen:
            seen.append(nb)
    if not seen:
        return ""
    if len(seen) <= 2:
        return "、".join(seen)
    return f"{'、'.join(seen[:2])} 等{len(seen)}個"


def article_blocks(articles, truncate_summary=None):
    """把一個景點底下的每篇文章轉成 [連結文字, 摘要文字, ...] 交錯的清單，給景點卡片用
    markdown 連結（`st.caption`／`st.markdown` 直接吃 `[文字](網址)` 語法）。連結文字本身
    就是文章標題（不是通用的「查看原文」），讓使用者先看摘要判斷哪篇對自己有幫助、
    再決定點開哪篇——這是使用者要的「摘要跟原文連結搭配」設計。地圖 popup 改用更精簡的
    跨文章共識摘要（見 extract_consensus_summary()），不需要逐篇列出，所以不用這支函式。"""
    blocks = []
    for title, summary, url in articles:
        label = title or "（原文標題抓不到）"
        blocks.append(f"[{label}]({url})" if url else label)
        if summary:
            s = summary
            if truncate_summary and len(s) > truncate_summary:
                s = s[:truncate_summary] + "…"
            blocks.append(s)
    return blocks


def event_csv_rows(selected_events, spot_records):
    """把「推薦旅遊日期」選單裡已選的活動整理成 CSV 列，供 build_itinerary_csv／
    build_wishlist_csv 共用一份寫法。跟景點一樣有名稱／類別／檔期／地點／原文連結，
    只是活動沒有公休、交通這兩個概念，欄位對不上的地方不硬填。「對應景點」欄留給
    有配對到 mitsugo_spots 的活動（175 場裡 50 場），沒配對到的活動這欄留空，不是
    資料缺漏——本來就有活動是橫跨整個街區、沒有單一對應景點。"""
    rows = []
    for e in selected_events:
        spot_name = spot_records.get(e.get("spot_id"), {}).get("name", "") if e.get("spot_id") else ""
        rows.append([
            e["title"], e["category"], e["period_str"],
            e.get("venue") or "", spot_name, e.get("url") or "",
        ])
    return rows


def write_event_section(writer, selected_events, spot_records):
    """有選活動才附加這個區塊，沒選就不動 CSV 內容（維持原本沒有活動時的樣子）。"""
    if not selected_events:
        return
    writer.writerow([])
    writer.writerow(["推薦旅遊日期活動"])
    writer.writerow(["活動名稱", "類別", "檔期", "地點", "對應景點", "原文連結"])
    for row in event_csv_rows(selected_events, spot_records):
        writer.writerow(row)


def build_itinerary_csv(days, day_dates, conflicts, spot_records, active_day, active_route, selected_events=None):
    """把規劃好的行程（天數、區域、公休/交通提醒、原文連結）匯出成 CSV（提案表格五）。

    每一天都要有排好序的造訪順序：使用者在「路線建議」互動選過起點的那一天
    （active_day），直接沿用畫面上顯示的那份 active_route，匯出的順序才會跟畫面上
    看到的一致；其他天使用者沒有互動過、沒有指定起點，用「該天 id 最小的景點」當
    預設起點跑一次 greedy_route()，順序才是確定性的（不會每次匯出都不一樣）。

    UTF-8 with BOM（`utf-8-sig`）是提案書表格五明訂的編碼，Excel 開啟含中文的
    CSV 沒有 BOM 常會亂碼，這是通用的已知解法，不是這個專案特有的設計。"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Day", "日期", "順序", "景點名稱", "區域", "地址", "公休資訊", "交通", "原文連結"])

    for day, group in days.items():
        if day == active_day:
            ordered = active_route
        else:
            start_id = min(s["id"] for s in group)
            ordered, _ = greedy_route(group, start_id)

        visit_date = day_dates[day]
        for i, s in enumerate(ordered, 1):
            rec = spot_records.get(s["id"], {})
            closed_text = rec.get("closed") or "查無公休資訊"
            result = conflicts.get(s["id"])
            if result == "red":
                closed_text = f"公休衝突：{closed_text}"
            elif result == "yellow":
                closed_text = f"請自行查證：{closed_text}"
            urls = "; ".join(url for _title, _summary, url in rec.get("articles", []) if url)
            writer.writerow([
                f"Day{day}", f"{visit_date:%Y-%m-%d}", i, rec.get("name", s.get("name", "")),
                rec.get("neighborhood", "") or "", rec.get("address", "") or "",
                closed_text, rec.get("transit", "") or "", urls,
            ])

    write_event_section(writer, selected_events, spot_records)
    return output.getvalue().encode("utf-8-sig")


ensure_saved_plans_table()
by_primary, id_to_primary = load_grouped_spots()
spot_records = by_primary
consensus_summaries = load_consensus_summaries(by_primary)
iwafu_events_by_spot = load_iwafu_spot_events(id_to_primary)
# 地圖圖釘的星號徽章只能有一個顏色，用檔期最早（清單第一筆）那個活動的類別。
iwafu_badges = {pid: evs[0]["category"] for pid, evs in iwafu_events_by_spot.items()}
# 篩選則要看「這個景點的全部活動類別」：徽章只留得下一個類別，但六本木之丘展望台這種
# 同時有美食＋活動兩類活動的景點，勾任一類都應該篩得到它（不然卡片上列著「活動」、
# 勾「活動」卻找不到這張卡，使用者會覺得篩選壞了）。
iwafu_spot_categories = {
    pid: {e["category"] for e in evs} for pid, evs in iwafu_events_by_spot.items()
}


def matches_themes(primary_id, themes):
    """側欄「期間限定活動」的篩選判斷：沒勾代表不篩選；有勾就看這個景點的活動類別集合
    跟勾選的類別有沒有交集。地圖／卡片／分頁重置三處共用，避免各寫一份寫歪。"""
    if not themes:
        return True
    return bool(iwafu_spot_categories.get(primary_id, set()) & set(themes))


# 瀏覽地圖「可點擊」的景點數上限。實測 render 時間跟圖釘數幾乎成正比（50個0.10s／
# 150個0.34s／300個0.63s／866個1.64s＋1.3MB HTML），而 st_folium 是每次重跑都要付一次
# 這個成本（勾卡片、換頁、打字搜尋全算）。150 個以內 0.34 秒可以接受，超過就退回快取
# 好的靜態 HTML（快但收不到點擊）。實務上要在地圖上點著加景點時本來就會先篩到某一區。
CLICKABLE_MAP_MAX_SPOTS = 150


def build_wishlist_csv(wishlist_ids, spot_records, selected_events=None):
    """還沒分天時的匯出：心願單裡每個景點的基本資料＋各篇文章連結。

    跟 build_itinerary_csv() 是兩份不同的東西，不是同一份表少幾欄——沒有分天就沒有
    Day／日期／造訪順序可言，硬留空欄位反而讓打開檔案的人以為資料缺漏。這份的定位是
    「我挑好的景點清單」，所以改放類型、營業時間、期間限定活動這些挑選階段有用的欄位。

    編碼一樣用 utf-8-sig（Excel 開中文 CSV 不會亂碼），跟行程匯出一致。"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["景點名稱", "區域", "類型", "地址", "營業時間", "公休資訊",
                     "交通", "期間限定活動", "原文連結"])
    for sid in sorted(wishlist_ids, key=lambda i: spot_records.get(i, {}).get("name", "")):
        rec = spot_records.get(sid, {})
        events = "; ".join(
            f"{e['title']}（{format_event_period(e['start'], e['end'])}）"
            for e in iwafu_events_by_spot.get(sid, [])
        )
        urls = "; ".join(url for _t, _s, url in rec.get("articles", []) if url)
        writer.writerow([
            rec.get("name", ""), rec.get("neighborhood", "") or "",
            rec.get("spot_type", "") or "", rec.get("address", "") or "",
            rec.get("hours", "") or "", rec.get("closed", "") or "",
            rec.get("transit", "") or "", events, urls,
        ])
    write_event_section(writer, selected_events, spot_records)
    return output.getvalue().encode("utf-8-sig")


def build_browse_map(records):
    """建好瀏覽地圖的 folium 物件但不 render。拆出來是為了讓兩條路線共用同一份圖釘/popup
    邏輯：景點少時直接把這個物件交給 st_folium（可點擊），景點多時由下面的
    build_browse_map_html() 拿去 render 成 HTML 字串快取起來（不可點擊但快）。"""
    b_lats = [r["latitude"] for r in records]
    b_lons = [r["longitude"] for r in records]
    b_center = [sum(b_lats) / len(b_lats), sum(b_lons) / len(b_lons)]

    bm = folium.Map(location=b_center, tiles=BASEMAP_TILES, attr=BASEMAP_ATTR)
    bm.fit_bounds([[min(b_lats), min(b_lons)], [max(b_lats), max(b_lons)]])
    cluster_layer = MarkerCluster().add_to(bm)
    for rec in records:
        popup_lines = [f"<b>{rec['name']}</b>"]
        if rec.get("neighborhood"):
            popup_lines.append(rec["neighborhood"])
        popup_lines += spot_fact_lines(rec, hours_limit=MAP_HOURS_MAX_CHARS)
        consensus = truncate_for_map(consensus_summaries.get(rec["primary_id"]))
        if consensus:
            popup_lines.append(consensus)
        iwafu_category = iwafu_badges.get(rec["primary_id"])
        badge_color = IWAFU_BADGE_COLORS.get(iwafu_category) if iwafu_category else None
        # 地圖是「快速掃過」的情境，只放第一個活動的名稱＋檔期，多的用「等 N 個活動」
        # 帶過（完整清單留給卡片的 popover）。popup 是 folium 產的靜態 HTML、跑在 iframe
        # 裡，連結點下去的行為不可控，所以這裡只顯示文字不做超連結。
        spot_events = iwafu_events_by_spot.get(rec["primary_id"], [])
        if spot_events:
            first = spot_events[0]
            more = f" 等 {len(spot_events)} 個活動" if len(spot_events) > 1 else ""
            popup_lines.append(
                f"🎪 {event_category_chip(first['category'])} {first['title']}"
                f"（{format_event_period(first['start'], first['end'])}）{more}"
            )
        folium.Marker(
            location=[rec["latitude"], rec["longitude"]],
            icon=folium.DivIcon(
                html=spot_marker_html(rec.get("spot_type"), badge_color),
                icon_size=(32, 32),
                icon_anchor=(16, 16),
            ),
            popup=folium.Popup("<br>".join(popup_lines), max_width=280),
            tooltip=rec["name"],
        ).add_to(cluster_layer)
    return bm


@st.cache_data(show_spinner="正在建立地圖…")
def build_browse_map_html(neighborhoods_key, spot_types_key, themes_key, wishlist_key=None):
    """瀏覽模式地圖（866 個景點）最花時間的不是建 folium 物件（約 0.2 秒），而是
    把它 render 成 HTML（約 5 秒、1.3MB）。所以這裡直接快取「render 好的 HTML 字串」
    ——用 st.cache_data（不是 cache_resource），cache key 是排序過的區域＋景點類型
    篩選 tuple，換篩選才重算一次，其他互動（勾卡片、換分頁、搜尋）都直接吃快取。

    這條路線是給「景點多到不適合互動」的情況用的（超過 CLICKABLE_MAP_MAX_SPOTS）：
    st_folium 每次重跑都會重新 render 整張地圖（866 個圖釘實測 1.64 秒、1.3MB），而且
    快取 folium.Map 物件重用會導致「圖釘畫不出來、地圖閃現」的渲染 bug。改用
    components.html 顯示靜態 HTML，穩定又快，代價是收不到點擊事件。景點少的時候會走
    另一條路線（直接把 build_browse_map() 的物件交給 st_folium），那邊就點得動。"""
    neighborhoods = list(neighborhoods_key)
    spot_types = list(spot_types_key)
    themes = list(themes_key)
    # wishlist_key 是「只看心願單」開著時傳進來的 id tuple（關著時是 None）。一定要納入
    # 參數而不是在外面先過濾好再傳 records 進來——records 是 dict 的 list、不可 hash，
    # cache_data 沒辦法當 key；而且 cache key 少了這個條件的話，開關切換前後會拿到
    # 同一份快取好的 HTML，地圖就不會跟著變。
    wishlist_filter = set(wishlist_key) if wishlist_key is not None else None
    records = [
        rec for rec in spot_records.values()
        if (not neighborhoods or rec["neighborhood"] in neighborhoods)
        and (not spot_types or (rec.get("spot_type") or "未分類") in spot_types)
        and matches_themes(rec["primary_id"], themes)
        and (wishlist_filter is None or rec["primary_id"] in wishlist_filter)
    ]
    if not records:
        return None
    return build_browse_map(records).get_root().render()

neighborhood_counts = {}
for _rec in spot_records.values():
    if _rec["neighborhood"]:
        neighborhood_counts[_rec["neighborhood"]] = neighborhood_counts.get(_rec["neighborhood"], 0) + 1
neighborhood_counts = dict(sorted(neighborhood_counts.items(), key=lambda kv: -kv[1]))

theme_spot_counts = dict(iwafu_theme_spot_counts(iwafu_spot_categories))
spot_type_count_list = spot_type_counts(spot_records)
iwafu_event_options = load_iwafu_event_options(id_to_primary)
# 側欄的類別清單依「活動場數」由多到少排（活動分頁是這顆勾選框的主場，175 場活動比
# 31 個景點更能代表這個類別的份量）；沒有任何進行中活動的類別不列出來。
theme_event_counts = {}
for _e in iwafu_event_options:
    theme_event_counts[_e["category"]] = theme_event_counts.get(_e["category"], 0) + 1
theme_counts = sorted(theme_event_counts.items(), key=lambda kv: -kv[1])
iwafu_event_labels = [e["label"] for e in iwafu_event_options]
iwafu_label_to_event = {
    e["label"]: (e["title"], e["start"], e["end"]) for e in iwafu_event_options
}
# CSV 匯出要用到 category／venue／url，上面那份只留 (title, start, end) 給重疊期間計算用，
# 不夠用，另外存一份完整 dict 版本。
iwafu_label_to_full_event = {e["label"]: e for e in iwafu_event_options}

if "wishlist_ids" not in st.session_state:
    st.session_state.wishlist_ids = set()

# 側欄「推薦旅遊日期」那個 multiselect 的已選清單。之所以要自己管一份 session_state（而不是
# 只用 widget 回傳值），是為了讓景點卡片上的「加入日期計算」按鈕能把活動塞進去——卡片在
# 腳本後段、選單在前段，只能靠 callback 改 session_state，下一次重跑選單才讀得到。
if "wanted_event_labels" not in st.session_state:
    st.session_state.wanted_event_labels = []


def add_event_to_date_picker(label):
    """卡片上的活動 →「推薦旅遊日期」的已選清單。按鈕 callback 在下一次重跑「建立元件之前」
    執行，所以這時候改 multiselect 的 key 是合法的（腳本執行到選單那行時才會讀這個值）。"""
    if label not in st.session_state.wanted_event_labels:
        st.session_state.wanted_event_labels = st.session_state.wanted_event_labels + [label]
        st.toast("已加入「推薦旅遊日期」，可在左邊看建議日期範圍")

# 「已經在看分天結果」的開關。分群不是條件湊齊就自動觸發，而是使用者自己按按鈕決定，
# 這樣才能在「挑景點」跟「看分天結果」兩個畫面之間來回（見下方 show_planning）。
if "planning_active" not in st.session_state:
    st.session_state.planning_active = False


def set_wishlist(sid, want_in_wishlist):
    """心願單有兩份狀態要保持同步：session_state.wishlist_ids（真正的心願單集合）跟
    每張景點卡自己的 checkbox 狀態（session_state["pick_<sid>"]，Streamlit 有 key 的
    widget一旦設過值，重繪時不會再理會 value= 參數，只認自己的 session_state）。
    任何從卡片 checkbox「以外」的地方改動心願單（側欄✕移除、清空、地圖點擊加入/移除），
    都要呼叫這支函式同時寫兩邊，卡片才不會顯示跟心願單對不上的勾勾。"""
    if want_in_wishlist:
        st.session_state.wishlist_ids.add(sid)
    else:
        st.session_state.wishlist_ids.discard(sid)
    st.session_state[f"pick_{sid}"] = want_in_wishlist


def _sync_wishlist_from_card_checkbox(sid):
    """卡片 checkbox 的 on_change callback：checkbox 自己的 session_state 這時已經是
    使用者點完之後的新值，這裡只要照著更新 wishlist_ids 集合。用 on_change（而不是在
    腳本主體裡讀 picked 再寫入）是因為 callback 保證在這次 rerun「最前面」執行，
    側欄的心願單清單（畫在腳本比較前面）才看得到這次點擊剛更新的結果，不會慢一拍。"""
    if st.session_state.get(f"pick_{sid}"):
        st.session_state.wishlist_ids.add(sid)
    else:
        st.session_state.wishlist_ids.discard(sid)

with st.sidebar:
    # 放在側欄最上面，原因是「載入」要在天數/心願單這些 widget 被建立之前，先把
    # session_state 寫好——Streamlit 規則是同一次 rerun 裡，widget 一旦用某個 key
    # 畫出來，就不能再改那個 key 的 session_state（會丟例外）。因為載入後緊接著
    # st.rerun()，理論上放哪裡都能生效，但放最上面比較保險、也不用擔心以後有人
    # 把這段搬到 wishlist/天數 widget 後面時忘記這個限制。
    st.markdown("#### 已儲存計畫")
    saved_plans = load_saved_plans_from_db()
    if not saved_plans:
        st.caption("還沒有儲存的計畫。排好行程後可以命名儲存，之後隨時回來載入。")
    else:
        for plan in saved_plans:
            plan_payload = json.loads(plan["data"])
            event_count = len(plan_payload.get("event_labels", []))
            event_suffix = f"｜{event_count}個活動" if event_count else ""
            sp_col1, sp_col2, sp_col3 = st.columns([3, 1, 1])
            sp_col1.markdown(
                f"<span style='font-size:13px'>{plan['name']}</span><br>"
                f"<span style='font-size:11px;color:#888'>{plan['created_at']}｜"
                f"{plan_payload['day_count']}天｜{len(plan_payload['wishlist_ids'])}個景點{event_suffix}</span>",
                unsafe_allow_html=True,
            )
            if sp_col2.button("載入", key=f"load_plan_{plan['id']}", use_container_width=True):
                for sid in list(st.session_state.wishlist_ids):
                    set_wishlist(sid, False)
                for sid in plan_payload["wishlist_ids"]:
                    set_wishlist(sid, True)
                st.session_state["day_count"] = plan_payload["day_count"]
                st.session_state["start_date"] = date.fromisoformat(plan_payload["start_date"])
                # 用 iwafu_event_labels 過濾一次：活動可能在存檔之後已經結束、被
                # load_iwafu_event_options() 濾掉不再是選單選項，直接塞回去會跟目前的
                # multiselect options 對不上（Streamlit 會噴錯），所以只還原還存在的部分。
                saved_event_labels = plan_payload.get("event_labels", [])
                st.session_state["wanted_event_labels"] = [
                    l for l in saved_event_labels if l in iwafu_event_labels
                ]
                # 存下來的計畫本來就是「規劃完成」的狀態，載入後直接進分天畫面，不用再按
                # 一次「分天規劃」——使用者要的是還原當時的結果，不是回到挑景點的階段。
                st.session_state.planning_active = True
                st.toast(f"已載入「{plan['name']}」")
                st.rerun()
            if sp_col3.button("刪除", key=f"delete_plan_{plan['id']}", use_container_width=True):
                delete_saved_plan_from_db(plan["id"])
                st.toast(f"已刪除「{plan['name']}」")
                st.rerun()
    st.divider()

    st.markdown("#### 區域篩選（地名）")
    nb_labels = [f"{n}（{c}）" for n, c in neighborhood_counts.items()]
    label_to_nb = {f"{n}（{c}）": n for n, c in neighborhood_counts.items()}
    selected_labels = st.multiselect(
        "區域篩選（地名）", nb_labels, default=[], label_visibility="collapsed"
    )
    selected_neighborhoods = [label_to_nb[l] for l in selected_labels]

    st.markdown("#### 景點類型")
    selected_spot_types = []
    for cat, cnt in spot_type_count_list:
        emoji = SPOT_TYPE_ICONS.get(cat, SPOT_TYPE_ICONS["未分類"])[0]
        if st.checkbox(f"{emoji} {cat}（{cnt}）", value=False, key=f"spot_type_{cat}"):
            selected_spot_types.append(cat)

    st.markdown("#### 期間限定活動")
    selected_themes = []
    for cat, ev_cnt in theme_counts:
        slug = IWAFU_CATEGORY_SLUGS.get(cat, cat)
        label = iwafu_theme_label(cat, ev_cnt, theme_spot_counts.get(cat, 0))
        if st.checkbox(label, value=False, key=f"theme_{slug}"):
            selected_themes.append(cat)
    st.caption("前面的數字是這一類有幾場活動，後面是有這類活動的景點有幾個——"
               "勾選會同時篩選「期間限定活動」和「景點」兩個分頁。")

    st.markdown("#### 推薦旅遊日期")
    selected_event_labels = st.multiselect(
        "想參加的期間限定活動", iwafu_event_labels, label_visibility="collapsed",
        placeholder="點此展開活動清單，或輸入關鍵字搜尋（可留空）",
        key="wanted_event_labels",
    )
    # 「只看已選的活動」放側欄而不是活動分頁，跟下面的「只看心願單」擺在一起：兩個都是
    # 「只看我選的東西」，同一個概念的控制項放同一區比較好找。沒選任何活動時不顯示，
    # 免得勾了得到一個空清單。
    only_selected_events = False
    if selected_event_labels:
        only_selected_events = st.checkbox(
            f"只看已選的 {len(selected_event_labels)} 個活動", value=False, key="only_selected_events",
        )
    if selected_event_labels:
        selected_events = [iwafu_label_to_event[l] for l in selected_event_labels]
        # 選 1 個活動、跟選多個但有共同交集，算出來的都是「一段能參加全部活動的區間」，
        # 下面要拿它跟使用者的行程比對，所以兩個分支都存到同一個變數，不各寫一份比對。
        common_window = None
        window_desc = None
        if len(selected_events) == 1:
            _title, start, end = selected_events[0]
            common_window = (start, end)
            # 活動名稱本身常常就以「」開頭（例如「星際大戰」CAFE : 銀河餐館），再包一層
            # 會變成「「星際大戰」CAFE…」這種雙層括號，看起來像 bug。
            _title_txt = _title if _title.startswith("「") else f"「{_title}」"
            window_desc = f"{_title_txt}的舉辦期間落在 {start:%Y/%m/%d}～{end:%Y/%m/%d}"
        else:
            full_overlap = find_full_overlap([(s, e) for _t, s, e in selected_events])
            if full_overlap:
                common_window = full_overlap
                # 不寫「都能參加」——那是使用者自己安排的事，這裡只陳述這幾個活動的
                # 檔期共同落在哪一段。
                window_desc = (
                    f"選的 {len(selected_events)} 個活動共同期間落在 "
                    f"{full_overlap[0]:%Y/%m/%d}～{full_overlap[1]:%Y/%m/%d}"
                )
            else:
                pairwise = find_pairwise_overlaps(selected_events)
                if pairwise:
                    st.warning(f"這 {len(selected_events)} 個活動沒有全部都重疊的共同期間，"
                               "但部分兩兩之間仍有重疊，可以考慮取捨：")
                    for t1, t2, ov_start, ov_end in pairwise:
                        st.caption(f"・「{t1}」×「{t2}」：{ov_start:%Y/%m/%d} ～ {ov_end:%Y/%m/%d}")
                else:
                    st.warning(f"這 {len(selected_events)} 個活動彼此的舉辦期間完全沒有重疊，"
                               "沒辦法一趟行程全部參加，可能需要分成兩趟或取捨其中幾個：")
                for _t, s, e in selected_events:
                    st.caption(f"・「{_t}」：{s:%Y/%m/%d} ～ {e:%Y/%m/%d}")
                # 沒有共同區間時不比對行程：畫面上已經在講「要取捨哪幾個活動」，這時候再
                # 插一行行程日期只是雜訊——得先決定要參加哪些，才有「該挪到哪天」的問題。
        if common_window:
            # 天數／出發日的 widget 排在這個區塊「下面」，還沒建立、拿不到回傳值，所以改讀
            # session_state。這樣讀不會慢一拍：Streamlit 是「先把 widget 新值寫進 session_state、
            # 再從頭重跑腳本」，所以腳本前段讀到的已經是這次互動後的值。第一次載入時兩個 key
            # 還不存在，用跟 widget 一樣的預設值（天數 0＝還沒設行程，不比對）。
            _level, _msg = describe_trip_vs_window(
                st.session_state.get("day_count", 0),
                st.session_state.get("start_date", date.today()),
                common_window[0], common_window[1], window_desc,
            )
            # 綠框＝行程已落在活動期間內（或還沒設行程），黃框＝完全沒重疊。
            (st.success if _level == "ok" else st.warning)(_msg)
    st.caption("選幾個想參加的活動，會算出都能參加的日期範圍。")

    st.markdown("#### 天數")
    day_count = st.number_input(
        "天數", min_value=0, max_value=6, value=0, step=1, label_visibility="collapsed", key="day_count"
    )

    st.markdown("#### 行程開始日期")
    start_date = st.date_input(
        "行程開始日期", value=date.today(), label_visibility="collapsed", key="start_date"
    )
    st.caption("行程從這天開始，也能檢查行程是否會遇到公休。")

    wl_head_col1, wl_head_col2 = st.columns([3, 1])
    wl_head_col1.markdown(f"#### 心願單（{len(st.session_state.wishlist_ids)}）")
    if st.session_state.wishlist_ids and wl_head_col2.button("清空", key="clear_wishlist", use_container_width=True):
        for sid in list(st.session_state.wishlist_ids):
            set_wishlist(sid, False)
        st.rerun()
    # 這顆一定要放側欄，不能放景點分頁裡：Streamlit 由上而下跑，瀏覽地圖（腳本前段）
    # 讀不到分頁（腳本後段）才建立的 widget，勾了地圖不會跟著變，要等下一次互動才更新
    # ——就是心願單那次踩過的一拍延遲。放在側欄，地圖跟卡片同一次重跑就都吃得到。
    only_wishlist = False
    if st.session_state.wishlist_ids:
        only_wishlist = st.checkbox("只看心願單", value=False, key="only_wishlist")
    if not st.session_state.wishlist_ids:
        st.caption("還沒有選景點，到右邊的景點卡片勾選。")
    else:
        for sid in sorted(st.session_state.wishlist_ids, key=lambda i: spot_records[i]["name"]):
            rec = spot_records[sid]
            wc1, wc2 = st.columns([3, 1])
            wc1.markdown(f"<span style='font-size:13px'>{rec['name']}</span>", unsafe_allow_html=True)
            if wc2.button("✕", key=f"remove_{sid}", help="從心願單移除", use_container_width=True):
                set_wishlist(sid, False)
                st.rerun()

st.title("旅遊前置決策與景點分群系統")
st.caption(
    f"{'、'.join(selected_neighborhoods) if selected_neighborhoods else '瀏覽全部區域'}　｜　"
    f"{f'{day_count} 天行程規劃中' if day_count else '尚未選擇天數'}"
)

wishlist_ids = sorted(st.session_state.wishlist_ids)
# 規劃模式才算得出「排好天數與順序」的行程 CSV；瀏覽模式改匯出心願單清單（見側欄最下方的
# 「行程操作」區塊，那一塊兩種模式都會出現，只是匯出的內容不同）。
itinerary_csv = None
# 「推薦旅遊日期」選單目前選了哪些活動，規劃/瀏覽兩種匯出模式跟「儲存計畫」都要用同一份，
# 在這裡算一次共用。
selected_event_objs = [
    iwafu_label_to_full_event[l] for l in st.session_state.wanted_event_labels
    if l in iwafu_label_to_full_event
]

st.subheader("地圖總覽・KMeans 分組結果")
st.caption("點地圖上的圖釘可以看景點詳情。")

planning_ready = day_count > 0 and wishlist_ids and len(wishlist_ids) >= day_count

# 分群改成「按鈕觸發」而不是「條件湊齊就自動跳走」（使用者實際操作後提的）：原本天數
# ＋心願單一到位畫面就直接切成分天結果，想回頭多逛幾個景點只能把天數改回 0，心願單
# 雖然留著但操作很反直覺。改成 planning_active 這個開關後，兩個畫面可以隨時來回，
# 心願單／天數／行程開始日期全程保留。條件不足時強制關掉開關，避免停在一個算不出
# 結果的規劃畫面（例如在規劃模式裡把心願單刪到比天數少）。
if not planning_ready:
    st.session_state.planning_active = False
show_planning = planning_ready and st.session_state.planning_active

if not show_planning:
    if planning_ready:
        st.success(
            f"心願單已有 {len(wishlist_ids)} 個景點、天數 {day_count} 天，可以開始分組了。"
            "也可以先繼續逛景點，選好再按下面的按鈕。"
        )
        if st.button("分天規劃", type="primary", key="start_planning"):
            st.session_state.planning_active = True
            st.rerun()
    elif day_count == 0:
        st.info("先在左邊選擇天數（至少 1 天），才能開始分天規劃。現在可以先在地圖上瀏覽全部景點。")
    elif not wishlist_ids:
        st.info("先在下方的景點卡片勾選想去的地方，心願單至少要有 1 個景點才能開始分天規劃。")
    else:
        st.warning(f"景點數（{len(wishlist_ids)}）比天數（{day_count}）少，無法分成 {day_count} 天，請減少天數或多選幾個景點。")

    # 瀏覽模式：套用側欄的區域篩選（跟下面景點卡片一致），但不分天／不跑 KMeans，
    # 讓使用者在還沒決定天數＋心願單之前，先在地圖上用滑鼠逛過全部候選景點。
    # 866 個 CircleMarker + popup 字串是這個頁面最花時間的部分，用 cache_resource
    # 存住建好的地圖物件——不然 Streamlit 每次互動（勾選卡片、換分頁…跟地圖完全
    # 無關的操作）都會整支重跑，把 866 個圖釘重新蓋一次、送一次 1MB+ 的 HTML 給
    # 瀏覽器，這是先前「載入/點擊都很慢」的主因。只有區域篩選真的改變時才重建。
    map_browse_records = [
        rec for rec in spot_records.values()
        if (not selected_neighborhoods or rec["neighborhood"] in selected_neighborhoods)
        and (not selected_spot_types or (rec.get("spot_type") or "未分類") in selected_spot_types)
        and matches_themes(rec["primary_id"], selected_themes)
        and (not only_wishlist or rec["primary_id"] in st.session_state.wishlist_ids)
    ]
    if not map_browse_records:
        st.warning(
            "「只看心願單」開著，但心願單裡的景點都被其他篩選條件排除了——"
            "清掉區域／類型／活動篩選就會出現。" if only_wishlist
            else "目前篩選條件下沒有符合的景點。"
        )
    elif len(map_browse_records) <= CLICKABLE_MAP_MAX_SPOTS:
        # 景點夠少：用 st_folium，點圖釘就能直接加入心願單（跟規劃模式那張地圖同一套做法）。
        st.caption(
            f"目前顯示 {len(map_browse_records)} 個景點"
            + ("（只看心願單），點圖釘可以看詳情，也能直接移出心願單。" if only_wishlist
               else "，點圖釘可以看詳情，也能直接加入心願單。")
        )
        browse_state = st_folium(
            build_browse_map(map_browse_records),
            use_container_width=True, height=520, key="browse_map",
        )
        browse_clicked = (browse_state or {}).get("last_object_clicked")
        if browse_clicked and browse_clicked.get("lat") is not None:
            nearest_browse = min(
                map_browse_records,
                key=lambda r: (r["latitude"] - browse_clicked["lat"]) ** 2
                + (r["longitude"] - browse_clicked["lng"]) ** 2,
            )
            nb_id = nearest_browse["primary_id"]
            nb_in = nb_id in st.session_state.wishlist_ids
            bclick_col1, bclick_col2 = st.columns([4, 1])
            bclick_col1.markdown(f"📍 你點的是：**{nearest_browse['name']}**")
            if bclick_col2.button(
                "從心願單移除" if nb_in else "加入心願單", key="browse_click_toggle"
            ):
                set_wishlist(nb_id, not nb_in)
                st.rerun()
    else:
        # 景點太多：退回快取好的靜態 HTML，快但收不到點擊，並告訴使用者怎麼換到可點的模式。
        map_html = build_browse_map_html(
            tuple(sorted(selected_neighborhoods)), tuple(sorted(selected_spot_types)),
            tuple(sorted(selected_themes)),
            tuple(sorted(st.session_state.wishlist_ids)) if only_wishlist else None,
        )
        st.caption(
            f"目前顯示 {len(map_browse_records)} 個景點，圖釘依縮放層級自動群聚，點圖釘可以看詳情。"
            f"用側欄篩到 {CLICKABLE_MAP_MAX_SPOTS} 個以內，就能直接點圖釘加入心願單。"
        )
        components.html(map_html, height=520)
else:
    # 回到瀏覽模式繼續挑景點：心願單／天數／日期都留著，等下再按一次「分天規劃」就會
    # 用最新的心願單重算（KMeans 的 random_state 固定，同一份心願單結果一定一樣）。
    if st.button("← 回到瀏覽地圖繼續挑景點", key="back_to_browse"):
        st.session_state.planning_active = False
        st.rerun()

    raw_days, _raw_warnings = cluster_by_days(wishlist_ids, day_count)

    # KMeans 給的 Day 編號是它自己的標籤順序，本身沒有意義——哪一群被叫做 Day1 是隨機的，
    # 使用者卻很可能想自己決定「哪一天去哪一區」（例如把有週一公休景點的那一區排到週二）。
    # 所以在這裡套一層順序對應：day_order[i] 是要顯示成 Day(i+1) 的那個分群編號。分群內容
    # 一變（心願單或天數改了）對應就沒意義了，用 signature 比對後重置成預設順序。
    order_signature = (tuple(wishlist_ids), day_count)
    if st.session_state.get("day_order_signature") != order_signature:
        st.session_state.day_order = list(raw_days.keys())
        st.session_state.day_order_signature = order_signature
    day_order = st.session_state.day_order
    days = {display_day: raw_days[cid] for display_day, cid in enumerate(day_order, 1)}

    # 單日超量警示原本由 kmeans_clustering 產生，但它用的是原始分群編號，重排之後 Day 編號
    # 會對不上，所以改用重排後的 days 重新產生一次（門檻仍用模組匯出的常數，不另外定義）。
    warnings = [
        f"Day{day} 有 {len(group)} 個景點，超過建議上限（{DAILY_SPOT_WARNING_THRESHOLD}），請規劃時注意時間安排"
        for day, group in days.items() if len(group) > DAILY_SPOT_WARNING_THRESHOLD
    ]
    for w in warnings:
        st.warning(w)

    hotel_rec = recommend_hotel_for_trip(days)

    day_dates = {d: start_date + timedelta(days=d - 1) for d in days}

    # 公休警示：每個景點的公休文字 vs 該天實際日期，紅燈＝確定衝突、黃燈＝規則看不懂要自己查證
    conflicts = {}
    closed_warnings = []
    unknown_closed = []
    for day, group in days.items():
        visit_date = day_dates[day]
        weekday_label = f"{visit_date:%m/%d}（週{WEEKDAY_ZH[visit_date.weekday()]}）"
        for s in group:
            closed_text = spot_records.get(s["id"], {}).get("closed")
            parsed = parse_closed_days(closed_text)
            result = check_conflict(parsed, visit_date)
            conflicts[s["id"]] = result
            if result == "red":
                closed_warnings.append(
                    ("red", f"🔴 Day{day}（{weekday_label}）**{s['name']}** 公休衝突：{closed_text}")
                )
            elif result == "yellow":
                closed_warnings.append(
                    ("yellow", f"🟡 Day{day}（{weekday_label}）**{s['name']}** 公休資訊無法自動判讀（{closed_text}），請自行查證")
                )
            elif parsed.get("status") == "none":
                unknown_closed.append(s["name"])
    for level, msg in closed_warnings:
        (st.error if level == "red" else st.warning)(msg)
    # 查無公休資訊的景點（全庫 725/888 筆）不在卡片跟地圖上逐筆印那句 fallback 文案，
    # 改成在這裡統整提醒一次——只針對使用者真的排進行程的那十幾個景點，出現在「已經
    # 決定要去、要排哪天」的決策時點，而不是在瀏覽 888 張卡片時每張都佔一行。
    # 用黃字（不是紅字）：這不是「確定衝突」，是「我們不知道」，跟🟡無法判讀同一個層級。
    if unknown_closed:
        st.warning(
            f"🟡 這 {len(unknown_closed)} 個景點查無公休資訊：{'、'.join(unknown_closed)}，"
            "建議出發前向店家確認。"
        )

    all_lats = [s["latitude"] for group in days.values() for s in group]
    all_lons = [s["longitude"] for group in days.values() for s in group]
    center = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]

    m = folium.Map(location=center, tiles=BASEMAP_TILES, attr=BASEMAP_ATTR)
    m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])
    for day, group in days.items():
        color = DAY_COLORS[(day - 1) % len(DAY_COLORS)]
        for s in group:
            rec = spot_records.get(s["id"], {})
            closed_text = rec.get("closed")
            result = conflicts.get(s["id"])
            if result == "red":
                stroke, stroke_width, status_line = "#A8412F", 5, "🔴 公休衝突"
            elif result == "yellow":
                stroke, stroke_width, status_line = "#B5651D", 4, "🟡 公休資訊需自行查證"
            else:
                stroke, stroke_width, status_line = color, 2, ""
            neighborhood = rec.get("neighborhood") or s["area"]
            popup_lines = [f"<b>{s['name']}</b>", f"Day {day}｜{neighborhood}"]
            popup_lines += spot_fact_lines(rec, closed_text, hours_limit=MAP_HOURS_MAX_CHARS)
            if status_line:
                popup_lines.append(status_line)
            consensus = truncate_for_map(consensus_summaries.get(s["id"]))
            if consensus:
                popup_lines.append(consensus)
            # 跟瀏覽模式地圖同一行格式：兩張地圖對同一個景點顯示的資訊要一致，不然
            # 使用者在瀏覽地圖看得到活動、切到規劃地圖活動就消失，會以為資料掉了。
            plan_events = iwafu_events_by_spot.get(s["id"], [])
            if plan_events:
                first_ev = plan_events[0]
                more_ev = f" 等 {len(plan_events)} 個活動" if len(plan_events) > 1 else ""
                popup_lines.append(
                    f"🎪 {event_category_chip(first_ev['category'])} {first_ev['title']}"
                    f"（{format_event_period(first_ev['start'], first_ev['end'])}）{more_ev}"
                )
            folium.CircleMarker(
                location=[s["latitude"], s["longitude"]],
                radius=10,
                color=stroke,
                weight=stroke_width,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=folium.Popup("<br>".join(popup_lines), max_width=280),
                tooltip=f"Day{day}｜{s['name']}" + (f"｜{status_line}" if status_line else ""),
            ).add_to(m)

    # 東京10大轉運站：固定標註在地圖上（提案表格四／五要求），圖示跟景點圖釘明顯區分
    # （灰色火車圖示 vs 彩色圓點），popup 附上「離整趟行程中心點的距離」讓使用者知道
    # 這站對這趟行程是不是方便，不只是單純的地標裝飾。飯店推薦是整趟一間（不是逐天
    # 各推薦一次，見hotel_recommendation.py），這裡的距離基準跟著改成同一個中心點。
    trip_center = hotel_rec["centroid"]
    for station_name, (slat, slon) in TOKYO_HUB_STATIONS.items():
        dist_to_trip = haversine_km(slat, slon, trip_center[0], trip_center[1])
        folium.Marker(
            location=[slat, slon],
            icon=folium.Icon(color="gray", icon="train", prefix="fa"),
            popup=folium.Popup(
                f"<b>{station_name}</b><br>離整趟行程中心點 {dist_to_trip:.2f} km", max_width=200
            ),
            tooltip=station_name,
        ).add_to(m)

    map_state = st_folium(m, use_container_width=True, height=520, key="map")

    # 地圖上直接加入/移出心願單：folium popup 是純 HTML，塞不進真的會動的 Streamlit
    # 元件，所以做法是讀 st_folium 回傳的「最後點擊的圖釘座標」，反查最接近的景點，
    # 在地圖下方浮出一個小按鈕區——體感接近「點地圖就能選」，只是操作區在地圖下面。
    clicked = (map_state or {}).get("last_object_clicked")
    if clicked and clicked.get("lat") is not None:
        all_group_spots = [s for group in days.values() for s in group]
        nearest = min(
            all_group_spots,
            key=lambda s: (s["latitude"] - clicked["lat"]) ** 2 + (s["longitude"] - clicked["lng"]) ** 2,
        )
        in_wishlist = nearest["id"] in st.session_state.wishlist_ids
        click_col1, click_col2 = st.columns([4, 1])
        click_col1.markdown(f"📍 你點的是：**{nearest['name']}**")
        toggle_label = "從心願單移除" if in_wishlist else "加入心願單"
        if click_col2.button(toggle_label, key="map_click_toggle"):
            set_wishlist(nearest["id"], not in_wishlist)
            st.rerun()

    # Day 摘要列：固定兩天一行（不管天數多少都一樣，使用者要求維持規律，不要天數不同
    # 排法就跟著變），比原本橫向 st.columns(len(days))等分6欄寬得多，區域文字才有地方
    # 顯示，不會被截斷。
    def _cluster_label(cluster_id):
        """選單顯示文字＝那一群的地名摘要＋景點數。選單的「值」用分群編號而不是這串文字，
        因為兩群的地名摘要有可能一模一樣（同一區被分成兩天），用編號當值就不會撞在一起。"""
        group = raw_days[cluster_id]
        nb = day_neighborhood_summary(group) or "未標示區域"
        return f"{nb}（{len(group)} 個景點）"

    def _swap_day_cluster(display_day):
        """選到別天的區域就跟那天「對調」，不是覆蓋——對調保證任何操作後 day_order 都還是
        一個合法的排列（每個分群剛好出現一次），不會有兩天指到同一群或某群消失的情況。"""
        chosen = st.session_state[f"day_pick_{display_day}"]
        order = st.session_state.day_order
        current = order[display_day - 1]
        if chosen == current:
            return
        other_pos = order.index(chosen)
        order[display_day - 1], order[other_pos] = chosen, current

    # 選單的值必須在建立元件之前就跟 day_order 同步：對調是在 callback 裡改 day_order，
    # 另一天那顆 selectbox 自己記住的還是舊值，不同步的話畫面上會出現兩天選到同一區。
    for display_day, cluster_id in enumerate(day_order, 1):
        st.session_state[f"day_pick_{display_day}"] = cluster_id

    day_items = list(days.items())
    cols_per_row = 2
    for row_start in range(0, len(day_items), cols_per_row):
        row_days = day_items[row_start:row_start + cols_per_row]
        legend_cols = st.columns(cols_per_row)
        for col, (day, group) in zip(legend_cols, row_days):
            color = DAY_COLORS[(day - 1) % len(DAY_COLORS)]
            visit_date = day_dates[day]
            col.markdown(
                f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
                f'background:{color};margin-right:6px;"></span>'
                f"**Day {day}**（{visit_date:%m/%d} 週{WEEKDAY_ZH[visit_date.weekday()]}）｜{len(group)} 個景點",
                unsafe_allow_html=True,
            )
            # 地名不再寫死在上面那行，改由這個選單顯示，順便就能換成別天的區域。
            col.selectbox(
                f"Day {day} 要去哪一區",
                options=list(raw_days.keys()),
                format_func=_cluster_label,
                key=f"day_pick_{day}",
                on_change=_swap_day_cluster,
                args=(day,),
                label_visibility="collapsed",
            )
    st.caption("每天的區域可以自己換：選單選到別天的區域就會跟那天對調，日期、公休檢查、路線建議都會跟著重算。")

    st.subheader("住宿建議")
    st.caption("以整趟行程（全部天數的景點）算一個地理中心點，找最近的東京轉運站，距離在 3 km 內才推薦；"
               "整趟只推薦一間，不是逐天各推薦一次。行程分散的話系統不硬猜，交由你自行斟酌住宿地點。")
    if hotel_rec["recommended"]:
        st.success(f"✅ 建議整趟入住 **{hotel_rec['station']}** 附近（行程中心點距離 {hotel_rec['distance_km']:.2f} km）")
    else:
        st.warning(f"⚠️ 行程較分散，最近的 **{hotel_rec['station']}** 也有 {hotel_rec['distance_km']:.2f} km，建議自行斟酌住宿地點或考慮分開找旅宿")

    st.subheader("路線建議")
    route_day = st.selectbox(
        "選擇要排路線的日期",
        options=list(days.keys()),
        format_func=lambda d: f"Day {d}（{day_dates[d]:%m/%d} 週{WEEKDAY_ZH[day_dates[d].weekday()]}）",
    )
    day_spots = days[route_day]
    start_id = st.selectbox(
        "起點（使用者指定）",
        options=[s["id"] for s in day_spots],
        format_func=lambda sid: next(s["name"] for s in day_spots if s["id"] == sid),
    )
    route, leg_distances = greedy_route(day_spots, start_id)

    cols = st.columns(len(route))
    for i, (col, s) in enumerate(zip(cols, route)):
        with col:
            result = conflicts.get(s["id"])
            prefix = "🔴 " if result == "red" else ("🟡 " if result == "yellow" else "")
            st.markdown(f"**{i + 1}. {prefix}{s['name']}**")
            if i < len(leg_distances):
                dist = leg_distances[i]
                if dist > LONG_LEG_WARNING_KM:
                    st.caption(
                        f"⚠️ 下一站 {dist:.2f} km，距離較遠，建議查詢交通方式，可對照"
                        f"<a href='{TOKYO_METRO_MAP_URL}' target='_blank'>東京地鐵路線圖</a>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(f"→ 下一站 {dist:.2f} km")
    st.caption(f"總距離：{sum(leg_distances):.2f} km（Haversine 直線估算，非實際步行路徑，示意用途）")

    itinerary_csv = build_itinerary_csv(days, day_dates, conflicts, spot_records, route_day, route, selected_event_objs)

# 景點跟活動分成兩個分頁：兩者是不同的挑選行為（挑地點 vs 挑檔期），混在同一個捲軸裡
# 會讓頁面過長也更難用。分頁標題直接把數量寫出來，活動那頁才不會被忽略——175 場活動裡
# 有 134 場沒配對到景點，在這個分頁出現之前，它們只存在於側欄那個要捲 175 筆、沒有連結
# 的下拉選單裡，實質上等於用不到。
trip_start = start_date
trip_end = start_date + timedelta(days=day_count - 1) if day_count else None
events_in_trip = [
    e for e in iwafu_event_options
    if trip_end and e["start"] <= trip_end and e["end"] >= trip_start
]
spot_tab, event_tab = st.tabs(["景點", f"期間限定活動（{len(iwafu_event_options)}）"])

with event_tab:
    st.caption("點活動名稱可以連到活動原頁看完整內容。「加入推薦旅遊日期」會把它加進左邊的日期計算。")
    # 勾選框跟搜尋框各佔一整行，不並排：並排時左邊那行字（「只顯示行程期間（08/03–08/05）
    # 的活動」）會被擠成兩行，反而更難讀，省下的那點高度不划算。
    only_in_trip = False
    if trip_end:
        only_in_trip = st.checkbox(
            f"只顯示行程期間（{trip_start:%m/%d}–{trip_end:%m/%d}）的活動",
            value=True, key="events_only_in_trip",
            # 側欄開了「只看已選的活動」時，這顆會被忽略（見下方 base_events），
            # disable 起來比讓它看起來還能生效誠實。
            disabled=only_selected_events,
        )
    else:
        st.caption("選了天數與行程開始日期之後，這裡可以只顯示行程期間內的活動。")
    if only_selected_events:
        st.caption("目前只顯示側欄已選的活動，不受「行程期間」限制——"
                   "檔期落在行程之外的活動也會列出來，才看得出要不要為它調整日期。")
    event_search = st.text_input(
        "搜尋活動名稱", placeholder="輸入活動名稱關鍵字（可留空）", label_visibility="collapsed",
        key="event_search",
    )

    # 「只看已選的活動」時故意不套行程期間篩選：選活動的用意就是判斷「要不要為了它挪日期」，
    # 而想挪日期的活動檔期本來就多半落在現在設定的行程之外（例如行程 8/3–8/5、活動 8/6 才開始）。
    # 兩個條件相乘的話，最該被看到的那幾筆會直接消失。
    base_events = iwafu_event_options if only_selected_events else (
        events_in_trip if only_in_trip else iwafu_event_options
    )
    # 類別篩選沿用側欄那組 checkbox，不在這裡再放一組：勾「展覽」時，景點分頁篩出「有展覽
    # 活動的景點」、活動分頁篩出「展覽類活動」，同一個勾選在兩邊語意一致。
    shown_events = [
        e for e in base_events
        if (not selected_themes or e["category"] in selected_themes)
        and (not event_search or event_search in e["title"])
        and (not only_selected_events or e["label"] in selected_event_labels)
    ]

    EVENT_PAGE_SIZE = 20
    event_signature = (
        only_in_trip, tuple(selected_themes), event_search, str(trip_end),
        tuple(selected_event_labels) if only_selected_events else None,
    )
    if st.session_state.get("event_filter_signature") != event_signature:
        st.session_state.event_page = 1
        st.session_state.event_filter_signature = event_signature
    ev_total_pages = max(1, (len(shown_events) + EVENT_PAGE_SIZE - 1) // EVENT_PAGE_SIZE)
    st.session_state.event_page = min(st.session_state.get("event_page", 1), ev_total_pages)

    ev_nav1, ev_nav2 = st.columns([4, 1], vertical_alignment="center")
    ev_nav1.caption(f"符合條件 {len(shown_events)} 場活動，第 {st.session_state.event_page}／{ev_total_pages} 頁")
    ev_prev, ev_next = ev_nav2.columns(2, gap="small")
    if ev_prev.button("← 上一頁", key="event_prev", disabled=st.session_state.event_page <= 1,
                      use_container_width=True):
        st.session_state.event_page -= 1
        st.rerun()
    if ev_next.button("下一頁 →", key="event_next",
                      disabled=st.session_state.event_page >= ev_total_pages, use_container_width=True):
        st.session_state.event_page += 1
        st.rerun()

    if not shown_events:
        st.info(
            "目前條件下沒有符合的活動，可以取消側欄的「只看已選的活動」或清掉類別篩選。"
            if only_selected_events
            else "目前條件下沒有符合的活動，可以取消「只顯示行程期間」或清掉左邊的類別篩選。"
        )
    ev_start = (st.session_state.event_page - 1) * EVENT_PAGE_SIZE
    ev_page = shown_events[ev_start:ev_start + EVENT_PAGE_SIZE]
    ev_cols = st.columns(2)
    for ev_idx, ev in enumerate(ev_page):
        with ev_cols[ev_idx % 2].container(border=True, key=f"event_card_{ev_start + ev_idx}"):
            title_html = (
                f"<a href='{ev['url']}' target='_blank'>{ev['title']}</a>" if ev["url"] else ev["title"]
            )
            st.markdown(
                f"{event_category_chip(ev['category'])} {title_html}", unsafe_allow_html=True
            )
            st.caption(f"🗓 {format_event_period(ev['start'], ev['end'])}")
            if ev["venue"]:
                st.caption(f"📍 {ev['venue']}")
            # 配對到景點的活動（41/175）多給一個入口：直接把那個景點加進心願單，不用自己
            # 回景點分頁再找一次。配不到的就只顯示會場文字——那些地點本來就不在景點庫裡。
            spot_id = ev.get("spot_id")
            if spot_id is not None and spot_id in spot_records:
                spot_name = spot_records[spot_id]["name"]
                if spot_id in st.session_state.wishlist_ids:
                    st.caption(f"✓ 地點「{spot_name}」已在心願單")
                else:
                    st.button(
                        f"把「{spot_name}」加入心願單",
                        key=f"event_spot_{ev_start + ev_idx}",
                        on_click=set_wishlist, args=(spot_id, True),
                        use_container_width=True,
                    )
            if ev["label"] in st.session_state.wanted_event_labels:
                st.caption("✓ 已加入推薦旅遊日期")
            else:
                st.button(
                    "加入推薦旅遊日期",
                    key=f"event_date_{ev_start + ev_idx}",
                    on_click=add_event_to_date_picker, args=(ev["label"],),
                    use_container_width=True,
                )

spot_tab_ctx = spot_tab.container()
with spot_tab_ctx:
    st.caption("每篇提到這個景點的文章都會列出標題（可點連結）＋約兩行摘要，方便先看摘要挑要點開哪篇。")
    search_text = st.text_input("搜尋景點名稱", placeholder="輸入店名關鍵字（可留空，直接瀏覽目前篩選的區域）")

    browse_records = [
        rec for rec in spot_records.values()
        if (not selected_neighborhoods or rec["neighborhood"] in selected_neighborhoods)
        and (not selected_spot_types or (rec.get("spot_type") or "未分類") in selected_spot_types)
        and matches_themes(rec["primary_id"], selected_themes)
        and (not search_text or search_text in rec["name"])
        and (not only_wishlist or rec["primary_id"] in st.session_state.wishlist_ids)
    ]
    browse_records.sort(key=lambda r: r["name"])

    # 分頁：篩選後可能有到 866 張卡片，一次全部渲染會讓頁面明顯變慢，改成每頁固定張數＋
    # 上一頁/下一頁；篩選條件（區域／類型／活動／搜尋字）一變就跳回第 1 頁，避免卡在一個空頁面。
    CARD_PAGE_SIZE = 24
    filter_signature = (
        tuple(selected_neighborhoods), tuple(selected_spot_types), tuple(selected_themes), search_text,
        # 「只看心願單」也算篩選條件的一部分：切換後符合的筆數會大幅變動，沒重置頁碼的話
        # 會停在一個超出範圍的空白頁。心願單內容本身變了也一樣要重置（開著的時候才算進去，
        # 關著時把它納入 signature 會變成「勾任何一張卡片都跳回第 1 頁」）。
        tuple(sorted(st.session_state.wishlist_ids)) if only_wishlist else None,
    )
    if st.session_state.get("card_filter_signature") != filter_signature:
        st.session_state.card_page = 1
        st.session_state.card_filter_signature = filter_signature

    total_cards = len(browse_records)
    total_pages = max(1, (total_cards + CARD_PAGE_SIZE - 1) // CARD_PAGE_SIZE)
    st.session_state.card_page = min(st.session_state.get("card_page", 1), total_pages)

    page_nav_col1, page_nav_col2 = st.columns([4, 1], vertical_alignment="center")
    page_nav_col1.caption(f"符合條件 {total_cards} 個景點，第 {st.session_state.card_page}／{total_pages} 頁")
    prev_col, next_col = page_nav_col2.columns(2, gap="small")
    if prev_col.button("← 上一頁", key="page_prev", disabled=st.session_state.card_page <= 1, use_container_width=True):
        st.session_state.card_page -= 1
        st.rerun()
    if next_col.button("下一頁 →", key="page_next", disabled=st.session_state.card_page >= total_pages, use_container_width=True):
        st.session_state.card_page += 1
        st.rerun()

    page_start = (st.session_state.card_page - 1) * CARD_PAGE_SIZE
    page_records = browse_records[page_start:page_start + CARD_PAGE_SIZE]

    # 一排兩張（原本三張）：三欄時每張卡片寬 437px，量過有 27% 的文字段落會折成兩行
    # （長地址、長文章標題、摘要都中招）；兩欄時每張約 730px，99% 的段落一行放得下。
    # 代價是同樣 24 張卡從 8 排變 12 排、整頁捲動長約 20%，但每張卡片的可讀性明顯較好，
    # 一眼能看完一個景點的完整資訊，不用在斷行之間跳。
    CARD_COLUMNS = 2
    card_cols = st.columns(CARD_COLUMNS)
    for i, rec in enumerate(page_records):
        sid = rec["primary_id"]
        with card_cols[i % CARD_COLUMNS].container(border=True, key=f"spot_card_{sid}"):
            st.markdown(f"**{rec['name']}**")
            spot_type = rec.get("spot_type") or "未分類"
            type_emoji = SPOT_TYPE_ICONS.get(spot_type, SPOT_TYPE_ICONS["未分類"])[0]
            st.caption(f"{rec['neighborhood'] or '（無區域資訊）'} · {type_emoji} {spot_type}")
            for line in spot_fact_lines(rec, include_transit=False, combine_hours_closed=True):
                st.caption(line)
            # 期間限定活動：收合成一行「🎪 期間限定活動（N）」，點開才展開完整清單（類別
            # 色塊＋活動名稱超連結＋檔期）。卡片是三欄窄版，活動名稱中位數 17 字、最長 36 字，
            # 直接攤平會把卡片撐長好幾行；收進 popover 可以讓預設畫面維持乾淨，需要的人再點開，
            # 連對到 5 個活動的景點也放得下。
            spot_events = iwafu_events_by_spot.get(sid, [])
            if spot_events:
                with st.popover(f"🎪 期間限定活動（{len(spot_events)}）", use_container_width=True):
                    for ev_i, ev in enumerate(spot_events):
                        title_html = (
                            f"<a href='{ev['url']}' target='_blank'>{ev['title']}</a>"
                            if ev["url"] else ev["title"]
                        )
                        st.markdown(
                            f"{event_category_chip(ev['category'])} {title_html}",
                            unsafe_allow_html=True,
                        )
                        st.caption(format_event_period(ev["start"], ev["end"]))
                        # 直接從這裡把活動加進「推薦旅遊日期」：不然使用者得自己記住活動名稱、
                        # 再去側欄那個 175 筆的選單裡找或打字，大部分人不會想打字。
                        if ev["label"] in st.session_state.wanted_event_labels:
                            st.caption("✓ 已加入推薦旅遊日期")
                        elif ev["label"] in iwafu_event_labels:
                            st.button(
                                "加入推薦旅遊日期",
                                key=f"add_event_{sid}_{ev_i}",
                                on_click=add_event_to_date_picker,
                                args=(ev["label"],),
                                use_container_width=True,
                            )
            for block in article_blocks(rec["articles"], truncate_summary=45):
                st.caption(block)
            if sid in BRANCH_SHARED_DESC_NOTE:
                st.caption(BRANCH_SHARED_DESC_NOTE[sid])
            st.checkbox(
                "加入心願單", value=(sid in st.session_state.wishlist_ids), key=f"pick_{sid}",
                on_change=_sync_wishlist_from_card_checkbox, args=(sid,),
            )

# 「行程操作」放在腳本最後（同一支腳本可以多次 with st.sidebar，內容依執行順序往下疊），
# 所以它會出現在側欄最下方，而不是被寫死在規劃模式那個分支裡。原本兩顆按鈕只在分天畫面
# 才看得到，但使用者常在瀏覽模式挑景點挑到一半就想存檔——而且就算還沒分天，心願單本身
# （景點基本資料＋各篇文章連結）就已經有帶走的價值。所以改成只要心願單非空就出現，
# 匯出的內容跟著當下模式走。
if st.session_state.wishlist_ids:
    with st.sidebar:
        st.markdown("#### 行程操作")
        if itinerary_csv is not None:
            csv_bytes, csv_name = itinerary_csv, f"行程規劃_{start_date:%Y%m%d}.csv"
            export_hint = "匯出排好天數與造訪順序的完整行程。"
        else:
            csv_bytes = build_wishlist_csv(st.session_state.wishlist_ids, spot_records, selected_event_objs)
            csv_name = f"心願單_{date.today():%Y%m%d}.csv"
            export_hint = "目前還沒分天，匯出的是心願單清單（景點基本資料＋文章連結）；分天之後匯出的會是完整行程。"
        default_plan_name = (
            f"{start_date:%Y-%m-%d} {day_count}天行程" if day_count
            else f"{date.today():%Y-%m-%d} 心願單"
        )
        plan_name = st.text_input(
            "計畫名稱", value=default_plan_name, key="plan_name_input", label_visibility="collapsed"
        )
        op_col1, op_col2 = st.columns(2)
        with op_col1:
            st.download_button(
                "匯出", data=csv_bytes, file_name=csv_name, mime="text/csv",
                use_container_width=True,
            )
        with op_col2:
            if st.button("儲存計畫", key="save_plan_btn", use_container_width=True):
                saved_name = plan_name.strip() or default_plan_name
                save_plan_to_db(
                    saved_name, sorted(st.session_state.wishlist_ids), day_count, start_date,
                    st.session_state.wanted_event_labels,
                )
                st.toast(f"已儲存「{saved_name}」")
                st.rerun()
        st.caption(export_hint + "「儲存計畫」記住的是心願單、天數、開始日期與想參加的期間限定活動，載入後分組會重新計算。")
