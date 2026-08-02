# -*- coding: utf-8 -*-
"""
飯店位置推薦（提案表格一、四）：整趟行程（心願單分好天之後的全部景點）算一個地理
中心點 → 找最近的東京大型轉運站 → 距離在門檻內（3km）才推薦該站附近住宿，否則
提示行程較分散、建議自行斟酌住宿地點或考慮分開找旅宿。跟系統既有「決策輔助、
不是自動決策」的原則一致，不會自動選飯店，只推薦「該住哪一站附近」。

**只推薦一間，不是逐天各推薦一間**：使用者實際用過後回饋，多天行程通常只會訂一間
飯店當根據地（不會每天換飯店），所以中心點要用「這幾天全部景點」一起算，不是
「每天分群中心各自找一次」——逐天算會給出好幾個不同站的推薦，對「訂一間飯店」
這個真實需求沒有幫助，甚至可能誤導使用者以為真的要每天換飯店。距離門檻（3km）
判斷「這個站夠不夠方便」的邏輯不變，只是套用對象從「單日分群中心」改成「全部
天數合併後的整體中心點」。

沿用 route_optimization.py 已驗證過的 Haversine 公式，不重複實作距離計算。
"""
from route_optimization import haversine_km

# 距離門檻：分群中心到最近轉運站在這個範圍內才推薦，提案書明訂「如 3 公里」
HOTEL_DISTANCE_THRESHOLD_KM = 3.0

# 東京 10 大轉運站經緯度（提案表格一「地鐵站座標資料」要求的固定字典，寫死常用站，
# 免除申請 API 金鑰與額度限制的成本）。挑選涵蓋市區主要旅遊/轉乘節點的車站，
# 跟 mitsugo_spots 景點分布的區域大致對應（新宿/澀谷/池袋=西側，上野/淺草/秋葉原=東北側，
# 東京/銀座=市中心，品川=南側門戶，六本木=南西夜生活區）。
TOKYO_HUB_STATIONS = {
    "東京駅": (35.681236, 139.767125),
    "新宿駅": (35.690921, 139.700258),
    "澀谷駅": (35.658034, 139.701636),
    "池袋駅": (35.728926, 139.710380),
    "品川駅": (35.630152, 139.740571),
    "上野駅": (35.713768, 139.777254),
    "秋葉原駅": (35.698683, 139.773843),
    "淺草駅": (35.711757, 139.799793),
    "六本木駅": (35.662836, 139.731992),
    "銀座駅": (35.671989, 139.765419),
}


def nearest_hub_station(lat, lon):
    """回傳 (最近的轉運站名稱, 距離km)。"""
    name, (slat, slon) = min(
        TOKYO_HUB_STATIONS.items(),
        key=lambda kv: haversine_km(lat, lon, kv[1][0], kv[1][1]),
    )
    return name, haversine_km(lat, lon, slat, slon)


def recommend_hotel_for_spots(spots):
    """spots: [{"latitude","longitude",...}, ...]（不限單一天，這裡吃的是「要一起算
    中心點」的整組景點清單）。先算這群景點的地理中心點，再找離中心點最近的轉運站，
    距離在門檻內才 recommended=True，否則 False（行程較分散，交給使用者自行判斷）。"""
    lat = sum(s["latitude"] for s in spots) / len(spots)
    lon = sum(s["longitude"] for s in spots) / len(spots)
    station, distance = nearest_hub_station(lat, lon)
    return {
        "centroid": (lat, lon),
        "station": station,
        "distance_km": distance,
        "recommended": distance <= HOTEL_DISTANCE_THRESHOLD_KM,
    }


def recommend_hotel_for_trip(days):
    """days: cluster_by_days() 回傳的 {day: [spot, ...]}。把全部天數的景點攤平成一個
    清單，只算一次中心點、只回傳一個推薦結果——整趟行程訂一間飯店，不是逐天各推薦
    一次（見上方模組說明的理由）。"""
    all_spots = [s for group in days.values() for s in group]
    return recommend_hotel_for_spots(all_spots)


if __name__ == "__main__":
    from kmeans_clustering import cluster_by_days
    import sqlite3

    conn = sqlite3.connect("travel_hub_official.db")
    cur = conn.cursor()

    # 情境一：市區集中景點（澀谷/惠比壽一帶），預期整趟行程的中心點離澀谷駅在門檻內、會被推薦
    cur.execute(
        "SELECT id FROM mitsugo_spots WHERE area IN ('澀谷・下北澤・原宿・青山・表參道','惠比壽・中目黑・目黑・代官山') ORDER BY RANDOM() LIMIT 15"
    )
    scenario1 = [r[0] for r in cur.fetchall()]

    # 情境二：整趟行程都在吉祥寺一帶（`neighborhood`分類裡連よみうりランド、秋川渓谷這種
    # 遠郊景點都被歸進這一區，離市中心10幾公里起跳），不管怎麼分天、每天分群中心怎麼變，
    # 「全部天數合併」算出來的整體中心點都會離所有轉運站太遠，預期 recommended=False
    cur.execute("SELECT id FROM mitsugo_spots WHERE neighborhood = '吉祥寺' ORDER BY RANDOM() LIMIT 15")
    scenario2 = [r[0] for r in cur.fetchall()]

    conn.close()

    for spot_ids, k, label in [
        (scenario1, 3, "情境一：澀谷/惠比壽一帶心願單，分3天"),
        (scenario2, 3, "情境二：吉祥寺一帶（含遠郊景點）心願單，分3天"),
    ]:
        print(f"=== {label} ===")
        days, _warnings = cluster_by_days(spot_ids, k)
        for day, group in days.items():
            names = "、".join(s["name"] for s in group)
            print(f"Day{day}（{len(group)}筆：{names}）")
        r = recommend_hotel_for_trip(days)
        status = f"✅ 建議整趟入住 {r['station']} 附近" if r["recommended"] else f"⚠️ 行程較分散（最近的{r['station']}也有 {r['distance_km']:.2f} km）"
        print(f"整趟行程中心點距最近轉運站「{r['station']}」{r['distance_km']:.2f} km ｜ {status}")
        print()
