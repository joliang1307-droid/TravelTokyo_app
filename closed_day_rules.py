# -*- coding: utf-8 -*-
"""
公休警示：把 mitsugo_spots.cleaned_closed_days 的中文描述解析成可比對的公休規則，
跟使用者指定的實際行程日期比對，判斷該日期是否命中公休。

規則直接從資料庫裡實際出現過的 63 種公休文字寫法歸納（週一公休／星期二公休／
每週一、二公休／週二三公休／週日～三公休／每月第1、第3個週三公休／
週一及第二個週二公休 這類週期＋每月第N個週X混合寫法都有涵蓋），不是憑空設計。

只在「抓到明確規則」時才判定紅燈（確定衝突）；規則含糊（不定期、請洽官網）、
格式沒見過、或抽取完畢仍找不到任何星期資訊，一律回傳「無法確定」交給使用者
自行查證，不假裝有把握——跟系統既有的紅黃燈警示設計語言一致，
KMeans 單日景點數過多的黃燈警示也是同一套原則。

## 國定假日判定（2026-08-01 補上）
一開始「國定假日」「假日」這幾個字只是被當雜訊直接刪掉，只留下同一句裡的星期
規則（例如「週日、國定假日公休」只留下「週日」），因為國定假日不是固定星期幾、
沒辦法用星期規則表示，而系統原本也沒有一份實際的國定假日日期清單。

改用 `jpholiday` 套件（純本地計算，不用連網、不用自己維護日期表）補上這塊：
日本的國定假日不是全部固定日期（例如敬老の日是每年 9 月第三個星期一、秋分之日
要算天文曆才知道是哪一天），這正是為什麼不能只憑規則硬寫死日期，需要專門的套件
逐年精確計算，含「振替休日」（假日遇週日順延）這類細節都算在內。

做法：解析時先偵測文字裡有沒有「假日」（涵蓋「國定假日」，因為它本身就包含
「假日」兩字，不用另外列規則），記成 `national_holiday` 布林值；比對衝突時，
除了原本的星期／每月第N個星期規則，多一條「這天是不是 jpholiday 認定的國定
假日」，命中一樣判紅燈。
"""
import re
from datetime import date

import jpholiday

NO_INFO_TEXT = "查無公休資訊，詳情請以商家公告為準"
ALWAYS_OPEN_TEXT = "無公休"

WEEKDAY_MAP = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
WEEKDAY_ORDER = ["日", "一", "二", "三", "四", "五", "六"]  # 供「週日～三」這類範圍展開用
NUMERAL_MAP = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}

UNCERTAIN_KEYWORDS = ["不定", "不固定", "請參考", "請參照", "請洽"]

# 「假日」本身就會比對到「國定假日」（子字串包含），不用另外列一條規則。
HOLIDAY_PATTERN = re.compile(r"假日")

EXTRA_TOKEN_PATTERN = re.compile(
    r"年末年始|年始年末|展場佈置期間|\d{1,2}/\d{1,2}"
)
MONTHLY_PATTERN = re.compile(r"第([一二三四五1-5、第]+)個週([一二三四五六日天])")
RANGE_PATTERN = re.compile(r"(?:每?週|星期)([一二三四五六日天])[～~-]([一二三四五六日天])")
WEEKLY_RUN_PATTERN = re.compile(r"(?:每?週|星期)([一二三四五六日天、]+)")


def _nth_weekday_of_month(d: date) -> int:
    """回傳 d 是當月第幾個「同星期幾」的日子（1-based）。"""
    return (d.day - 1) // 7 + 1


def _parse_number_run(s: str):
    """把「1、3」「二四」「第1、第3」這類數字/國字混合字串拆成整數集合。"""
    nums = set()
    for token in re.split("、", s):
        token = token.replace("第", "").strip()
        if not token:
            continue
        if token.isdigit():
            nums.add(int(token))
        else:
            for ch in token:
                if ch in NUMERAL_MAP:
                    nums.add(NUMERAL_MAP[ch])
    return nums


def parse_closed_days(text):
    """回傳 dict，status 為以下四種之一：
    - "none"：查無公休資訊（fallback 值），不判定
    - "always_open"：文字明確寫「無公休」，任何日期都不會衝突
    - "uncertain"：不定期／請洽官網／格式看不懂，交給使用者自行查證
    - "resolved"：抓到明確規則，附 weekly_weekdays（每週固定公休星期，0=一…6=日）、
      monthly_rules（[(第幾個, 星期)] 每月第N個週X公休）、national_holiday
      （布林值，文字裡有沒有提到國定假日／假日公休）
    """
    if text is None or text.strip() == "" or text.strip() == NO_INFO_TEXT:
        return {"status": "none", "raw": text}

    normalized = re.sub(r"\s+", "", text.strip())

    if normalized == ALWAYS_OPEN_TEXT:
        return {"status": "always_open", "raw": text}

    if any(kw in normalized for kw in UNCERTAIN_KEYWORDS):
        return {"status": "uncertain", "raw": text, "reason": "文字寫不定期或請洽官方，無法自動判讀"}

    # 「週末」在沒有「第N個」修飾時，視為週六＋週日；有「第N個週末」修飾則太模糊，留給 uncertain 兜底
    working = re.sub(r"(?<!個)週末", "週六週日", normalized)

    national_holiday = bool(HOLIDAY_PATTERN.search(working))
    working = HOLIDAY_PATTERN.sub("", working)
    working = EXTRA_TOKEN_PATTERN.sub("", working)

    monthly_rules = []
    for m in MONTHLY_PATTERN.finditer(working):
        weekday = WEEKDAY_MAP[m.group(2)]
        for n in _parse_number_run(m.group(1)):
            monthly_rules.append((n, weekday))
    working = MONTHLY_PATTERN.sub("", working)

    weekly_weekdays = set()
    range_match = RANGE_PATTERN.search(working)
    if range_match:
        start_idx = WEEKDAY_ORDER.index(range_match.group(1))
        end_idx = WEEKDAY_ORDER.index(range_match.group(2))
        for ch in WEEKDAY_ORDER[start_idx:end_idx + 1]:
            weekly_weekdays.add(WEEKDAY_MAP[ch])
        working = working[:range_match.start()] + working[range_match.end():]

    for m in WEEKLY_RUN_PATTERN.finditer(working):
        for ch in m.group(1):
            if ch in WEEKDAY_MAP:
                weekly_weekdays.add(WEEKDAY_MAP[ch])

    if not weekly_weekdays and not monthly_rules and not national_holiday:
        # 抓到「公休」相關文字，但規則抽取不出任何星期資訊：格式沒見過，不要裝作「無公休」
        return {"status": "uncertain", "raw": text, "reason": "看得出有公休規則但格式無法自動解析"}

    return {
        "status": "resolved",
        "raw": text,
        "weekly_weekdays": weekly_weekdays,
        "monthly_rules": monthly_rules,
        "national_holiday": national_holiday,
    }


def check_conflict(parsed, visit_date: date):
    """回傳 "red"（確定該日期公休）／"yellow"（無法確定，建議查證）／None（確定不衝突或無資訊）。"""
    status = parsed["status"]
    if status in ("none", "always_open"):
        return None
    if status == "uncertain":
        return "yellow"

    weekday = visit_date.weekday()
    if weekday in parsed["weekly_weekdays"]:
        return "red"
    nth = _nth_weekday_of_month(visit_date)
    for n, w in parsed["monthly_rules"]:
        if w == weekday and n == nth:
            return "red"
    if parsed["national_holiday"] and jpholiday.is_holiday(visit_date):
        return "red"
    return None


if __name__ == "__main__":
    import sqlite3

    conn = sqlite3.connect("travel_hub_official.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT cleaned_closed_days FROM mitsugo_spots WHERE cleaned_closed_days IS NOT NULL"
    )
    samples = [r[0] for r in cur.fetchall()]
    conn.close()

    print(f"=== 解析 {len(samples)} 種實際出現過的公休文字 ===")
    counts = {"none": 0, "always_open": 0, "uncertain": 0, "resolved": 0}
    for text in samples:
        parsed = parse_closed_days(text)
        counts[parsed["status"]] += 1
        detail = ""
        if parsed["status"] == "resolved":
            detail = f"weekly={sorted(parsed['weekly_weekdays'])} monthly={parsed['monthly_rules']}"
        elif parsed["status"] == "uncertain":
            detail = parsed["reason"]
        print(f"[{parsed['status']:^11}] {text!r:40} {detail}")
    print()
    print("統計：", counts)

    print()
    print("=== 用實際日期測試衝突判斷 ===")
    test_cases = [
        ("週一公休", date(2026, 8, 3)),   # 一
        ("週一公休", date(2026, 8, 4)),   # 二
        ("每月第1、第3個週三公休", date(2026, 8, 5)),   # 8/5 是第1個週三
        ("每月第1、第3個週三公休", date(2026, 8, 12)),  # 8/12 是第2個週三
        ("週一及第二個週二公休", date(2026, 8, 11)),   # 8/11 是第2個週二
        ("無公休", date(2026, 8, 3)),
        ("不定期公休", date(2026, 8, 3)),
        ("週日、國定假日公休", date(2026, 8, 11)),   # 8/11 山の日，平日但是國定假日
        ("週日、國定假日公休", date(2026, 8, 4)),    # 8/4 是平日也不是假日
    ]
    for text, d in test_cases:
        parsed = parse_closed_days(text)
        result = check_conflict(parsed, d)
        print(f"{text!r:28} @ {d}（週{'一二三四五六日'[d.weekday()]}） -> {result}")
