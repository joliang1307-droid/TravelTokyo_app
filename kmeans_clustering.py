# -*- coding: utf-8 -*-
"""
多天行程空間分群：使用者勾選心願單景點 + 指定天數 k，用 KMeans 依經緯度
把景點分成 Day1、Day2...，協助使用者判斷「這幾個景點適合排同一天」。

這是決策輔助，不是自動決策：單日景點數超過門檻只跳警示，不會自動拆天或搬移景點。

日後 Streamlit 串接方式：使用者在地圖/卡片頁勾選景點（收集 spot_id list）、
天數滑桿給 k，呼叫 cluster_by_days(spot_ids, k) 取得分組結果直接渲染。
"""
import sqlite3
import numpy as np
from sklearn.cluster import KMeans

DB_PATH = "travel_hub_official.db"
DAILY_SPOT_WARNING_THRESHOLD = 6


def get_spots(spot_ids):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    placeholders = ",".join("?" * len(spot_ids))
    cur.execute(
        f"SELECT id, name, area, latitude, longitude FROM mitsugo_spots WHERE id IN ({placeholders})",
        spot_ids,
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def cluster_by_days(spot_ids, k):
    """回傳 {day: [{"id","name","area","latitude","longitude"}, ...]} 及每天景點數過多的警示清單。"""
    spots = get_spots(spot_ids)
    if len(spots) < k:
        raise ValueError(f"景點數（{len(spots)}）比天數（{k}）還少，無法分成 {k} 天")

    coords = np.array([[s[3], s[4]] for s in spots])
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(coords)

    days = {i + 1: [] for i in range(k)}
    for spot, label in zip(spots, labels):
        days[label + 1].append(
            {"id": spot[0], "name": spot[1], "area": spot[2], "latitude": spot[3], "longitude": spot[4]}
        )

    warnings = []
    for day, day_spots in days.items():
        if len(day_spots) > DAILY_SPOT_WARNING_THRESHOLD:
            warnings.append(f"Day{day} 有 {len(day_spots)} 個景點，超過建議上限（{DAILY_SPOT_WARNING_THRESHOLD}），請規劃時注意時間安排")

    return days, warnings


def demo(spot_ids, k, label):
    print(f"=== {label}（{len(spot_ids)} 筆景點，分 {k} 天）===")
    days, warnings = cluster_by_days(spot_ids, k)
    for day, day_spots in days.items():
        names = "、".join(s["name"] for s in day_spots)
        areas = {s["area"] for s in day_spots}
        print(f"Day{day}（{len(day_spots)}筆，{'/'.join(areas)}）: {names}")
    if warnings:
        print("[警示]")
        for w in warnings:
            print(" -", w)
    print()


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 情境一：模擬使用者用區域篩選器選了「澀谷/惠比壽」一帶，勾了裡面約15筆景點，排3天
    cur.execute(
        "SELECT id FROM mitsugo_spots WHERE area IN ('澀谷・下北澤・原宿・青山・表參道','惠比壽・中目黑・目黑・代官山') ORDER BY RANDOM() LIMIT 15"
    )
    scenario1 = [r[0] for r in cur.fetchall()]
    demo(scenario1, 3, "情境一：澀谷/惠比壽一帶心願單")

    # 情境二：模擬使用者跨區亂選（澀谷市區為主 + 1個近郊樂園），測試離群景點會不會自成一天
    cur.execute(
        "SELECT id FROM mitsugo_spots WHERE area = '澀谷・下北澤・原宿・青山・表參道' ORDER BY RANDOM() LIMIT 8"
    )
    scenario2 = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT id FROM mitsugo_spots WHERE name LIKE '%よみうりランド%' LIMIT 1")
    outlier = cur.fetchall()
    scenario2 += [r[0] for r in outlier]
    demo(scenario2, 3, "情境二：市區為主+近郊離群景點")

    # 情境三：單一區域景點數多、測試單日上限警示會不會被觸發
    cur.execute(
        "SELECT id FROM mitsugo_spots WHERE area = '中野・高圓寺・荻窪・吉祥寺' ORDER BY RANDOM() LIMIT 20"
    )
    scenario3 = [r[0] for r in cur.fetchall()]
    demo(scenario3, 2, "情境三：單區域20筆分2天（測試上限警示）")

    conn.close()
