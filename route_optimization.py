# -*- coding: utf-8 -*-
"""
一鍵順序優化（近似 TSP）：使用者指定當天第一個要去的景點當起點，
用 Haversine 距離 + 貪婪演算法（每步選離目前位置最近的未造訪景點）
排出不走回頭路的近似最短造訪順序。

只排單一天的順序（輸入通常是 kmeans_clustering.cluster_by_days() 某一天的結果），
起點由使用者指定，不自動猜測，跟系統既有「協助決策、不代替決策」的原則一致。
"""
import math

# 交通警示門檻：單段距離超過這個數字，提醒使用者這段用走的可能太遠，查一下交通方式
LONG_LEG_WARNING_KM = 1.5


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def greedy_route(spots, start_id):
    """spots: [{"id","name","area","latitude","longitude"}, ...]（單一天）
    start_id: 使用者指定的起點景點 id
    回傳依造訪順序排好的 spots list，並附上每段距離。"""
    remaining = {s["id"]: s for s in spots}
    if start_id not in remaining:
        raise ValueError(f"起點 id={start_id} 不在這天的景點清單裡")

    current = remaining.pop(start_id)
    route = [current]
    leg_distances = []

    while remaining:
        nearest_id = min(
            remaining,
            key=lambda sid: haversine_km(
                current["latitude"], current["longitude"],
                remaining[sid]["latitude"], remaining[sid]["longitude"],
            ),
        )
        dist = haversine_km(
            current["latitude"], current["longitude"],
            remaining[nearest_id]["latitude"], remaining[nearest_id]["longitude"],
        )
        current = remaining.pop(nearest_id)
        route.append(current)
        leg_distances.append(dist)

    return route, leg_distances


if __name__ == "__main__":
    from kmeans_clustering import cluster_by_days
    import sqlite3

    conn = sqlite3.connect("travel_hub_official.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM mitsugo_spots WHERE area IN ('澀谷・下北澤・原宿・青山・表參道','惠比壽・中目黑・目黑・代官山') ORDER BY RANDOM() LIMIT 15"
    )
    spot_ids = [r[0] for r in cur.fetchall()]
    conn.close()

    days, warnings = cluster_by_days(spot_ids, 3)

    for day, day_spots in days.items():
        start_id = day_spots[0]["id"]  # 示範：demo 用當天第一筆模擬使用者指定起點
        route, leg_distances = greedy_route(day_spots, start_id)
        print(f"=== Day{day}（起點：{route[0]['name']}）===")
        print(f"  1. {route[0]['name']}")
        for i, (spot, dist) in enumerate(zip(route[1:], leg_distances), start=2):
            print(f"  {i}. {spot['name']}（距上一站 {dist:.2f} km）")
        print(f"  總距離：{sum(leg_distances):.2f} km")
        print()
