# -*- coding: utf-8 -*-
"""
文章重點摘要抽取：抽取式摘要，從景點介紹 description 挑出最具代表性的 2-3 句
顯示在卡片上，讓使用者不用點進原文就能抓到重點。

方法：句子與全文平均向量相似度比對——把整篇拆成句子，各句子與全篇分別做
TF-IDF 向量化，用每句對「全篇平均向量」的 cosine similarity 排名，
取分數最高的幾句，但輸出時照原文順序排列（不是照分數排列），維持可讀性。

資料觀察（2026-07-11）：1004 筆有效 description 裡，只有 226 筆（~22.5%）
句數 >3、真正需要抽取；其餘本來就只有 1-3 句，直接視為摘要即可，不需要跑演算法。
"""
import sqlite3
import re
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = "travel_hub_official.db"
SUMMARY_MAX_SENTENCES = 3
SPLIT_THRESHOLD = 3  # 句數 <= 此值就整篇當摘要，不跑抽取


def split_sentences(text):
    parts = re.split(r"(?<=[。！？])", text)
    return [p.strip() for p in parts if p.strip()]


def _rank_sentences(sentences, max_sentences):
    """句子與「全體平均向量」的 TF-IDF cosine similarity 排名，取分數最高幾句、
    依原文（或多篇合併後的）出現順序排列。extract_summary／extract_consensus_summary
    共用同一套排名邏輯，差別只在於餵進來的 sentences 是單篇還是跨篇合併。"""
    if len(sentences) <= max_sentences:
        return sentences

    docs = [" ".join(jieba.cut(s)) for s in sentences]
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(docs)
    doc_vector = np.asarray(tfidf.mean(axis=0))
    scores = cosine_similarity(tfidf, doc_vector).flatten()

    top_idx = sorted(np.argsort(scores)[-max_sentences:])
    return [sentences[i] for i in top_idx]


def extract_summary(text, max_sentences=SUMMARY_MAX_SENTENCES):
    sentences = split_sentences(text)
    if len(sentences) <= SPLIT_THRESHOLD:
        return sentences
    return _rank_sentences(sentences, max_sentences)


def extract_consensus_summary(texts, max_sentences=SUMMARY_MAX_SENTENCES):
    """同一個景點如果被好幾篇文章介紹過，每篇各自的摘要（或 description）合併成一個
    句子池，一樣用「跟全體平均向量的相似度」排名——因為平均向量會被多篇文章都提到
    的主題／用詞拉高權重，排名靠前的句子自然比較是「跨文章有共識」的內容，不是隨便
    挑某一篇的句子。地圖 popup 用這個取代「列出每篇文章各自摘要＋連結」的做法，
    卡片區則保留逐篇列出（使用者想要「先看摘要挑要點開哪篇原文」，兩種呈現方式
    服務不同情境，不是互相取代）。"""
    sentences = []
    for text in texts:
        if text:
            sentences.extend(split_sentences(text))
    if not sentences:
        return ""
    return "".join(_rank_sentences(sentences, max_sentences))


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(mitsugo_spots)")
    cols = [c[1] for c in cur.fetchall()]
    if "summary" not in cols:
        cur.execute("ALTER TABLE mitsugo_spots ADD COLUMN summary TEXT")
    conn.commit()

    cur.execute("SELECT id, description FROM mitsugo_spots WHERE description IS NOT NULL AND description != ''")
    rows = cur.fetchall()

    extracted_count = 0
    for id_, desc in rows:
        summary_sentences = extract_summary(desc)
        summary = "".join(summary_sentences)
        cur.execute("UPDATE mitsugo_spots SET summary = ? WHERE id = ?", (summary, id_))
        if len(split_sentences(desc)) > SPLIT_THRESHOLD:
            extracted_count += 1

    conn.commit()
    conn.close()
    print(f"處理 {len(rows)} 筆，其中 {extracted_count} 筆有跑抽取式摘要（其餘句數已在門檻內，原文照登）")


if __name__ == "__main__":
    main()
