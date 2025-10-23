from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer, util

MILVUS_URI = "http://120.46.59.93:19530"
MILVUS_TOKEN = "root:3MEf;yra~A$ze!%w"
COLL = "law_kb_chunks"
MODEL = "/media/user/7b0c3c84-5ba2-4d83-b72d-6a0ac8e6fcf2/law-llm/models/bge-large-zh-v1.5"

queries = [
  ("民间借贷利率上限是多少？", r"民间借贷"),
  ("起诉期限怎么计算（行政诉讼）", r"行政诉讼法.*起诉|起诉期限"),
  ("什么是人脸识别个人信息？", r"侵犯公民个人信息刑.*解释|人脸识别"),
  ("建设工程价款优先受偿如何主张", r"建设工程.*解释"),
  ("买卖合同标的物风险转移在何时", r"买卖合同")
]

def milvus_search(texts, topk=8):
    connections.connect("default", uri = MILVUS_URI, token=MILVUS_TOKEN)
    col = Collection(COLL)
    embed = SentenceTransformer(MODEL, trust_remote_code=True)
    embs = embed.encode([t for t, _ in texts], normalize_embeddings=True)
    expr = 'doc_type in ["statute", "judicial_interpretation"]'
    res = col.search(embs, "embedding", param={"metric_type":"IP","params":{"ef":128}},
                     limit=topk, expr = expr, output_fields=["chunk_id", "law_name", "article_no", "doc_type"])
    return res

if __name__ == "__main__":
    result = milvus_search(queries, topk = 8)
    hit_at_5 = 0
    for qi, (q, must) in enumerate(queries):
        print(f"\nQ{qi+1}:{q}")
        ok = False
        for j, hit in enumerate(result[qi]):
            law = hit.entity.get("law_name") or ""
            art = hit.entity.get("article_no") or ""
            dt = hit.entity.get("doc_type")
            print(f"  {j+1:>2}. [{dt}] {law} | {art}")
            if __import__("re").search(must, f"{law} {art}"):
                ok = True
        if ok: hit_at_5 += 1
    print(f"\nHit@8: {hit_at_5}/{len(queries)}")
