# 合同模板批量入库脚本说明

本说明基于 `rag/ingest_contract_templates.py`，详细解释脚本如何处理 `*-docs.jsonl` 和 `*-chunks.jsonl` 两类文件，并分别写入 MongoDB 与 Milvus。

## 输入文件结构

- **`*-docs.jsonl`**：文档级元数据，每行一个 JSON 对象，至少包含 `doc_id`，可选字段如 `title`、`business_type`、`legal_type` 等。
- **`*-chunks.jsonl`**：切块数据，每行一个 JSON 对象，包含文本切块 `text` 及关联元数据，如 `doc_id`、`chunk_id`、`order`/`chunk_index`、`section`/`clause_no` 等。
- 两个文件同名不同后缀（如 `xxx-docs.jsonl` 与 `xxx-chunks.jsonl`）视为一组；脚本会递归扫描 `--input_dir` 下满足通配符的文件（默认 `*-docs.jsonl`、`*-chunks.jsonl`）。

## 处理流程

1. **加载文档元数据**
   - 先读取所有 `*-docs.jsonl`，按 `doc_id` 建立映射（缺失字段会填充默认值，如 `doc_type=contract_template`）。
2. **逐个切块文件处理**
   - 读取对应 `*-chunks.jsonl` 的所有行；每行按以下规则标准化：
     - `doc_type` 默认 `contract_template`。
   - `business_type` 默认取 `contract_type`，否则为空。
   - `chunk_index` 优先取 `order`，否则取原 `chunk_index`，再默认 0；`order` 字段同步为切块顺序（用于 Milvus 标量过滤）。
     - 记录当前向量维度 `embedding_dim`（用于 Mongo 记录）。
3. **编码文本（可选）**
   - 若未设置 `--mongo_only 1`，使用指定模型（默认 `BAAI/bge-m3`）按批量计算切块文本向量，并更新实际维度。
4. **写入 MongoDB**
   - 切块：批量 `upsert` 到 `--mongo_chunk_col`（默认 `contract_kb_chunks`），补充 `created_at`/`updated_at` 时间戳和上一步标准化字段。
   - 文档：将当前文件涉及的 `doc_id`（优先使用步骤 1 的元数据）批量 `upsert` 到 `--mongo_doc_col`（默认 `contract_kb_docs`）。
5. **写入 Milvus（向量库）**
   - 当 `--mongo_only 0` 时：
     - 确认/创建集合（默认 `contract_kb_chunks`，可 `--recreate 1` 先删除重建）。
     - 将 `chunk_id`、`doc_id`、`doc_type`、`business_type`、`legal_type`、`order`、`chunk_index`、`section` 及向量一并插入；如主键冲突会先删除再插入。

## 与样例文件的对应关系

假设文件夹中有两组测试文件：
- `sample-docs.jsonl`
- `sample-chunks.jsonl`

脚本执行时会：
- 先从 `sample-docs.jsonl` 取出每条文档的 `doc_id` 及元数据。**文档级记录只会写入 Mongo（默认集合 `contract_kb_docs`，可通过 `--mongo_doc_col` 自定义），不会写入 Milvus**。
- 处理 `sample-chunks.jsonl`：为每个切块补全字段、计算向量（如未指定 `--mongo_only`），然后：
  - **切块元数据与时间戳写入 Mongo（默认集合 `contract_kb_chunks`）**；
  - **切块向量与主键及标量字段写入 Milvus（默认集合 `contract_kb_chunks`），其中 doc_id / doc_type / business_type / legal_type / order 作为标量字段用于 filter**。
- 处理完一组，再继续下一组文件，直到目录内所有匹配文件处理完毕。

## 运行示例
```bash
python rag/ingest_contract_templates.py \
  --input_dir data/processed/contract_template \
  --mongo_uri "mongodb://localhost:27017" --mongo_db lawkb \
  --milvus_host 127.0.0.1 --milvus_port 19530 \
  --collection contract_kb_chunks
```
- 如只写 Mongo、暂不计算向量，可添加 `--mongo_only 1`。
```
