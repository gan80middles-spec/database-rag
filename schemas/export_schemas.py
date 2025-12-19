from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Type

from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

# 这里按你的真实模块路径改：
# 例如你的文件是 schemas/graphs.py，则用 from schemas.graphs import CaseGraph, ContractGraph
from graphs import CaseGraph, ContractGraph


DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_model_schema(
    model: Type[BaseModel],
    out_path: Path,
    *,
    mode: str = "validation",
    by_alias: bool = True,
    ref_template: str = "#/$defs/{model}",
) -> dict:
    # Pydantic v2: BaseModel.model_json_schema(...) :contentReference[oaicite:5]{index=5}
    schema = model.model_json_schema(
        mode=mode,
        by_alias=by_alias,
        ref_template=ref_template,
    )
    schema.setdefault("$schema", DRAFT_2020_12)
    _write_json(out_path, schema)
    return schema


def dump_bundle_schema(
    models: list[tuple[Type[BaseModel], str]],
    out_path: Path,
    *,
    title: str = "LawLLM Schemas Bundle",
    by_alias: bool = True,
    ref_template: str = "#/$defs/{model}",
) -> dict:
    # Pydantic v2: models_json_schema 用于“多个模型一次性生成 defs” :contentReference[oaicite:6]{index=6}
    schemas_map, definitions_schema = models_json_schema(
        models,
        by_alias=by_alias,
        title=title,
        ref_template=ref_template,
    )
    bundle = {
        "$schema": DRAFT_2020_12,
        "title": title,
        # definitions_schema 本身就包含 $defs（以及可选 title/description）:contentReference[oaicite:7]{index=7}
        "$defs": definitions_schema.get("$defs", {}),
        "oneOf": [schemas_map[(m, mode)] for (m, mode) in models],
    }
    _write_json(out_path, bundle)
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="artifacts/schemas", help="输出目录")
    ap.add_argument("--mode", default="validation", choices=["validation", "serialization"])
    ap.add_argument("--by-alias", action="store_true", default=True)
    ap.add_argument("--no-by-alias", dest="by_alias", action="store_false")
    ap.add_argument("--ref-template", default="#/$defs/{model}")
    ap.add_argument("--bundle", action="store_true", help="同时输出 bundle schema")
    args = ap.parse_args()

    outdir = Path(args.outdir)

    dump_model_schema(
        CaseGraph,
        outdir / f"CaseGraph.{args.mode}.schema.json",
        mode=args.mode,
        by_alias=args.by_alias,
        ref_template=args.ref_template,
    )
    dump_model_schema(
        ContractGraph,
        outdir / f"ContractGraph.{args.mode}.schema.json",
        mode=args.mode,
        by_alias=args.by_alias,
        ref_template=args.ref_template,
    )

    if args.bundle:
        dump_bundle_schema(
            [(CaseGraph, args.mode), (ContractGraph, args.mode)],
            outdir / f"bundle.{args.mode}.schema.json",
            title="LawLLM Graph Schemas Bundle",
            by_alias=args.by_alias,
            ref_template=args.ref_template,
        )

    print(f"[OK] schemas exported to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
