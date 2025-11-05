#!/usr/bin/env python3
"""Utility script to drop a MongoDB index from a collection.

This script exists because it is easy to forget that ``mongosh`` exposes
collections via ``db.<collection_name>`` instead of bare variables.  Running the
script ensures that the index drop uses the correct syntax regardless of the
shell environment.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from pymongo import MongoClient
from pymongo.collection import Collection


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://localhost:27017",
        help="MongoDB connection URI (default: %(default)s)",
    )
    parser.add_argument(
        "--database",
        default="lawKB",
        help="Target database name (default: %(default)s)",
    )
    parser.add_argument(
        "--collection",
        default="law_kb_links",
        help="Target collection name (default: %(default)s)",
    )
    parser.add_argument(
        "--index",
        default="from_doc_1_to_doc_1_edge_1",
        help="Name of the index to drop (default: %(default)s)",
    )
    return parser.parse_args(argv)


def drop_index(collection: Collection, index_name: str) -> bool:
    """Drop ``index_name`` from ``collection`` if it exists.

    Returns ``True`` when the index was dropped and ``False`` when the index was
    not present.
    """

    existing = {spec["name"] for spec in collection.list_indexes()}
    if index_name not in existing:
        return False

    collection.drop_index(index_name)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    client = MongoClient(args.mongo_uri)
    collection = client[args.database][args.collection]

    dropped = drop_index(collection, args.index)
    if dropped:
        print(
            f"Dropped index '{args.index}' from collection "
            f"'{args.database}.{args.collection}'."
        )
        return 0

    print(
        f"Index '{args.index}' was not found on collection "
        f"'{args.database}.{args.collection}'."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
