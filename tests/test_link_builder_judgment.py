import sys
import types


def _ensure_pymongo_stub() -> None:
    if "pymongo" in sys.modules:
        return
    stub = types.ModuleType("pymongo")

    class _MongoClient:  # pragma: no cover - simple stub for import side effects
        pass

    stub.MongoClient = _MongoClient
    sys.modules["pymongo"] = stub


_ensure_pymongo_stub()


from rag.link_builder_judgment import cn_to_int  # noqa: E402  (after stub setup)


def test_cn_to_int_handles_basic_numerals() -> None:
    assert cn_to_int("第一条") == 1
    assert cn_to_int("第十条") == 10
    assert cn_to_int("第十二条") == 12
    assert cn_to_int("第101条") == 101


def test_cn_to_int_handles_mixed_chinese_units() -> None:
    assert cn_to_int("第一百一十条") == 110
    assert cn_to_int("第一百二十三条") == 123
    assert cn_to_int("第一千一百六十四条") == 1164
    assert cn_to_int("第一千二百条") == 1200


def test_cn_to_int_ignores_non_numeric_content() -> None:
    assert cn_to_int("第十条之二") == 10
    assert cn_to_int("第  十一  条") == 11
