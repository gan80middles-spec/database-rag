from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ContractType(str, Enum):
    BUY_SELL = "buy_sell"
    LEASE = "lease"
    LABOR = "labor"
    NDA = "nda"
    OUTSOURCING = "outsourcing"
    SOFTWARE = "software"


class PresenceDefinition(BaseModel):
    """Template-level definition for contract presence entries."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="presence 条目的唯一标识，例如 party_info/subject/quantity")
    label: str = Field(description="中文说明，例如 当事人信息（出卖人/买受人）")
    required: bool = Field(description="模板层意义上的应当出现项")
    contract_type: ContractType = Field(description="presence 定义所属的合同类型")
    weight: float = Field(default=1.0, ge=0.0, description="用于风险评分的权重")


class PresenceResult(BaseModel):
    """Result-level presence check outcome."""

    model_config = ConfigDict(extra="ignore")

    contract_type: ContractType = Field(description="合同类型，便于检索与统计")
    presence_id: str = Field(description="对应 PresenceDefinition.id")
    present: bool = Field(description="要素是否在合同正文中出现")
    required: bool = Field(description="从模板继承的 required 快照")
    label: str = Field(description="从模板继承的 label 快照，用于展示")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="0~1 的置信度，可空",
    )
    source_clause_ids: List[str] = Field(
        default_factory=list,
        description="命中的条款或 chunk 的 id 列表",
    )
    comment: Optional[str] = Field(
        default=None, description="缺失原因、风险说明等描述性文字"
    )


class ContractDoc(BaseModel):
    """Contract document container with presence results."""

    model_config = ConfigDict(extra="ignore")

    doc_id: str = Field(description="对应 Mongo/Milvus 的 doc_id")
    contract_type: ContractType = Field(description="合同类型")
    title: str = Field(description="合同标题")
    signed_date: Optional[datetime] = Field(
        default=None, description="签署日期，无法解析则为 None"
    )
    presence_results: List[PresenceResult] = Field(
        default_factory=list, description="各个 presence 条目的检查结果"
    )


def make_presence_result(
    contract_type: ContractType,
    presence_def: PresenceDefinition,
    present: bool,
    *,
    confidence: float | None = None,
    source_clause_ids: list[str] | None = None,
    comment: str | None = None,
) -> PresenceResult:
    """Construct a PresenceResult using a template definition and detection data."""

    return PresenceResult(
        contract_type=contract_type,
        presence_id=presence_def.id,
        present=present,
        required=presence_def.required,
        label=presence_def.label,
        confidence=confidence,
        source_clause_ids=source_clause_ids or [],
        comment=comment,
    )


__all__ = [
    "ContractType",
    "PresenceDefinition",
    "PresenceResult",
    "ContractDoc",
    "make_presence_result",
]
