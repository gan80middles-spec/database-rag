# schemas/graphs.py
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Literal, Union, get_args, get_origin

from pydantic import BaseModel, Field, ConfigDict, field_validator
from schemas.presence_schema import ContractDoc, ContractType, PresenceDefinition, PresenceResult
from rag.contract_template_chunker import ALL_PRESENCE_DEFINITIONS


# =========================
# 通用类型 & Enums
# =========================

class Jurisdiction(str, Enum):
    CN = "CN"
    JP = "JP"
    OTHER = "OTHER"


class PartyType(str, Enum):
    NATURAL_PERSON = "natural_person"
    COMPANY = "company"
    GOVERNMENT = "government"
    OTHER = "other"


class CaseSourceType(str, Enum):
    USER_QUERY = "user_query"
    JUDGMENT = "judgment"
    MIXED = "mixed"


class CaseProcedureStage(str, Enum):
    CONSULTATION = "consultation"
    FIRST_INSTANCE = "first_instance"
    SECOND_INSTANCE = "second_instance"
    RETRIAL = "retrial"


class RelationshipType(str, Enum):
    LOAN = "loan"
    LABOR = "labor"
    LEASE = "lease"
    SALE = "sale"
    MARRIAGE = "marriage"
    TORT = "tort"
    OTHER = "other"


class EvidenceType(str, Enum):
    BANK_RECORD = "bank_record"
    IOU = "IOU"
    CHAT_LOG = "chat_log"
    LABOR_CONTRACT = "labor_contract"
    WITNESS_TESTIMONY = "witness_testimony"
    AUDIO_VIDEO = "audio_video"
    OTHER = "other"


class StrengthLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PriorityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LawDocType(str, Enum):
    STATUTE = "statute"
    JUDICIAL_INTERPRETATION = "judicial_interpretation"


class RiskSeverity(str, Enum):
    RISK_LOW = "low"
    RISK_MEDIUM = "medium"
    RISK_HIGH = "high"


class ContractSource(str, Enum):
    TEMPLATE = "template"
    REAL_CASE = "real_case"
    UPLOADED = "uploaded"


class ContractClauseCategory(str, Enum):
    DEFINITION = "definition"
    SCOPE = "scope"
    PAYMENT = "payment"
    TERM = "term"
    TERMINATION = "termination"
    LIABILITY = "liability"
    CONFIDENTIALITY = "confidentiality"
    IP = "ip"
    DISPUTE_RESOLUTION = "dispute_resolution"
    LABOR_PROTECTION = "labor_protection"
    WORK_TIME = "work_time"
    SOCIAL_INSURANCE = "social_insurance"
    OTHER = "other"


class ObligationType(str, Enum):
    PAY_SALARY = "pay_salary"
    DELIVER_GOODS = "deliver_goods"
    PROVIDE_SERVICE = "provide_service"
    KEEP_CONFIDENTIAL = "keep_confidential"
    NON_COMPETE = "non_compete"
    OTHER = "other"


class RightType(str, Enum):
    TERMINATE = "terminate"
    INSPECT = "inspect"
    USE_IP = "use_ip"
    WITHHOLD_PAYMENT = "withhold_payment"
    OTHER = "other"


class ConditionType(str, Enum):
    SUSPENSIVE = "suspensive"
    RESOLUTORY = "resolutory"


class RemedyType(str, Enum):
    LIQUIDATED_DAMAGES = "liquidated_damages"
    PENALTY = "penalty"
    INDEMNITY = "indemnity"
    OTHER = "other"


class RiskType(str, Enum):
    UNILATERAL_TERMINATION = "unilateral_termination"
    HIGH_LIQUIDATED_DAMAGES = "high_liquidated_damages"
    WAIVER_OF_RIGHTS = "waiver_of_rights"
    NON_COMPETE_OVERBROAD = "non_compete_overbroad"
    MISSING_MANDATORY_TERM = "missing_mandatory_term"
    OTHER = "other"


# =========================
# CaseGraph 相关模型
# =========================

class CaseMeta(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    jurisdiction: Jurisdiction = Field(default=Jurisdiction.CN, description="法域")
    court: Optional[str] = Field(default=None, description="法院名称，可空（咨询阶段）")
    case_type: Optional[str] = Field(default=None, description="案件类型/案由简写")
    procedure_stage: CaseProcedureStage = Field(
        default=CaseProcedureStage.CONSULTATION,
        description="程序阶段",
    )
    cause: Optional[str] = Field(default=None, description="案由全称")
    created_at: Optional[str] = Field(
        default=None,
        description="创建时间（ISO 8601 字符串）",
    )
    user_id: Optional[str] = Field(default=None, description="用户ID")
    language: str = Field(default="zh-CN", description="语言代码")


class CaseParty(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: str = Field(..., description="当事人ID，如 P1/P2")
    role: str = Field(..., description="原告/被告/申请人等，保持原文")
    name: str = Field(..., description="当事人姓名/名称")
    type: PartyType = Field(..., description="当事人类型")
    attributes: dict = Field(
        default_factory=dict,
        description="一些布尔属性，如 is_employer/is_employee/is_lender 等",
    )


class CaseRelationship(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: str
    type: RelationshipType
    from_party_id: str
    to_party_id: str
    since: Optional[str] = Field(default=None, description="关系起始时间")
    until: Optional[str] = Field(default=None, description="关系结束时间")
    description: Optional[str] = Field(default=None, description="补充说明")


class CaseClaim(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: str
    claimant_id: str
    respondent_id: str
    type: str = Field(..., description="请求类型，如支付借款本息/支付劳动报酬")
    amount: Optional[float] = Field(default=None, description="请求金额")
    currency: Optional[str] = Field(default="CNY")
    other_demands: List[str] = Field(
        default_factory=list,
        description="其他请求事项",
    )


class CaseFact(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: str
    time: Optional[str] = Field(default=None, description="时间，ISO 日期")
    stage: Optional[Literal["pre_litigation", "during_litigation"]] = Field(
        default="pre_litigation",
    )
    description: str
    evidence_ids: List[str] = Field(default_factory=list)


class CaseEvidence(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: str
    type: EvidenceType
    description: str
    strength: StrengthLevel = StrengthLevel.MEDIUM
    is_disputed: bool = False


class LegalIssue(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: str
    issue: str = Field(..., description="争点问题描述")
    focus: Optional[str] = Field(default=None, description="争点聚焦点")
    priority: PriorityLevel = PriorityLevel.MEDIUM


class LawCandidate(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    doc_type: LawDocType
    law_name: str
    law_doc_id: str = Field(..., description="对应 law_kb_docs 中的ID")
    article_no: Optional[str] = Field(default=None, description="具体条号，如 第680条")
    article_range: Optional[str] = Field(default=None, description="范围，如 680-682")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SimilarJudgment(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    judgment_id: str = Field(..., description="文书ID，关联 law_kb_docs")
    similarity: float = Field(..., ge=0.0, le=1.0)
    court: Optional[str] = None
    judgment_date: Optional[str] = None
    result_brief: Optional[str] = None


class CaseTimelineItem(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    time: str
    label: str
    fact_ids: List[str] = Field(default_factory=list)


class CaseGraph(BaseModel):
    """
    案情图谱：CaseGraph
    """
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    case_id: str = Field(..., description="内部唯一案件ID")
    schema_version: str = Field(default="0.1.0", description="schema 版本号")

    source_type: CaseSourceType = Field(
        default=CaseSourceType.USER_QUERY,
        description="来源：用户案情/裁判文书/混合",
    )

    raw_text: str = Field(..., description="原始案情描述或整理后案情")
    summary: Optional[str] = Field(default=None, description="案情摘要（200-400字）")

    meta: CaseMeta = Field(default_factory=CaseMeta)
    parties: List[CaseParty] = Field(default_factory=list)
    relationships: List[CaseRelationship] = Field(default_factory=list)
    claims: List[CaseClaim] = Field(default_factory=list)
    facts: List[CaseFact] = Field(default_factory=list)
    evidence: List[CaseEvidence] = Field(default_factory=list)
    legal_issues: List[LegalIssue] = Field(default_factory=list)
    law_candidates: List[LawCandidate] = Field(default_factory=list)
    similar_judgments: List[SimilarJudgment] = Field(default_factory=list)
    timeline: List[CaseTimelineItem] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def check_schema_version(cls, v: str) -> str:
        if not v:
            raise ValueError("schema_version 不能为空")
        return v


# =========================
# ContractGraph 相关模型
# =========================

class ContractContact(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    person: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None


class ContractMeta(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    contract_type: ContractType
    title: Optional[str] = Field(default=None, description="合同标题")
    signed_date: Optional[str] = Field(default=None, description="签署日期")
    effective_date: Optional[str] = Field(default=None, description="生效日期")
    termination_date: Optional[str] = Field(default=None, description="终止日期")
    governing_law: Optional[str] = Field(
        default="中华人民共和国法律",
        description="适用法律",
    )
    jurisdiction: Optional[str] = Field(default=None, description="争议解决管辖法院/仲裁机构")
    language: str = Field(default="zh-CN")
    version: Optional[str] = Field(default=None)
    source: ContractSource = ContractSource.TEMPLATE


class ContractParty(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: str
    role: str = Field(..., description="甲方/乙方/用人单位/劳动者 等")
    name: str
    type: PartyType
    registration_no: Optional[str] = None
    address: Optional[str] = None
    contact: ContractContact = Field(default_factory=ContractContact)


class ContractDefinition(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    term: str = Field(..., description="术语名")
    definition_text: str = Field(..., description="定义内容")
    source_clause_id: Optional[str] = Field(default=None, description="来源条款ID")


class ContractRefLaw(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    law_doc_id: str
    law_name: Optional[str] = None
    article_no: Optional[str] = None
    article_range: Optional[str] = None


class ContractObligation(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: str
    from_party_id: str
    to_party_id: Optional[str] = None
    type: ObligationType = ObligationType.OTHER
    description: str
    amount: Optional[float] = None
    currency: Optional[str] = Field(default="CNY")
    frequency: Optional[str] = Field(default=None, description="频率，如 monthly/once")
    due_date: Optional[str] = None
    condition: Optional[str] = Field(default=None, description="履行条件")


class ContractRight(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: str
    party_id: str
    type: RightType = RightType.OTHER
    description: str


class ContractCondition(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: str
    type: ConditionType
    description: str


class ContractRemedy(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: str
    type: RemedyType
    description: str
    amount: Optional[float] = None
    multiplier: Optional[float] = Field(
        default=None,
        description="倍数，如违约金为月工资3倍则为3.0",
    )


class ContractClause(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: str
    clause_no: Optional[str] = Field(default=None, description="条号，如 第一条")
    title: Optional[str] = Field(default=None, description="小标题")
    category: ContractClauseCategory = ContractClauseCategory.OTHER

    text: str = Field(..., description="条款原文")

    ref_laws: List[ContractRefLaw] = Field(default_factory=list)
    obligations: List[ContractObligation] = Field(default_factory=list)
    rights: List[ContractRight] = Field(default_factory=list)
    conditions: List[ContractCondition] = Field(default_factory=list)
    remedies: List[ContractRemedy] = Field(default_factory=list)


class ContractPresenceSummary(BaseModel):
    """
    合同要素覆盖情况（以劳动合同为主做 v1）
    不同 contract_type 可以择用不同字段，但统一放在这个对象里。
    """
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    has_party_info: bool = False
    has_term: bool = False
    has_work_content: bool = False
    has_work_place: bool = False
    has_work_time_and_rest: bool = False
    has_salary_and_payment: bool = False
    has_social_insurance: bool = False
    has_labor_protection: bool = False
    has_confidentiality: bool = False
    has_non_compete: bool = False
    has_ip_ownership: bool = False
    has_dispute_resolution: bool = False
    has_termination_conditions: bool = False


class ContractRisk(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: str
    source_clause_ids: List[str] = Field(
        default_factory=list,
        description="触发该风险的条款ID列表",
    )
    risk_type: RiskType = RiskType.OTHER
    severity: RiskSeverity = RiskSeverity.RISK_MEDIUM
    description: str = Field(..., description="风险描述，面向律师/用户")

    rule_id: Optional[str] = Field(
        default=None,
        description="命中的规则ID（对接规则引擎）",
    )
    ref_laws: List[ContractRefLaw] = Field(
        default_factory=list,
        description="与该风险相关的法律条文引用",
    )


class ContractAttachment(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: str
    title: str
    type: Optional[str] = Field(default=None, description="附件类型，如 job_description 等")
    related_clause_ids: List[str] = Field(default_factory=list)


class ContractGraph(BaseModel):
    """
    合同图谱：ContractGraph
    """
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )

    contract_id: str = Field(..., description="内部合同ID")
    schema_version: str = Field(default="0.1.0", description="schema 版本号")

    raw_text: str = Field(..., description="合同全文（或主体文本）")

    meta: ContractMeta
    parties: List[ContractParty] = Field(default_factory=list)
    definitions: List[ContractDefinition] = Field(default_factory=list)
    clauses: List[ContractClause] = Field(default_factory=list)

    presence_summary: ContractPresenceSummary = Field(
        default_factory=ContractPresenceSummary
    )

    risks: List[ContractRisk] = Field(default_factory=list)
    attachments: List[ContractAttachment] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def check_schema_version(cls, v: str) -> str:
        if not v:
            raise ValueError("schema_version 不能为空")
        return v


# =========================
# Mermaid 图生成工具
# =========================

def _format_literal_values(values: tuple[object, ...]) -> str:
    return " | ".join(str(value) for value in values)


def _format_type(annotation: object) -> str:
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation)]
        return " | ".join(_format_type(arg) for arg in args)
    if origin is list or origin is List:
        args = get_args(annotation)
        inner = _format_type(args[0]) if args else "Any"
        return f"list[{inner}]"
    if origin is Literal:
        return _format_literal_values(get_args(annotation))
    if annotation is None or annotation is type(None):
        return "None"
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _render_enum(enum_cls: type[Enum]) -> list[str]:
    lines = [f"    class {enum_cls.__name__} {{", "        <<Enumeration>>"]
    for member_name, member in enum_cls.__members__.items():
        lines.append(f"        {member_name}: str = '{member.value}'")
    lines.append("    }")
    return lines


def _render_model(model: type[BaseModel]) -> list[str]:
    lines = [f"    class {model.__name__} {{"]
    for field_name, model_field in model.model_fields.items():
        type_name = _format_type(model_field.annotation)
        lines.append(f"        {field_name}: {type_name}")
    lines.append("    }")
    return lines


def build_presence_class_diagram() -> str:
    lines: list[str] = ["classDiagram"]
    lines.extend(_render_enum(ContractType))

    for model in (PresenceDefinition, PresenceResult, ContractDoc):
        lines.extend(_render_model(model))

    lines.append("    ContractDoc \"1\" --> \"*\" PresenceResult : presence_results")
    lines.append("    PresenceResult --> ContractType : contract_type")
    lines.append("    PresenceDefinition --> ContractType : contract_type")
    lines.append(
        "    PresenceResult ..> PresenceDefinition : presence_id + contract_type"
    )

    return "\n".join(lines)


def _sanitize_mermaid_id(value: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in value)


def build_contract_presence_relationships(
    definitions: Iterable[PresenceDefinition] = ALL_PRESENCE_DEFINITIONS,
) -> str:
    lines = ["flowchart TD"]
    contract_nodes: dict[ContractType, str] = {}

    for contract_type in ContractType:
        node_id = f"{_sanitize_mermaid_id(contract_type.value)}_type"
        contract_nodes[contract_type] = node_id
        lines.append(f"    {node_id}[{contract_type.value}]")

    rendered_presence_nodes: set[str] = set()
    for definition in definitions:
        presence_node = (
            f"{_sanitize_mermaid_id(definition.contract_type.value)}_"
            f"{_sanitize_mermaid_id(definition.id)}"
        )
        if presence_node not in rendered_presence_nodes:
            label = f"{definition.label} ({definition.id})"
            lines.append(f"    {presence_node}([{label}])")
            rendered_presence_nodes.add(presence_node)

        relation_label = "required" if definition.required else "optional"
        lines.append(
            f"    {contract_nodes[definition.contract_type]} -->|{relation_label}| {presence_node}"
        )

    return "\n".join(lines)


def generate_mermaid_file(
    output_path: Union[str, Path] = Path(__file__).with_name(
        "contracts_presence_schema.mmd"
    ),
) -> Path:
    output_path = Path(output_path)
    diagrams = [
        "```mermaid",
        build_presence_class_diagram(),
        "```",
        "",
        "```mermaid",
        build_contract_presence_relationships(),
        "```",
    ]
    output_path.write_text("\n".join(diagrams), encoding="utf-8")
    print(f"Mermaid diagrams written to {output_path}")
    return output_path


if __name__ == "__main__":
    generate_mermaid_file()
