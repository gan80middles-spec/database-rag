"""
合同要素覆盖情况的规范定义（schema 层），
按合同类型划分。这里不使用 Pydantic，只是单纯的配置常量。

业务逻辑：
- contract_type 对应一个 key，例如 "buy_sell" / "lease" / "labor" / "nda" / "outsourcing" / "software"；
- 每个 key 对应一个列表，每个元素定义一个 presence 要素：
  - id: 机器可读 ID，例如 "party_info"、"subject"、"term" 等；
  - label: 人类可读名称，将来可以直接展示给用户；
  - required: 是否为该合同类型下的必备条款。
"""

PRESENCE_SCHEMAS = {
    "buy_sell": [
        {"id": "party_info", "label": "当事人信息（出卖人/买受人）", "required": True},
        {"id": "subject", "label": "标的物名称/型号", "required": True},
        {"id": "quantity", "label": "数量与计量方式", "required": True},
        {"id": "quality", "label": "质量标准/技术规格", "required": True},
        {"id": "price_payment", "label": "价款与结算方式", "required": True},
        {"id": "delivery", "label": "交付期限/地点/方式", "required": True},
        {"id": "inspection", "label": "检验标准与方法/验收", "required": True},
        {"id": "packaging", "label": "包装与随附资料", "required": False},
        {"id": "risk_title_transfer", "label": "风险转移/所有权转移", "required": False},
        {"id": "after_sales", "label": "质保/售后", "required": False},
        {"id": "ip_warranty", "label": "知识产权不侵权保证（如适用）", "required": False},
        {"id": "breach_liability", "label": "违约责任/损害赔偿", "required": True},
        {"id": "dispute_resolution", "label": "争议解决（适用法/仲裁/法院）", "required": True},
    ],
    "lease": [
        {"id": "party_info", "label": "出租人/承租人信息", "required": True},
        {"id": "leased_property", "label": "租赁物名称/数量/状况", "required": True},
        {"id": "purpose", "label": "用途与使用限制", "required": True},
        {"id": "term", "label": "租赁期限", "required": True},
        {"id": "rent_payment", "label": "租金及支付期限/方式", "required": True},
        {"id": "delivery_return", "label": "交付/期满返还与原状恢复", "required": True},
        {"id": "maintenance", "label": "维修责任与费用分担", "required": True},
        {"id": "deposit", "label": "押金/担保（如有）", "required": False},
        {"id": "sublease_assignment", "label": "转租/转借/转让限制", "required": False},
        {"id": "risk_liability", "label": "风险承担/保险（如适用）", "required": False},
        {"id": "breach_termination", "label": "违约与解除/提前终止", "required": True},
        {"id": "dispute_resolution", "label": "争议解决", "required": True},
    ],
    "labor": [
        {"id": "party_info", "label": "用人单位/劳动者信息", "required": True},
        {"id": "term", "label": "合同期限/试用期（如有）", "required": True},
        {"id": "job_and_location", "label": "工作内容/岗位与地点", "required": True},
        {"id": "hours_leave", "label": "工作时间/休息休假", "required": True},
        {"id": "compensation", "label": "劳动报酬（工资标准/发放周期）", "required": True},
        {"id": "social_insurance", "label": "社会保险/公积金约定", "required": True},
        {"id": "safety_conditions", "label": "劳动保护/条件/职业危害防护", "required": True},
        {"id": "training", "label": "培训条款（如约定）", "required": False},
        {"id": "confidentiality_noncompete", "label": "保密/竞业限制（如约定）", "required": False},
        {"id": "discipline", "label": "劳动纪律/规章制度遵守", "required": False},
        {"id": "termination", "label": "解除/终止与经济补偿", "required": True},
        {"id": "dispute_resolution", "label": "争议解决（仲裁/诉讼）", "required": True},
    ],
    "nda": [
        {"id": "party_info", "label": "当事人信息", "required": True},
        {"id": "confidential_definition", "label": "保密信息的范围/定义", "required": True},
        {"id": "use_scope", "label": "使用范围/目的限制", "required": True},
        {"id": "exceptions", "label": "保密义务例外（公开/已知/独立获得等）", "required": True},
        {"id": "security_measures", "label": "安全措施/访问控制", "required": False},
        {"id": "term_duration", "label": "保密期限/存续期", "required": True},
        {"id": "return_destroy", "label": "资料返还/销毁", "required": True},
        {"id": "third_party", "label": "第三方披露与传递限制", "required": False},
        {"id": "ip_ownership", "label": "知识产权归属与不授予条款", "required": False},
        {"id": "breach_remedy", "label": "违约责任/禁令救济/损害赔偿", "required": True},
        {"id": "dispute_resolution", "label": "争议解决", "required": True},
    ],
    "outsourcing": [
        {"id": "party_info", "label": "委托方/服务方信息", "required": True},
        {"id": "scope_deliverables", "label": "工作范围/交付成果与规格", "required": True},
        {"id": "timeline_milestones", "label": "进度/里程碑/服务期限", "required": True},
        {"id": "acceptance", "label": "验收标准与流程", "required": True},
        {"id": "fees_settlement", "label": "费用/结算/发票", "required": True},
        {"id": "change_control", "label": "变更管理（需求/范围/价格）", "required": False},
        {"id": "materials_ip", "label": "资料/工具/知识产权归属与许可", "required": True},
        {"id": "confidentiality", "label": "保密义务/数据合规", "required": True},
        {"id": "warranty_support", "label": "质量保证/维护支持（如有）", "required": False},
        {"id": "subcontracting", "label": "分包/人员更替限制", "required": False},
        {"id": "breach_liability", "label": "违约责任/赔偿/限责", "required": True},
        {"id": "termination", "label": "解除/终止与费用清算", "required": True},
        {"id": "dispute_resolution", "label": "争议解决", "required": True},
    ],
    "software": [
        {"id": "party_info", "label": "许可方/被许可方信息", "required": True},
        {"id": "license_grant", "label": "许可范围（地域/期限/方式/是否独占）", "required": True},
        {"id": "usage_restrictions", "label": "使用限制（并发/终端/不得反向工程等）", "required": True},
        {"id": "delivery_support", "label": "交付方式/安装部署/技术支持", "required": False},
        {"id": "updates_maintenance", "label": "升级/维护/服务级别（SLA）", "required": False},
        {"id": "fees_audit", "label": "费用/计费方式/审计权", "required": True},
        {"id": "ip_ownership", "label": "知识产权归属与不授予条款", "required": True},
        {"id": "data_protection", "label": "数据安全/个人信息/接口调用合规", "required": False},
        {"id": "oss_thirdparty", "label": "开源与第三方组件约束（如适用）", "required": False},
        {"id": "infringement_indemnity", "label": "侵权担保与赔偿", "required": True},
        {"id": "restricted_clauses_compliance", "label": "技术合同限制性条款合规（民法典864）", "required": True},
        {"id": "termination", "label": "终止/到期后的处置（停用/返还/销毁）", "required": True},
        {"id": "dispute_resolution", "label": "争议解决", "required": True},
    ],
}
