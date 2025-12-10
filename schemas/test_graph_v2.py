from graphs import CaseGraph, CaseMeta, CaseParty, CaseClaim
import json


def main():
    cg = CaseGraph(
        case_id="case_001",
        raw_text="2019年我借给朋友李四10万元，对方一直不还，现在想起诉要回本息。",
        summary="张三向李四出借10万元未获清偿，拟起诉要求返还本息。",
        meta=CaseMeta(
            case_type="民间借贷纠纷",
            cause="民间借贷纠纷",
        ),
        parties=[
            CaseParty(
                id="P1",
                role="原告",
                name="张三",
                type="natural_person",
                attributes={"is_lender": True},
            ),
            CaseParty(
                id="P2",
                role="被告",
                name="李四",
                type="natural_person",
                attributes={"is_borrower": True},
            ),
        ],
        claims=[
            CaseClaim(
                id="C1",
                claimant_id="P1",
                respondent_id="P2",
                type="支付借款本息",
                amount=100000.0,
                currency="CNY",
                other_demands=["自起诉之日起按同期LPR计算利息"],
            )
        ],
    )

    print(json.dumps(cg.model_dump(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
