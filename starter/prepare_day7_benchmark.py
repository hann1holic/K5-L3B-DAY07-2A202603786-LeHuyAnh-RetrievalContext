"""Build Day 7's 20 answerable cases from Day 6 and add 4 abstention cases.

Absence labels were checked against the full Day 6 indexed corpus. A retrieval
miss alone is never considered proof that the source lacks evidence.
"""
import json
from pathlib import Path
from day6_adapter import load_questions

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "day7_questions.json"

ABSENCE_CASES = [
    {"id": "U01", "split": "dev", "question": "What ISBN does the encyclical give for its printed edition?",
     "absence_reason": "The complete indexed corpus contains no ISBN or print-edition identifier."},
    {"id": "U02", "split": "dev", "question": "What minimum age does the encyclical prescribe for opening a social media account?",
     "absence_reason": "The source calls for legislators to set age limits but does not specify a numeric minimum age."},
    {"id": "U03", "split": "test", "question": "How many euros of public funding does the encyclical prescribe for AI research?",
     "absence_reason": "The complete indexed corpus gives no euro-denominated public funding amount for AI research."},
    {"id": "U04", "split": "test", "question": "Which commercial AI product does the encyclical recommend schools use as their preferred teaching assistant?",
     "absence_reason": "The education and AI passages do not endorse a named commercial teaching product."},
]

# Fresh answerable questions from pages outside Day 6's original 5-question
# test set (which was concentrated on pages 35–36 and already inspected).
FRESH_TEST = [
    {"id": "Q16", "question": "Why does the text say job insecurity is especially devastating for young people?",
     "page": 29, "evidence": "a crucial sphere in which identity is formed"},
    {"id": "Q17", "question": "What control over personal data does the document say should be restored to individuals?",
     "page": 31, "evidence": "the ability to decide how it is used, by whom and for whose\nbenefit"},
    {"id": "Q18", "question": "When did the Treaty on the Prohibition of Nuclear Weapons come into force, and how many countries supported it?",
     "page": 33, "evidence": "came into force in 2021\nwith the support of over seventy countries"},
    {"id": "Q19", "question": "What should an international framework do about the technological arms race and civilian protection?",
     "page": 34, "evidence": "curb the technological arms race and ensure robust protection for civilians"},
    {"id": "Q20", "question": "How does the document say tax systems should distribute the burden of social justice?",
     "page": 28, "evidence": "tax systems that lighten\nthe burden on the weakest and ask for more from those with greater resources"},
]


def main():
    answerable = []
    for q in load_questions():
        # Day 6's answerable test questions had already been explored. Keep its
        # dev set, but replace its test set with freshly authored cases.
        if q["split"] != "dev":
            continue
        gold = [{"source_id": "magnifica_humanitas", "source": q["gold_source"][min(i, len(q["gold_source"])-1)],
                 "page": q["gold_pages"][min(i, len(q["gold_pages"])-1)], "evidence": quote}
                for i, quote in enumerate(q["gold_evidence"])]
        answerable.append({"id": q["id"], "split": q["split"], "question": q["question"],
                           "answerable": True, "gold": gold, "filter": q.get("filters", {}),
                           "expected_answer": "Evidence: " + " / ".join(q["gold_evidence"])})
    cases = answerable + [{"id": q["id"], "split": "test", "question": q["question"],
               "answerable": True, "gold": [{"source_id": "magnifica_humanitas",
               "source": "Magnifica_Humanitas.pdf", "page": q["page"], "evidence": q["evidence"]}],
               "filter": {}, "expected_answer": q["evidence"]} for q in FRESH_TEST]
    cases += [{**q, "answerable": False, "expected_behavior": "search_and_abstain",
                           "audit_required": False} for q in ABSENCE_CASES]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} questions ({sum(q['split']=='dev' for q in cases)} dev, {sum(q['split']=='test' for q in cases)} test).")
    print("Absence labels U01–U04 were checked against the full indexed corpus.")


if __name__ == "__main__":
    main()
