"""Probe real Stage 2 generation and grounding on curated source challenges.

All results, including rejected generated questions, are saved for inspection.
These are assistant-curated regressions, not representative quality estimates.
"""
import argparse
import json
import os
from pathlib import Path

from sft_pipeline.config import Stage2Config
from sft_pipeline.stages.stage2_generate import _call_generator, _check_grounding


CASES = [
    {"name": "missing_sport", "passage": "In indoor volleyball, men's nets are 2.43 m high and women's nets are 2.24 m high.",
     "questions": ["How high is the net?", "How much higher is the men's indoor volleyball net than the women's net?"],
     "expected": [False, True]},
    {"name": "unnamed_parish", "passage": "The parish governing body was originally called the Fabrique. The source does not identify the parish or its location.",
     "questions": ["What was the original name of the parish's governing body?"], "expected": [False]},
    {"name": "missing_denominator", "passage": "In the Luma experiment, Group A had a 20% response rate. Group B had seven responses, but its total population was not reported.",
     "questions": ["What percentage of Group B responded in the Luma experiment?",
                   "How many responses were recorded for Group B in the Luma experiment?"], "expected": [False, True]},
    {"name": "conditional_fact", "passage": "The fictional Orin Museum charges adults 10 credits. Students pay 6 credits only on Tuesdays and 10 credits on other days.",
     "questions": ["How much do students pay to enter the Orin Museum?",
                   "How much do students pay to enter the Orin Museum on Tuesdays?"], "expected": [True, True]},
]


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").unlink(missing_ok=True)
    cfg = Stage2Config(generator_endpoint=args.api_base, generator_model=args.model,
                       generator_api_key=os.environ[args.api_key_env], grounding_max_tokens=4096)
    cases = json.loads(Path(args.cases).read_text()) if args.cases else CASES
    (output / "cases.used.json").write_text(json.dumps(cases, indent=2)+"\n")
    results = []
    for case in cases:
        verdicts = _check_grounding(case["passage"], case["questions"], cfg)
        generated = _call_generator(case["passage"], 3, args.api_base, args.model, 0, 2048, cfg.generator_api_key)
        generated_verdicts = _check_grounding(case["passage"], generated, cfg)
        results.append({**case, "verdicts": verdicts,
                        "generated": [{"question": q, **v} for q, v in zip(generated, generated_verdicts)]})
        (output / "cases.json").write_text(json.dumps(results, indent=2)+"\n")
    errors = [{"case": c["name"], "question": q, "expected": expected, "accepted": v["accepted"]}
              for c in results for q, expected, v in zip(c["questions"], c["expected"], c["verdicts"])
              if v["accepted"] != expected]
    report = {"status": "completed", "challenge_passed": not errors, "errors": errors,
              "model": args.model, "api_base": args.api_base,
              "challenge_questions": sum(len(c["questions"]) for c in results),
              "generated": sum(len(c["generated"]) for c in results),
              "generated_accepted": sum(v["accepted"] for c in results for v in c["generated"]),
              "limitations": "Assistant-curated cases. Generated acceptance is the model's judgment, not an independent correctness label."}
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="SFT_API_KEY")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", help="Optional JSON array of source/question/expected-decision cases")
    main(parser.parse_args())
