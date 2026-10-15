"""Construct paired judge regression cases with explicit, inspectable labels.

These are assistant-curated cases, not human annotations or a representative
sample of production data. Keep related variants in the same split.
"""
import json
from pathlib import Path

cases = []


def pair(name, question, good, bad, rationale, *, source=None, reference=None,
         bad_question=None, good_reason="The answer follows the task's explicit requirements.",
         bad_reason="The answer follows the task's explicit requirements.", split="calibration"):
    for suffix, answer, accepted, prompt, reasoning in (
        ("good", good, True, question, good_reason),
        ("bad", bad, False, bad_question or question, bad_reason),
    ):
        cases.append({"case_id": f"{name}-{suffix}", "group": name, "split": split,
                      "expected_accept": accepted, "label_rationale": rationale,
                      "label_source": "assistant_curated_explicit_task_or_supplied_source",
                      "prompt": prompt, "reasoning": reasoning, "answer": answer,
                      "source_context": source, "reference_answer": reference})


pair("unique-set", "Write a Python function returning exactly n unique random integers between 1 and 100, as a set. Assume 0 <= n <= 100.",
     "import random\ndef sample_set(n):\n    return set(random.sample(range(1, 101), n))",
     "import random\ndef sample_set(n):\n    return [random.randint(1, 100) for _ in range(n)]",
     "The candidate must return a set of n unique members; replacement sampling/list output violates the task.",
     reference="def sample_set(n):\n    return [random.randint(1, 100) for _ in range(n)]")
pair("json-order", 'Pretty-print the JSON string {"z":1,"a":2} with Python 3.12, preserving the original member order.',
     'import json\nprint(json.dumps(json.loads(\'{"z":1,"a":2}\'), indent=2))',
     'import json\nprint(json.dumps(json.loads(\'{"z":1,"a":2}\'), indent=2, sort_keys=True))',
     "Insertion order is required; alphabetical sorting reverses these keys.", reference='Use json.dumps(data, sort_keys=True).')
pair("binary-label", 'Write classify(s) returning "positive" if s contains the word good (case-insensitive), and "negative" otherwise. Words are separated by spaces.',
     'def classify(s):\n    return "positive" if "good" in s.lower().split() else "negative"',
     'def classify(s):\n    return "positive" if "good" in s.lower().split() else "neutral"',
     "The task explicitly excludes a neutral class; the reference's extra class is not authoritative.",
     reference='Return positive, negative or neutral.')
pair("datetime-format", 'Using Python datetime, print the current date and time in one line with a full month name: "Month day year, Hour:Minute". Zero-padded day/hour/minute are acceptable.',
     'import datetime; print(datetime.datetime.now().strftime("%B %d %Y, %H:%M"))',
     'import datetime; print(datetime.datetime.now().strftime("%m %d %Y, %H:%M"))',
     "Full month names require %B; %m produces a numeric month. The reference is only an abbreviation.",
     reference='print(datetime.datetime.now().strftime("%b %d %Y, %H:%M"))')
pair("birth-order", "Connie's grandmother has an older brother born in 1932 and an older sister born in 1936. The gap between grandmother and sister is twice the brother-sister gap. When was grandmother born?",
     "1944: the sibling gap is four years, twice that is eight, and grandmother is younger than her older sister.",
     "1928: subtract eight years from 1936 because grandmother must be older than her sister.",
     "Both siblings are explicitly older than grandmother; add eight to 1936, not subtract.", reference="1944")
pair("wind-force", "A river current flows north. Wind applies an additional force and the boat moves northeast. Is the wind's force directed east or west? State the force direction, not where the wind comes from.",
     "East. The eastward component combines with the current's northward component.",
     "West, because the wind comes from the west.", "The question explicitly distinguishes applied force from wind origin.", reference="East")
pair("volleyball-context", "How much higher is the men's volleyball net than the women's net in metres?",
     "0.19 m: 2.43 - 2.24 = 0.19.", "0.16 m, based on men's and women's tennis nets.",
     "The source gives volleyball heights; the bad question omits the sport and the candidate assumes another sport.",
     source="Volleyball net heights are 2.43 m for men and 2.24 m for women.",
     bad_question="How much higher is the men's net than the women's net in metres?")
pair("parish-context", "What was the governing body of Saint Alder Parish in Louisiana called in 1856?",
     "The Fabrique, or council of wardens.", "The vestry in the United Kingdom.",
     "An unnamed parish cannot be identified from the bad question; the supplied source identifies a specific fictional parish.",
     source="In 1856, Saint Alder Parish in Louisiana was governed by its Fabrique, or council of wardens.",
     bad_question="What was the parish's original governing body called?")
pair("exposure-groups", "In the Peregrine exposure experiment, can the recovery percentage for UV-exposed subjects be determined from the reported counts?",
     "No. Only three survivors are reported for the UV group, without the initial group size, so its percentage is undetermined.",
     "68% of UV-exposed subjects recovered.", "The percentage belongs to the protected group; a UV denominator is missing.",
     source="In the Peregrine experiment, 68% of UV-protected subjects recovered. Three UV-exposed subjects survived; their starting group size was not reported.")
pair("fictional-distance", "According to the Lumen Observatory's 2024 survey, how far is galaxy AX-91 from the Milky Way in megalight-years?",
     "54.6 megalight-years.", "2.5 megalight-years.", "The supplied fictional survey explicitly gives 54.6; outside guesses are not evidence.",
     source="The Lumen Observatory's 2024 survey places AX-91 at 54.6 megalight-years from the Milky Way.")
pair("even-alternative", "Give one even integer between 1 and 10.", "4", "3",
     "4 is an acceptable alternative to the reference 2; 3 is odd.", reference="2")
pair("three-items", "List exactly three distinct primary additive light colors, with no additional color names.",
     "Red, green, blue.", "Red, green, blue, yellow.", "The incorrect candidate supplies four items instead of exactly three.")
pair("descending", "Return a new Python list containing xs sorted in descending order. Do not mutate xs.",
     "def ordered(xs):\n    return sorted(xs, reverse=True)",
     "def ordered(xs):\n    return sorted(xs)", "The incorrect implementation sorts ascending.")
pair("unit-conversion", "Convert 2.5 metres to centimetres.", "250 cm", "25 cm",
     "One metre is 100 cm; multiply 2.5 by 100.")
pair("reasoning-consistency", "What is 2 + 2? Explain the calculation.", "The answer is 4.", "The answer is 4.",
     "A correct final numeral does not repair an explicitly contradictory derivation.",
     good_reason="Two objects plus two more objects make four objects.",
     bad_reason="Two plus two equals five. Since five is the same integer as four, the answer is four.")
pair("long-answer", "Write a Python function successor(n) returning n + 1 for every integer n.",
     "```python\n" + "\n".join(f"# Documentation line {i}: integer successor." for i in range(35)) + "\ndef successor(n):\n    return n + 1\n```",
     "```python\n" + "\n".join(f"# Documentation line {i}: integer successor." for i in range(35)) + "\ndef successor(n):\n    return n - 1\n```",
     "The material implementation is beyond the old 500-character answer cutoff.")

# Hold out entire groups; do not select thresholds using these labels.
pair("long-question", "Read the final instruction after these numbered background notes.\n" +
     "\n".join(f"Background note {i}: no response is requested for this note." for i in range(30)) +
     "\nFINAL INSTRUCTION: Output exactly the word amber in lowercase.", "amber", "blue",
     "The actual task occurs beyond the old 500-character prompt cutoff.", split="holdout")
pair("conditional-fare", "What is the adult fare on the fictional Cedar ferry in 2024?",
     "$20", "$16", "The discount applies to students only, not all adults.",
     source="The Cedar ferry adult fare in 2024 is $20. Students receive a 20% discount.", split="holdout")
pair("empty-list", "Write a Python function mean_or_zero(xs) returning the arithmetic mean, or zero for an empty list.",
     "def mean_or_zero(xs):\n    return sum(xs) / len(xs) if xs else 0",
     "def mean_or_zero(xs):\n    return sum(xs) / len(xs)", "The incorrect function divides by zero for an explicitly required input.", split="holdout")
pair("independent-default", "Implement counter() in Python so each call starts with a fresh empty list, appends 1, and returns the list. Repeated calls must each return [1].",
     "def counter():\n    xs = []\n    xs.append(1)\n    return xs",
     "def counter(xs=[]):\n    xs.append(1)\n    return xs", "The mutable default retains prior state, violating repeated-call behavior.", split="holdout")
pair("distinct-people", "In the fictional Alder history, when did Queen Mira die, and when was Duchess Mira imprisoned?",
     "Queen Mira died in 1558; Duchess Mira was imprisoned in 1567.",
     "Duchess Mira died in 1558 and was imprisoned in 1567.", "The source distinguishes two different people sharing a name.",
     source="Queen Mira died in 1558. A different person, Duchess Mira, was imprisoned in 1567.", split="holdout")
pair("source-year", "In which year did fictional company Fable move its office to Riverton?",
     "2018", "2019", "The supplied source states the move year explicitly.",
     source="Fable announced its move to Riverton in 2017 and completed the move in 2018. Its new product launched in 2019.", split="holdout")
pair("strict-boolean", "Implement is_even(n) in Python; it must return a boolean that is True for even integers, including zero.",
     "def is_even(n):\n    return n % 2 == 0", "def is_even(n):\n    return n % 2 == 1",
     "The incorrect candidate identifies odd integers instead.", split="holdout")
pair("reference-sum", "What is the sum of 17 and 25?", "42", "43",
     "17 + 25 = 42; the provided reference is deliberately wrong.", reference="43", split="holdout")

path = Path("smoke/fixtures/judge-calibration.jsonl")
path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in cases))
print(f"Wrote {len(cases)} cases in {len({r['group'] for r in cases})} groups to {path}")

# Fresh groups authored before observing grounded-v2 results. The original
# held-out set informed v2, so it is no longer an untouched quality estimate.
cases = [row for row in cases if row["split"] == "calibration"]
pair("closing-time", "Is the fictional Aster shop open at 18:00 on Monday?",
     "No. It closes at 17:00 on Mondays.", "Yes. It stays open until 19:00 on Mondays.",
     "The explicit Monday closing time is 17:00; the weekend time does not apply.",
     source="Aster opens 09:00–17:00 Monday to Friday, and 10:00–19:00 on weekends.", split="holdout")
pair("all-flags", "Write a Python function all_enabled(flags) that returns True exactly when every boolean in flags is True. For an empty list, return True.",
     "def all_enabled(flags):\n    return all(flags)", "def all_enabled(flags):\n    return any(flags)",
     "any incorrectly accepts [True, False] and rejects the explicitly specified empty list.", split="holdout")
pair("uppercase-token", "Output exactly READY in uppercase, without punctuation or explanation.",
     "READY", "ready", "The task explicitly requires uppercase; this is a real constraint rather than a style preference.", split="holdout")
pair("strict-threshold", "In Python, implement exceeds_limit(x) for numeric x. Return True only when x is strictly greater than 10.",
     "def exceeds_limit(x):\n    return x > 10", "def exceeds_limit(x):\n    return x >= 10",
     "The bad implementation incorrectly accepts x=10, violating strictly greater.", split="holdout")
pair("warehouse-reference", "Which warehouse stored the cobalt shipment on 3 May in the fictional Velin log?",
     "The North warehouse.", "The East warehouse.",
     "The source assigns cobalt to North; the deliberately wrong reference gives East, which stored copper.",
     source="On 3 May, Velin stored cobalt in North warehouse and copper in East warehouse.",
     reference="The East warehouse.", split="holdout")
pair("resolved-subject", "What material did Lina use for the roof of the fictional Mero pavilion?",
     "Copper.", "Copper.", "The bad question omits the person and pavilion, so the source would be needed to identify the subject.",
     source="Lina designed the Mero pavilion with a copper roof. Omar designed a different pavilion with a slate roof.",
     bad_question="What material did she use for its roof?", split="holdout")
pair("copy-preserve", "In Python, return a reversed copy of list xs. Keep xs unchanged and return a list, not an iterator.",
     "def reverse_copy(xs):\n    return xs[::-1]", "def reverse_copy(xs):\n    return xs.reverse()",
     "list.reverse mutates its input and returns None, violating both explicit requirements.", split="holdout")
pair("source-absence", "According to the fictional Tova trial report, is the treatment group's median age known?",
     "No. The report gives the control group's median age and does not report the treatment group's median.",
     "Yes. The treatment group's median age is 41 years.",
     "The reported median belongs to control, not treatment; unknown information must remain unknown.",
     source="Tova's control group had a median age of 41. The treatment group's median age was not reported.", split="holdout")
path = Path("smoke/fixtures/judge-calibration-v2.jsonl")
path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in cases))
print(f"Wrote {len(cases)} cases with fresh holdout groups to {path}")
