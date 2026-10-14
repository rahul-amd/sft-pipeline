# External sources and proposed comparison roles

Primary sources inspected on 2026-10-14. These are candidate baselines and evaluation resources, not experiments already run. Pin immutable revisions and inspect component terms before the actual study. No third-party performance numbers are imported into this sketch.

| Resource | Primary source | Role and boundary |
|---|---|---|
| Tülu 3 SFT mixture | [Dataset card](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture); [training recipe](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md) | Broad SFT baseline. Use the SFT data, not published full-post-training model scores. Card lists ODC-BY at collection level and explicitly notes varying component terms, including non-commercial portions. |
| SmolTalk | [Dataset card](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | Broad instruction mixture covering such tasks as rewriting, summarization and reasoning, with public components. Candidate dataset comparison; no uniform license claim is made here. |
| OpenThoughts3 | [Dataset card](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M); [paper](https://arxiv.org/abs/2506.04178) | Math/code/science reasoning baseline; card identifies QwQ-32B annotations and Apache-2.0 metadata. Do not equate reasoning specialization with coverage of a general instruction mixture. |
| LongAlign | [Official repository](https://github.com/THUDM/LongAlign); [paper](https://arxiv.org/abs/2401.18058) | Long-input instruction tuning baseline. Retrain under our controlled budget rather than compare published model scores. Source token lengths require recounting with the chosen student tokenizer. |
| HELMET | [Official repository](https://github.com/princeton-nlp/HELMET); [paper](https://arxiv.org/abs/2410.02694) | Held-out application-oriented long-context evaluation across categories. Fix model-judged scoring versions and isolate training overlap before reporting. |
| LongBench v2 | [Official repository](https://github.com/THUDM/LongBench) | Additional long-context reasoning evaluation. Pin v2 explicitly; the same repository also hosts original LongBench with a different task mix and protocol. A dataset split named `train` does not authorize using benchmark examples for our student training. |
| RULER | [Official repository](https://github.com/NVIDIA/RULER) | Related-family diagnostic. Because LC training includes RULER-style tasks, results need overlap disclosure and cannot carry the independent-transfer claim. |

The experiment design and proposed controls in `sections/evaluation.md` are our suggestions. Resource descriptions above come from their respective creators. A code license, dataset-card label or accessible download alone does not establish rights to redistribute all underlying source material.
