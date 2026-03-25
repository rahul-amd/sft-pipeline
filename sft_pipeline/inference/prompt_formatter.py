"""
Format prompts for the teacher model.

Applies a system prompt that instructs the model to produce structured
reasoning traces with the configured delimiters.
"""
from __future__ import annotations

from sft_pipeline.config import ReasoningDelimiters

_SYSTEM_TEMPLATE = """\
You are an expert reasoning assistant. For every question or task:

1. Think through the problem step by step inside {think_start}...{think_end} tags.
2. Provide your final answer inside {answer_start}...{answer_end} tags.

Always use this exact format:
{think_start}
[Your step-by-step reasoning here]
{think_end}
{answer_start}
[Your concise final answer here]
{answer_end}"""


def build_chat_messages(
    prompt: str,
    delimiters: ReasoningDelimiters,
) -> list[dict[str, str]]:
    """
    Build the chat message list for a single prompt.
    Returns a list suitable for passing to an OpenAI-compatible /chat/completions endpoint
    or vLLM's apply_chat_template.
    """
    system = _SYSTEM_TEMPLATE.format(
        think_start=delimiters.think_start,
        think_end=delimiters.think_end,
        answer_start=delimiters.answer_start,
        answer_end=delimiters.answer_end,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def apply_chat_template(
    tokenizer,
    prompt: str,
    delimiters: ReasoningDelimiters,
    add_generation_prompt: bool = True,
) -> str:
    """
    Apply a HuggingFace tokenizer's chat template to produce a formatted string.
    Used when vLLM is in offline batch mode and needs a raw string input.
    """
    messages = build_chat_messages(prompt, delimiters)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
