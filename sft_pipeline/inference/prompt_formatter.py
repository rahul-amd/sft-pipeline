"""
Format prompts for the teacher model.

Builds the chat message list for an OpenAI-compatible /chat/completions call
or vLLM's offline apply_chat_template.  No delimiter instructions are injected —
the model responds in its native format (thinking models will naturally produce
<think>...</think> output without being told to).
"""
from __future__ import annotations


def build_chat_messages(prompt: str) -> list[dict[str, str]]:
    """
    Build the chat message list for a single prompt.

    Returns a bare [user] turn so each model uses its own default behavior.
    Thinking models (Qwen3, DeepSeek-R1, etc.) produce their chain-of-thought
    automatically; non-thinking models return a direct answer.
    No delimiter or format instructions are injected — the full raw response
    is stored by Stage 5 and parsed downstream.
    """
    return [{"role": "user", "content": prompt}]


def apply_chat_template(
    tokenizer,
    prompt: str,
    add_generation_prompt: bool = True,
) -> str:
    """
    Apply a HuggingFace tokenizer's chat template to produce a formatted string.
    Used when vLLM is in offline batch mode and needs a raw string input.
    """
    messages = build_chat_messages(prompt)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
