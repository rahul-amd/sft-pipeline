"""Launch vLLM with an experiment-local, explicit full-context Clef encoder.

The pinned runtime's ClefFlashStrategy caps state at 16,384 tokens despite
the checkpoint backbone's 262,144 position limit. Raise only this adapter cap
and reject oversized inputs instead of silently truncating. No weights change.
"""
import json
import sys
from vllm.entrypoints.openai.decisions.strategies import ClefFlashStrategy, StructuredDecisionError

original_encode = ClefFlashStrategy.encode
original_limit = ClefFlashStrategy.MAX_LENGTH
ClefFlashStrategy.MAX_LENGTH = 262144


def full_encode(self, questions, state, prefix=None):
    empty, _, _ = original_encode(self, questions, "", prefix)
    expected = len(empty) + len(self._tokens(state))
    if expected > self.max_length:
        raise StructuredDecisionError(f"Full input requires {expected} tokens, exceeding {self.max_length}; truncation is disabled")
    result = original_encode(self, questions, state, prefix)
    if len(result[0]) != expected:
        raise RuntimeError("Encoder dropped tokens despite full-context policy")
    return result


ClefFlashStrategy.encode = full_encode
print(json.dumps({"adapter_original_cap":original_limit,"adapter_cap":ClefFlashStrategy.MAX_LENGTH,
                  "truncation":"forbidden; oversized requests fail"}),flush=True)
from vllm.entrypoints.cli.main import main

if __name__ == '__main__':
    main()
