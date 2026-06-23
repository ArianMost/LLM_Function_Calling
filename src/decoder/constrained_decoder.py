import re
from typing import Callable, Optional
from ..state_machine.token_filter import TokenFilterEngine
from llm_sdk import Small_LLM_Model


class ConstrainedDecoder:
    def __init__(
        self,
        llm: Small_LLM_Model,
        engine: TokenFilterEngine
    ):
        self.llm = llm
        self.engine = engine

    def decode(
            self,
            prompt: str,
            max_tokens: int = 256,
            on_token: Optional[Callable[[str], None]] = None
        ) -> str:
        context = (
            f'{prompt}\n'
            "Respond only with the function arguments. "
            "For regex and replacement values, "
            "provide only 1 pattern.\n"
        )
        generated_tokens = self.llm.encode(context).squeeze().tolist()
        generated_output: list[int] = []
        if isinstance(generated_tokens, int):
            generated_tokens = [generated_tokens]

        steps = 0
        while not self.engine.state_machine.is_done():
            if steps >= max_tokens:
                raise RuntimeError(f"decode exceeded max_tokens")

            logits = self.llm.get_logits_from_input_ids(generated_tokens)
            allowed = self.engine.allowed_tokens()
            next_token = self._select_next_token(logits, allowed)

            generated_tokens.append(next_token)
            generated_output.append(next_token)
            self.engine.consume_generated_token(next_token)

            # Handles typing on terminal
            if on_token is not None:
                on_token(self.llm.decode([next_token]))
            steps += 1

        return self.llm.decode(generated_output)


    def _select_next_token(
        self,
        logits: list[float],
        allowed_tokens: list[int],
    ) -> int:
        if not allowed_tokens:
            raise ValueError("No valid tokens available")

        def token_score(token_id: int) -> float:
            return logits[token_id]

        return max(allowed_tokens, key=token_score)

