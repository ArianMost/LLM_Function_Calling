from state_machine.token_filter import TokenFilterEngine
from llm_sdk import Small_LLM_Model


class ConstrainedDecoder:
    def __init__(
        self,
        llm: Small_LLM_Model,
        engine: TokenFilterEngine
    ):
        self.llm = llm
        self.engine = engine

    def decode(self, prompt: str):
        generated_tokens = self.llm.encode(prompt).squeeze().tolist()
        generated_output = []

        if isinstance(generated_tokens, int):
            generated_tokens = [generated_tokens]

        while not self.engine.state_machine.is_done():
            logits = self.llm.get_logits_from_input_ids(generated_tokens)
            allowed = self.engine.allowed_tokens()
            next_token = self._select_next_token(logits, allowed)
            generated_tokens.append(next_token)
            generated_output.append(next_token)
            self.engine.consume_generated_token(next_token)

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

