from llm_sdk import Small_LLM_Model
from ..models.functions_definition import FunctionDefinition


class FunctionSelector:
    def __init__(self, llm: Small_LLM_Model, functions: list[FunctionDefinition]) -> None:
        self.llm = llm

        self.candidate_functions: list[tuple[list[int], FunctionDefinition]] = [
            (self._extract_token_code(f'"{fn.name}"'), fn) for fn in functions
        ]

    def _extract_token_code(self, text: str) -> list[int]:
        tokens = self.llm.encode(text).squeeze().tolist()
        if isinstance(tokens, int):
            return [tokens]
        return tokens

    def _make_context(self, prompt: str) -> str:
        catalogue = "\n".join(f"- {fn.name}: {fn.description}" for _, fn in self.candidate_functions)
        return (
            f"These are the available functions:\n{catalogue}\n\n"
            f'This is the request: "{prompt}"\n'
            "Find the function name: "
        )

    def select(self, prompt: str) -> FunctionDefinition:
        def score(id):
            return logits[id]

        context = self._extract_token_code(self._make_context(prompt))
        buffer: list[int] = []

        while True:
            for sequence, fn in self.candidate_functions:
                if sequence == buffer:
                    return fn

            # Tokens that keep us on at least one valid candidate path
            allowed = {
                sequence[len(buffer)]
                for sequence, _ in self.candidate_functions
                if len(sequence) > len(buffer) and sequence[: len(buffer)] == buffer
            }
            if not allowed:
                raise RuntimeError("Prompt did not resolve to any defined function")

            logits = self.llm.get_logits_from_input_ids(context + buffer)

            buffer.append(max(allowed, key=score))
