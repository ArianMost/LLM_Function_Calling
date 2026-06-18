import json
from pathlib import Path


class VocabularyIndex:
    def __init__(self, vocab_path: str):
        with Path(vocab_path).open("r", encoding="utf-8") as f:
            vocab: dict[str, int] = json.load(f)

        self.token_to_id = vocab
        self.id_to_token = {
            token_id: token
            for token, token_id in vocab.items()
        }

    def token_string(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, "")

    def token_id(self, token: str) -> int | None:
        return self.token_to_id.get(token)

    def all_tokens(self) -> dict[int, str]:
        return self.id_to_token
