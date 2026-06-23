from ..decoder.vocabulary_index import VocabularyIndex
from llm_sdk import Small_LLM_Model
from .json_state_machine import JSONStateMachine
from .states import State


class TokenFilterEngine:
    def __init__(
            self,
            state_machine: JSONStateMachine,
            llm: Small_LLM_Model
    ) -> None:
        self.state_machine = state_machine
        self.llm = llm
        self.current_sequence: list[int] = []
        self.sequence_position: int = 0
        self.generated_value_tokens: list[int] = []
        self.vocabulary = VocabularyIndex(llm.get_path_to_vocab_file())
        self.string_started = False
        self.string_finished = False
        self._string_content: list[int] | None = None
        self.max_string_tokens = 40

    def allowed_tokens(self) -> list[int]:
        while self.state_machine.current_state == State.OPEN_PARAMETERS_OBJECT:
            self.state_machine.move_forward()

        if self.state_machine.current_state == State.PARAMETER_VALUE:
            return self._dynamic_allowed_tokens()

        if not self.current_sequence:
            self.current_sequence = self._build_current_sequence()
            self.sequence_position = 0

            if not self.current_sequence:
                raise RuntimeError(
                    f"No valid token sequence for state "
                    f"{self.state_machine.current_state}"
                )

        return [self.current_sequence[self.sequence_position]]

    def consume_generated_token(
        self,
        token: int,
    ) -> None:
        if self.current_sequence:
            self.sequence_position += 1

            if self.sequence_position >= len(self.current_sequence):
                self.current_sequence = []
                self.sequence_position = 0
                self.state_machine.move_forward()
            return

        if self.state_machine.current_state == State.PARAMETER_VALUE:
            self._consume_parameter_value(token)

    def _consume_parameter_value(
        self,
        token: int,
    ) -> None:
        parameter_type = self.state_machine.current_parameter_type()
        self.generated_value_tokens.append(token)
        decoded = self.llm.decode(self.generated_value_tokens)

        if parameter_type == "boolean":
            if decoded in ("true", "false"):
                self._reset_dynamic_state()
                self.state_machine.move_forward()
            return

        if parameter_type == "number":
            terminator = "," if self.state_machine.has_more_parameters() else "}"
            if token == self._extract_single_token(terminator):
                self._reset_dynamic_state()
                # Leave PARAMETER_VALUE and skip the PARAMETER_COMMA state to avoid having ,, between params
                self.state_machine.move_forward()
                self.state_machine.move_forward()
            return

        if parameter_type == "string":
            quote = self._extract_single_token('"')

            # Opening quote
            if not self.string_started:
                if token == quote:
                    self.string_started = True
                return

            # Closing quote
            if token == quote:
                self.string_started = False
                self.string_finished = True
                self._reset_dynamic_state()
                self.state_machine.move_forward()
            return

    def _build_current_sequence(self) -> list[int]:
        state = self.state_machine.current_state
        function_name = self.state_machine.function_definition.name
        current_parameter = self.state_machine.current_parameter()

        match state:
            case State.START:
                return self._extract_token_code("{")

            case State.OPEN_ROOT_OBJECT:
                return self._extract_token_code('"name"')

            case State.NAME_KEY:
                return self._extract_token_code(":")

            case State.NAME_COLON:
                return self._extract_token_code(f'"{function_name}"')

            case State.FUNCTION_NAME:
                return self._extract_token_code(",")

            case State.COMMA_AFTER_NAME:
                return self._extract_token_code('"parameters"')

            case State.PARAMETERS_KEY:
                return self._extract_token_code(":")

            case State.PARAMETERS_COLON:
                return self._extract_token_code("{")

            case State.OPEN_PARAMETERS_OBJECT:
                return []

            case State.PARAMETER_KEY:
                return self._extract_token_code(f'"{current_parameter}"')

            case State.PARAMETER_COLON:
                # Parameter values are handled dynamically
                return self._extract_token_code(":")

            case State.PARAMETER_VALUE:
                return []

            case State.PARAMETER_COMMA:
                return self._extract_token_code(",")

            case State.CLOSE_PARAMETERS_OBJECT:
                return self._extract_token_code("}")

            case State.CLOSE_ROOT_OBJECT:
                return self._extract_token_code("}")

            case State.DONE:
                return []

        return []

    def _dynamic_allowed_tokens(self) -> list[int]:
        parameter_type = self.state_machine.current_parameter_type()

        if parameter_type == "boolean":
            true_tokens: list[int] = self._extract_token_code("true")
            false_tokens: list[int] = self._extract_token_code("false")

            return true_tokens + false_tokens

        if parameter_type == "string":
            quote = self._extract_single_token('"')

            if not self.string_started:
                return [quote]

            # Guarantee termination to prevent an infinite loop
            if len(self.generated_value_tokens) >= self.max_string_tokens:
                return [quote]

            # build once, cache it, don't rebuild every step
            if self._string_content is None:
                banned_characters = set('"{}:,')
                self._string_content = [
                    token_id
                    for token_id, token_text in self.vocabulary.all_tokens().items()
                    if not any(char in banned_characters for char in token_text)
                ]
            return self._string_content + [quote]

        if parameter_type == "number":
            return self._number_allowed_tokens()

        raise ValueError(
            f"Unsupported parameter type: {parameter_type}"
        )

    def _number_allowed_tokens(
        self
    ) -> list[int]:

        tokens = set()

        for digit in "0123456789":
            tokens.update(self._extract_token_code(digit))

        if not self.generated_value_tokens:
            tokens.update(self._extract_token_code("-"))
        else:
            tokens.update(self._extract_token_code("."))
            terminator = "," if self.state_machine.has_more_parameters() else "}"
            tokens.update(self._extract_token_code(terminator))

        return list(tokens)

    def _extract_token_code(self, text: str) -> list[int]:
        # Converts grammar symbol to token ids
        tokens = self.llm.encode(text).squeeze().tolist()
        if isinstance(tokens, int):
            return [tokens]
        return tokens

    def _extract_single_token(self, text: str) -> int:
        tokens = self._extract_token_code(text)
        if len(tokens) < 1:
            raise ValueError(
                f"Expected '{text}' to encode to one token, got {tokens}"
            )
        return tokens[0]

    def _reset_dynamic_state(self):
        self.generated_value_tokens.clear()
        self.string_started = False
        self.string_finished = False
