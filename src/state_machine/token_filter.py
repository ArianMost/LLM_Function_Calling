from llm_sdk import Small_LLM_Model
from state_machine.json_state_machine import JSONStateMachine
from state_machine.states import State


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

    def allowed_tokens(self) -> list[int]:
        if self.state_machine.current_state == State.PARAMETER_COLON:
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

    # def advance_sequence(self) -> None:
    #     self.sequence_position += 1

    #     if self.sequence_position >= len(self.current_sequence):
    #         self.current_sequence = []
    #         self.sequence_position = 0
    #         self.state_machine.move_forward()

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

        if self.state_machine.current_state == State.PARAMETER_COLON:
            self._consume_parameter_value(token)

    def _consume_parameter_value(
        self,
        token: int,
    ) -> None:
        parameter_type = self.state_machine.current_parameter_type()

        if parameter_type == "boolean":
            self.state_machine.move_forward()
            return

        if parameter_type == "number":
            # TODO: Needs more implementation.
            self.state_machine.move_forward()
            return

        if parameter_type == "string":
            # TODO: Needs more implementation.
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
                return self._extract_token_code(f'"{current_parameter}"')

            case State.PARAMETER_KEY:
                return self._extract_token_code(":")

            case State.PARAMETER_COLON:
                # Parameter values are handled dynamically
                return []

            case State.PARAMETER_VALUE:
                if self.state_machine.has_more_parameters():
                    return self._extract_token_code(",")

                return self._extract_token_code("}")

            case State.PARAMETER_COMMA:
                return self._extract_token_code(f'"{current_parameter}"')

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
            # TODO: Needs more implementation.
            return self._extract_token_code('"')

        if parameter_type == "number":
            tokens = []

            for digit in "0123456789":
                tokens.extend(
                    self._extract_token_code(digit)
                )

            return list(set(tokens))

        raise ValueError(
            f"Unsupported parameter type: {parameter_type}"
        )

    def _extract_token_code(self, text: str) -> list[int]:
        # Converts grammar symbol to token ids
        tokens = self.llm.encode(text).squeeze().tolist()
        if isinstance(tokens, int):
            return [tokens]
        return tokens
