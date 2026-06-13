from models.functions_definition import FunctionDefinition
from state_machine.states import State


class JSONStateMachine:
    def __init__(self, function_definition: FunctionDefinition):
        self.current_state = State.START
        self.function_definition = function_definition
        self.parameter_names = list(function_definition.parameters.keys())
        self.current_parameter_index = 0

    def move_forward(self):
        match self.current_state:

            case State.START:
                self.current_state = State.OPEN_ROOT_OBJECT

            case State.OPEN_ROOT_OBJECT:
                self.current_state = State.NAME_KEY

            case State.NAME_KEY:
                self.current_state = State.NAME_COLON

            case State.NAME_COLON:
                self.current_state = State.FUNCTION_NAME

            case State.FUNCTION_NAME:
                self.current_state = State.COMMA_AFTER_NAME

            case State.COMMA_AFTER_NAME:
                self.current_state = State.PARAMETERS_KEY

            case State.PARAMETERS_KEY:
                self.current_state = State.PARAMETERS_COLON

            case State.PARAMETERS_COLON:
                self.current_state = State.OPEN_PARAMETERS_OBJECT

            case State.OPEN_PARAMETERS_OBJECT:
                self.current_state = State.PARAMETER_KEY

            case State.PARAMETER_KEY:
                self.current_state = State.PARAMETER_COLON

            case State.PARAMETER_COLON:
                self.current_state = State.PARAMETER_VALUE

            case State.PARAMETER_VALUE:

                if self.has_more_parameters():
                    self.current_state = State.PARAMETER_COMMA
                else:
                    self.current_state = State.CLOSE_PARAMETERS_OBJECT

            case State.PARAMETER_COMMA:
                self.move_to_next_parameter()
                self.current_state = State.PARAMETER_KEY

            case State.CLOSE_PARAMETERS_OBJECT:
                self.current_state = State.CLOSE_ROOT_OBJECT

            case State.CLOSE_ROOT_OBJECT:
                self.current_state = State.DONE

            case State.DONE:
                pass

    def current_parameter(self) -> str:
        return self.parameter_names[
            self.current_parameter_index
        ]

    def current_parameter_type(self) -> str:
        parameter_name = self.current_parameter()

        return self.function_definition.parameters[
            parameter_name
        ].type

    def has_more_parameters(self) -> bool:
        return (
            self.current_parameter_index
            < len(self.parameter_names) - 1
        )

    def move_to_next_parameter(self):
        if self.has_more_parameters():
            self.current_parameter_index += 1

    def is_done(self) -> bool:
        return self.current_state == State.DONE

