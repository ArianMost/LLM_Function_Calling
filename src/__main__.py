from .decoder.constrained_decoder import ConstrainedDecoder
from .decoder.function_selector import FunctionSelector
from .models.function_call import FunctionCall
from .parser.output_parser import parse_output
from .state_machine.token_filter import TokenFilterEngine
from llm_sdk import Small_LLM_Model
from .parser.input_parser import parse_function_definitions, parse_prompts
from .state_machine.json_state_machine import JSONStateMachine
from .utils import parse_args, render_progress, stream_token, write_output



def main() -> None:
    try:
        args = parse_args()
        llm = Small_LLM_Model()
        functions = parse_function_definitions(args.functions_definition)
        prompts_list = parse_prompts(args.input)
        function_selector = FunctionSelector(llm, functions)
        output: list[FunctionCall] = []

        for i, prompt in enumerate(prompts_list, start=1):
            render_progress(i, len(prompts_list))
            print(f"\n{prompt.prompt}")
            function_definition = function_selector.select(prompt.prompt)
            machine = JSONStateMachine(function_definition)
            engine = TokenFilterEngine(machine, llm)
            decoder = ConstrainedDecoder(llm, engine)
            result = decoder.decode(prompt.prompt, on_token=stream_token)
            function_call = parse_output(prompt.prompt, function_definition, result)
            print()
            output.append(function_call)

        write_output(output, args.output)
        print(f"\nwrote {len(output)}/{len(prompts_list)} results to {args.output}")
    except Exception as e:
        print(f"\n  ! Skipped: {e}")


if __name__ == "__main__":
    main()
