import sys
import argparse
import json
from pathlib import Path
from .models.function_call import FunctionCall


def render_progress(current_index: int, total: int, width: int = 30) -> None:
    filled = int(width * current_index / total)
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"{bar} {current_index}/{total}")


def stream_token(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json")
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json")
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json")
    )

    return p.parse_args()


def write_output(results: list[FunctionCall], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([call.model_dump() for call in results], f, indent=4)
