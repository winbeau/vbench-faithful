"""Offline schema examples; deliberately not a training command."""

import argparse
import json

EXAMPLES = {
    "spatial": {"relationships": [{"subject": "cat", "relation": "left", "object": "dog"}]},
    "action": {"actions": ["playing guitar"]},
    "objects": {"entities": ["cat", "dog", "bird"]},
    "scene": "supported",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect minimal task outputs (no models loaded).")
    parser.add_argument("--task", choices=EXAMPLES, required=True)
    args = parser.parse_args()
    result = EXAMPLES[args.task]
    print(result if isinstance(result, str) else json.dumps(result))


if __name__ == "__main__":
    main()
