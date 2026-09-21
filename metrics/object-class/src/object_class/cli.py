from __future__ import annotations

import argparse

from vbench_audit_core.cli import build_parser as build_core_parser
from vbench_audit_core.contracts import execute_metric

from .metric import evaluate_batch, summarize

METRIC = "object-class"


def build_parser() -> argparse.ArgumentParser:
    parser = build_core_parser(METRIC, "VBench object class evaluator")
    parser.add_argument(
        "--audit-variant",
        choices=("diagnostic", "legacy", "repair"),
        default="repair",
        help="repair: deterministic aliases; legacy: exact strings on retained evidence",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return execute_metric(METRIC, args, evaluate_batch, description=build_parser().description, summarize=summarize)


if __name__ == "__main__":
    raise SystemExit(main())
