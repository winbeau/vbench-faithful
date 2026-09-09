from vbench_audit_core.cli import build_parser, execute
from .metric import official_backend


def main() -> int:
    parser = build_parser("dynamic-degree", "VBench dynamic degree evaluator")
    return execute("dynamic-degree", parser.parse_args(), official_backend)


if __name__ == "__main__":
    raise SystemExit(main())
