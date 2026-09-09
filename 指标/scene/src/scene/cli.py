from vbench_audit_core.cli import build_parser, execute


def main() -> int:
    parser = build_parser("scene", "VBench scene evaluator")
    return execute("scene", parser.parse_args(), required_metadata=True)


if __name__ == "__main__":
    raise SystemExit(main())
