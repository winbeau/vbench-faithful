from vbench_audit_core.cli import build_parser, execute


def main() -> int:
    parser = build_parser("motion-smoothness", "VBench motion smoothness evaluator")
    return execute("motion-smoothness", parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
