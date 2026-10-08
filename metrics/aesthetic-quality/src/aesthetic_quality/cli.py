"""Aesthetic Quality through the shared YAML evaluation controller."""
from vbench_audit_core.eval_cli import metric_main


def main(argv: list[str] | None = None) -> int:
    return metric_main("aesthetic_quality", argv)


if __name__ == "__main__":
    raise SystemExit(main())
