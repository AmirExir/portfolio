"""Compatibility entry point for the shared fault-classifier training CLI."""

if __package__:
    from .fault_classification_v2 import main
else:
    from fault_classification_v2 import main


if __name__ == "__main__":
    raise SystemExit(main())
