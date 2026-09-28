"""Compatibility entry point for the shared Streamlit fault-classifier app."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fault_classifier.fault_classifier_app import main


if __name__ == "__main__":
    main()
