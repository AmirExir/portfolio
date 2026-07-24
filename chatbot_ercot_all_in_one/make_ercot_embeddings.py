"""Compatibility entrypoint for the guarded legacy ERCOT cache generator.

The historical implementation silently overwrote 3072-dimensional OpenAI
vectors with incompatible 384-dimensional local vectors.  Delegate to the
validated generator instead; it is read-only unless ``--force`` is supplied.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    runpy.run_path(
        str(Path(__file__).with_name("generate_ercot_all_embeddings.py")),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
