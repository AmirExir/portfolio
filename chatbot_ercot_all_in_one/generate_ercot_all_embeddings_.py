"""Legacy filename for the guarded ERCOT cache generator.

Importing this compatibility module is inert.  Running it delegates to the
maintained generator, whose paid rebuild path requires ``--force``.
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
