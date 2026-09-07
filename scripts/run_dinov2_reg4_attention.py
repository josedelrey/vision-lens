from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vision_lens.cli import main as cli_main  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "vit_attention.dinov2_reg4.example.yaml"


def main(argv: list[str] | None = None) -> int:
    forwarded = sys.argv[1:] if argv is None else argv
    return cli_main(["--config", str(CONFIG), *forwarded])


if __name__ == "__main__":
    raise SystemExit(main())
