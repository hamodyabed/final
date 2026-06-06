#!/usr/bin/env python3
"""Generate (or regenerate) the blank coding workbook from extracted_keys.json.

Usage:
    python generate_excel.py [output_path]
"""

import sys
from pathlib import Path

from app.excel_export import create_workbook


def main() -> int:
    project_root = Path(__file__).resolve().parent
    json_path = project_root / "data" / "extracted_keys.json"
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else project_root / "output" / "narrative_coding_template.xlsx"
    )
    written = create_workbook(json_path, out)
    print(f"✓ Workbook written to: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
