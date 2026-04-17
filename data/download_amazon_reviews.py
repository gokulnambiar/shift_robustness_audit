from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import prepare_domain_splits


def main() -> None:
    bundle = prepare_domain_splits(data_dir=PROJECT_ROOT / "data")
    summary_path = PROJECT_ROOT / "data" / "prepared_summary.json"
    summary_path.write_text(json.dumps(bundle.dataset_summary, indent=2))
    print(f"Prepared dataset written to {PROJECT_ROOT / 'data' / 'prepared_reviews.csv'}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
