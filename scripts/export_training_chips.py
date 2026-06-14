from __future__ import annotations

import argparse
import json

from geoai_roads.training_data import export_training_chips, load_training_data_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export image/mask training chips from imagery and QGIS-edited labels."
    )
    parser.add_argument(
        "--config",
        default="config/training.whu-taos.example.yaml",
        help="Training data export config path.",
    )
    args = parser.parse_args()

    config = load_training_data_config(args.config)
    summary = export_training_chips(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
