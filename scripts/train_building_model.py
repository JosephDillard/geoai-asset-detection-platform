from __future__ import annotations

import argparse
import json

from geoai_roads.training import load_building_training_config, train_building_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune the WHU building segmentation model from exported chips."
    )
    parser.add_argument(
        "--config",
        default="config/training.whu-taos.example.yaml",
        help="Building training config path.",
    )
    args = parser.parse_args()

    config = load_building_training_config(args.config)
    result = train_building_model(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
