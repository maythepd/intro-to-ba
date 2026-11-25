"""Verify original LR/RF holdout predictions before publishing challenger manifests."""
import argparse
import json
from api import module5
from api.challenger_models import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("lr", "rf"), default=["lr", "rf"])
    parser.add_argument("--verify-study", action="store_true", help="Always enforced; retained for explicit operator commands")
    args = parser.parse_args()
    model = module5.load()
    if model is None:
        parser.error("Prepare the fixed Module 5 artifacts first")
    result = prepare(model, args.models)
    print(json.dumps([{key: item[key] for key in ("algorithm", "model_id", "study_reproduced", "holdout_metrics", "comparisons")}
                      for item in result], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
