"""Remove only known legacy exports after their consolidated replacements exist."""
from pathlib import Path
import re
import stat


def cleanup_legacy_outputs(output_dir, experiments):
    root = Path(output_dir).resolve()
    names = {"permutation_results.csv", "transition_summary.csv", "feature_types.csv",
             "original_distribution_comparison.png", "environment.txt",
             "existing_regression.log", "regression_temporal_encoder.pt"}
    for name in experiments:
        if not isinstance(name, str) or not re.fullmatch(r"(?:[ABC]|D_\d+|[BC]_reduced_dependency|C_weight_\d+(?:\.\d+)?(?:e[+-]?\d+)?)", name):
            raise ValueError("Legacy cleanup names must stay inside the output allowlist")
        names.update({f"k_selection_{name}.csv", f"k_selection_{name}.png", f"churn_common_k_{name}.csv"})
    for pair in ("A_vs_B", "A_vs_C", "B_vs_C"):
        names.update({f"contingency_{pair}.csv", f"transitions_{pair}.csv", f"transition_{pair}.png"})
    targets = [root / name for name in names]
    # Do not follow symbolic links or Windows reparse points to their targets.
    targets = [path for path in targets if path.is_file() and not path.is_symlink()
               and not getattr(path.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)]
    # Validate every absolute target before deleting anything; never recurse.
    if any(path.resolve().parent != root for path in targets):
        raise ValueError("Legacy cleanup targets must stay directly inside the output directory")
    removed = []
    for path in targets:
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return sorted(removed)
