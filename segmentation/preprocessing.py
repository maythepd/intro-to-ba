"""Train-fitted imputation/OHE/scaling with equal expected group distance."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .features import ORIGINAL_FEATURES, SYNTHETIC_FEATURES, CATEGORICAL, LOG_FEATURES


def signed_log1p(values):
    return np.sign(values) * np.log1p(np.abs(values))


def make_group(columns):
    numeric = [c for c in columns if c not in CATEGORICAL]
    logs = [c for c in numeric if c in LOG_FEATURES]
    linear = [c for c in numeric if c not in LOG_FEATURES]
    blocks = []
    for name, cols in (("log", logs), ("linear", linear)):
        if cols:
            steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True))]
            if name == "log":
                steps.append(("log1p", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")))
            steps.append(("scale", StandardScaler()))
            blocks.append((name, Pipeline(steps), cols))
    categories = [c for c in columns if c in CATEGORICAL]
    if categories:
        blocks.append(("category", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categories))
    return Pipeline([("columns", ColumnTransformer(blocks, remainder="drop")),
                     ("varying", VarianceThreshold())])


class GroupPreprocessor(TransformerMixin, BaseEstimator):
    """Multiply each group by sqrt(weight / sum(training column variances)).

    This controls expected squared Euclidean contribution, including OHE width
    and missing indicators, without using held-out values or customer labels.
    """

    def __init__(self, original_weight=.5):
        self.original_weight = original_weight

    def fit(self, X, y=None):
        if y is not None:
            raise ValueError("Segmentation preprocessing does not accept targets")
        if not 0 < self.original_weight < 1:
            raise ValueError("original_weight must be between 0 and 1")
        approved = set(ORIGINAL_FEATURES + SYNTHETIC_FEATURES)
        if set(X) - approved:
            raise ValueError(f"Unapproved clustering inputs: {sorted(set(X) - approved)}")
        self.columns_ = list(X.columns)
        self.groups_, self.scales_, self.slices_, self.feature_names_ = {}, {}, {}, []
        groups = {"original": [c for c in ORIGINAL_FEATURES if c in X],
                  "synthetic": [c for c in SYNTHETIC_FEATURES if c in X]}
        groups = {name: cols for name, cols in groups.items() if cols}
        offset = 0
        for name, cols in groups.items():
            prep = make_group(cols)
            values = prep.fit_transform(X[cols])
            energy = values.var(axis=0).sum()
            if energy <= 0:
                raise ValueError(f"{name} features have no training variation")
            weight = 1. if len(groups) == 1 else (self.original_weight if name == "original" else 1 - self.original_weight)
            self.groups_[name] = prep
            self.scales_[name] = np.sqrt(weight / energy)
            self.slices_[name] = slice(offset, offset + values.shape[1])
            names = prep.named_steps["columns"].get_feature_names_out()
            names = names[prep.named_steps["varying"].get_support()]
            self.feature_names_.extend(f"{name}__{col}" for col in names)
            offset += values.shape[1]
        if not groups:
            raise ValueError("No approved features")
        self.n_samples_fit_ = len(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "groups_")
        if set(X) != set(self.columns_):
            raise ValueError("Prediction columns must match fitted features")
        values = [prep.transform(X) * self.scales_[name] for name, prep in self.groups_.items()]
        return np.hstack(values)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)
