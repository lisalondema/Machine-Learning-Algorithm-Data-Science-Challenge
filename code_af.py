#%%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from category_encoders import TargetEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

import warnings
warnings.filterwarnings("ignore")

#%%
##### DATA UNDERSTANDING #####

# Target feature: IN_DEFAULT_FLG (1 = default, 0 = no default)
target_feature = "IN_DEFAULT_FLG"

# Loading the dataset in a pandas DataFrame
training_dataset = pd.read_excel(Path(__file__).resolve().parent / "DSC_training.xlsx", na_values=np.nan)

### DATA DESCRIPTION ###
print(training_dataset.describe())
print("")  # add empty line for readability

#%%
### DATA EXPLORATION AND QUALITY EXAMINATION ###
print("First 10 rows of the dataset:")
print(training_dataset.head(10))
print("")
print("Information about the dataset:")
print(training_dataset.info())
print("")
print("Shape of the dataset (#rows, #columns):")
print(training_dataset.shape)
print("")
default_percentage = training_dataset['IN_DEFAULT_FLG'].mean() * 100
print("Amount of default cases:", default_percentage, "%")
print("")
counts = training_dataset['IN_DEFAULT_FLG'].value_counts()
plt.bar(counts.index.astype(str), counts.values)
plt.xlabel("Default status")
plt.ylabel("Amount of cases")
plt.title("Amount of default and non-default cases")
plt.show()

#%%
##### DATA PREPARATION #####
selected_columns = training_dataset.copy()

# Try to convert possible datetime columns
for col in selected_columns.columns:
    if col == target_feature:
        continue
    if any(word in col.lower() for word in ['date', 'dt', 'timestamp']):
        try:
            selected_columns[col] = pd.to_datetime(
                selected_columns[col], errors='raise', format='%Y-%m-%d %H:%M:%S'
            )
        except (ValueError, TypeError):
            pass

# Identify datetime columns and expand
datetime_cols = selected_columns.select_dtypes(include=['datetime', 'datetime64', 'datetimetz']).columns.tolist()
new_cols = {}
for col in datetime_cols:
    new_cols[col + '_year'] = selected_columns[col].dt.year
    new_cols[col + '_month'] = selected_columns[col].dt.month
    new_cols[col + '_day'] = selected_columns[col].dt.day
if new_cols:
    datetime_features_df = pd.DataFrame(new_cols, index=selected_columns.index)
    selected_columns = pd.concat([selected_columns.drop(columns=datetime_cols), datetime_features_df], axis=1)

print("After datetime expansion:")
print(selected_columns.head())
print("")

#%%
##### DATA SPLITTING #####
n = len(selected_columns)
indices_all = np.arange(n)

indices_train, indices_test = train_test_split(
    indices_all,
    test_size=0.2,
    random_state=0,
    stratify=selected_columns[target_feature].iloc[indices_all]
)

indices_train, indices_val = train_test_split(
    indices_train,
    test_size=0.2,
    random_state=0,
    stratify=selected_columns[target_feature].iloc[indices_train]
)

def drop_columns(training_df, indices=None):
    if indices is not None:
        df_for_stats = training_df.iloc[indices]
        context = "TRAINING SUBSET"
    else:
        df_for_stats = training_df
        context = "FULL DATA"
    missing_percentages = df_for_stats.isna().mean() * 100
    print(f"Missing % (computed on {context}):")
    print(missing_percentages.sort_values(ascending=False).to_string())
    zero_percentages = (df_for_stats == 0).mean() * 100
    print(f"\nZero % (computed on {context}):")
    print(zero_percentages.sort_values(ascending=False).to_string())
    print("")
    
    missing_cols = []
    for col in training_df.columns:
        if missing_percentages.get(col, 0.0) > 90 and col != target_feature:
            missing_cols.append(col)
    
    zero_cols = []
    for col in training_df.columns:
        if zero_percentages.get(col, 0.0) > 95 and col != target_feature:
            zero_cols.append(col)
    cols_to_drop = list(set(missing_cols + zero_cols))
    print(f"Columns dropped due to missing (>90%) or zeros (>95%) computed on {context}: {cols_to_drop}")
    print("")
    return training_df.drop(columns=cols_to_drop), cols_to_drop

selected_columns, dropped_on_train = drop_columns(selected_columns, indices=indices_train)
print(selected_columns.head())
print("")

X = selected_columns.drop(target_feature, axis=1)
y = selected_columns[target_feature]

X_train = X.iloc[indices_train].copy()
X_val = X.iloc[indices_val].copy()
X_test = X.iloc[indices_test].copy()
y_train = y.iloc[indices_train].copy()
y_val = y.iloc[indices_val].copy()
y_test = y.iloc[indices_test].copy()

print("Raw split sizes (train/val/test):", X_train.shape, X_val.shape, X_test.shape)
print("")

#%%
##### IDENTIFY COLUMN TYPES #####
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

high_card_cols = []
low_card_cols = []
for col in cat_cols:
    number_unique_values = X_train[col].nunique(dropna=True)
    if number_unique_values > 10:
        high_card_cols.append(col)
    else:
        low_card_cols.append(col)

print("Numerical cols count:", len(num_cols))
print("Low-card categorical cols:", low_card_cols)
print("High-card categorical cols:", high_card_cols)
print("")

feature_names = list(X.columns)

#%%
##### HELPER CLASSES FOR PREPROCESSING #####

# TargetEncoderWrapper and Clipper: custom transformers for pipeline
# class for target encoding using category_encoders.TargetEncoder
class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, min_samples_leaf=5, smoothing=2.0):
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self.cols = cols
        self.encoder_ = None

# function that fits the TargetEncoder on specified columns
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("TargetEncoderWrapper requires y for fit.")
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            if self.cols is None:
                cols = [f"col_{i}" for i in range(X.shape[1])]
            else:
                cols = list(self.cols)
            if isinstance(y, (pd.Series, pd.DataFrame)):
                X_df = pd.DataFrame(X, columns=cols, index=y.index)
            else:
                X_df = pd.DataFrame(X, columns=cols)
        cols_for_encoder = list(X_df.columns)
        self.encoder_ = TargetEncoder(cols=cols_for_encoder, min_samples_leaf=self.min_samples_leaf, smoothing=self.smoothing)
        self.encoder_.fit(X_df, y)
        return self

# function that transforms the data using the fitted TargetEncoder
    def transform(self, X):
        if self.encoder_ is None:
            raise RuntimeError("TargetEncoderWrapper not fitted")
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            if self.cols is None:
                cols = [f"col_{i}" for i in range(X.shape[1])]
            else:
                cols = list(self.cols)
            X_df = pd.DataFrame(X, columns=cols)
        return self.encoder_.transform(X_df)

# class that clips numerical values to a specified range
class Clipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower=-3.0, upper=3.0):
        self.lower = lower
        self.upper = upper
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_arr = np.array(X, copy=True)
        np.clip(X_arr, self.lower, self.upper, out=X_arr)
        return X_arr

#%%
##### BUILD PREPROCESSOR #####
# Define transformers for different column types
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('clipper', Clipper(-3.0, 3.0))
])

# Low-card categorical transformer: impute missing with most frequent, then one-hot encode
low_card_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
])

# High-card categorical transformer: impute missing with most frequent, then target encode
high_card_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('target_enc', TargetEncoderWrapper(cols=high_card_cols, min_samples_leaf=5, smoothing=2.0))
])

# Combine transformers into a ColumnTransformer
transformers = []
if num_cols:
    transformers.append(('num', numeric_transformer, num_cols))
if low_card_cols:
    transformers.append(('low_cat', low_card_transformer, low_card_cols))
if high_card_cols:
    transformers.append(('high_cat', high_card_transformer, high_card_cols))

preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0)

# Fit the preprocessor on X_train to inspect fitted imputers and avoid NaN statistics
try:
    preprocessor.fit(X_train, y_train)
    print("Preprocessor successfully fitted for diagnostics.")

    # replace any NaN statistics_ with 0.0
    try:
        num_tr = preprocessor.named_transformers_.get('num')
        if num_tr is not None:
            num_imputer = num_tr.named_steps.get('imputer', None)
            if num_imputer is not None and hasattr(num_imputer, "statistics_"):
                num_stats = np.asarray(num_imputer.statistics_, dtype=float)
                nan_idx = np.where(np.isnan(num_stats))[0]
                if nan_idx.size > 0:
                    print("Numeric imputer has NaN statistics for columns:", [num_cols[i] for i in nan_idx])
                    num_stats[nan_idx] = 0.0
                    num_imputer.statistics_ = num_stats.tolist()
                    print("Replaced NaN numeric-imputer statistics with 0.0")
    except Exception as _e:
        print("Warning while checking numeric imputer:", _e)

    # replace any NaN statistics_ with empty string
    try:
        low_tr = preprocessor.named_transformers_.get('low_cat')
        if low_tr is not None:
            low_imputer = low_tr.named_steps.get('imputer', None)
            if low_imputer is not None and hasattr(low_imputer, "statistics_"):
                low_stats = list(low_imputer.statistics_)
                bad_idxs = [i for i, v in enumerate(low_stats) if pd.isna(v)]
                if bad_idxs:
                    print("Low-card categorical imputer has NaN statistics for columns:", [low_card_cols[i] for i in bad_idxs])
                    for i in bad_idxs:
                        low_stats[i] = ""
                    low_imputer.statistics_ = low_stats
                    print("Replaced NaN categorical-imputer statistics with empty string for those columns")
    except Exception as _e:
        print("Warning while checking low-card categorical imputer:", _e)

    # 3) High-card categorical imputer (if present): similar to low-card
    try:
        high_tr = preprocessor.named_transformers_.get('high_cat')
        if high_tr is not None:
            high_imputer = high_tr.named_steps.get('imputer', None)
            if high_imputer is not None and hasattr(high_imputer, "statistics_"):
                high_stats = list(high_imputer.statistics_)
                bad_idxs = [i for i, v in enumerate(high_stats) if pd.isna(v)]
                if bad_idxs:
                    print("High-card categorical imputer has NaN statistics for columns:", [high_card_cols[i] for i in bad_idxs])
                    for i in bad_idxs:
                        high_stats[i] = ""
                    high_imputer.statistics_ = high_stats
                    print("Replaced NaN high-card categorical-imputer statistics with empty string for those columns")
    except Exception as _e:
        print("Warning while checking high-card categorical imputer:", _e)

except Exception as e:
    print("Could not fit preprocessor for diagnostics (continuing). Error:", e)


#%%
##### MODELING PHASE #####
MODELS = {
    "logistic": LogisticRegression,
    "rf": RandomForestClassifier,
    "dt": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
    "svm": SVC,
    "xgb": XGBClassifier
}

HYPERPARAM = {
    "logistic": {"C": [0.001, 0.01, 0.1, 1.0]},
    "rf": {"n_estimators": [100, 200, 500], "min_samples_leaf": [1, 5], "max_depth": [None, 70], "random_state": [0]},
    "dt": {"criterion": ["gini", "entropy"], "max_depth": [None, 70], "min_samples_leaf": [2, 5, 10], "min_samples_split": [2]},
    "knn": {"n_neighbors": [5, 10, 20, 50]},
    "svm": {"C": [0.1, 1.0], "kernel": ["linear", "rbf"], "probability": [True]},
    "xgb": {"n_estimators": [100, 200], "max_depth": [3, 6, 10], "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0], "reg_alpha": [0, 1], "reg_lambda": [1, 2], "random_state": [0]}
}

model_inits = {
    "logistic": LogisticRegression(max_iter=10000, solver="lbfgs"),
    "svm": SVC(probability=True)
}

RANDOM_SEED = 0
fitted_models = {}
model_results = {}

def _get_proba_or_scores(model, X_):
    try:
        proba = model.predict_proba(X_)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            out = proba[:, 1].astype(float)
        else:
            out = proba.ravel().astype(float)
        return np.asarray(out)
    except Exception:
        pass
    try:
        dec = model.decision_function(X_)
        return np.asarray(dec).ravel().astype(float)
    except Exception:
        pass
    try:
        preds = model.predict(X_)
        preds_arr = np.asarray(preds).ravel().astype(float)
        warnings.warn(UserWarning)
        return preds_arr
    except Exception as e:
        raise RuntimeError(f"Kon geen scores/proba/predict op model uitvoeren: {e}")

# helper to check if a pipeline's preprocessor is fitted
def is_pipeline_fitted(p):
    try:
        return (hasattr(p, "named_steps") and
                "preproc" in p.named_steps and
                hasattr(p.named_steps["preproc"], "named_transformers_"))
    except Exception:
        return False

for model_name, ModelClass in MODELS.items():
    print(f"Fitting model: {model_name}")
    param_grid = HYPERPARAM.get(model_name, {})

    base_model = model_inits.get(model_name, None)
    if base_model is None:
        base_model = ModelClass()

    candidate_pipeline = Pipeline(steps=[
        ('preproc', preprocessor),
        ('model', base_model)
    ])

    param_grid_prefixed = {f"model__{k}": v for k, v in param_grid.items()}

    gs = GridSearchCV(
        estimator=candidate_pipeline,
        param_grid=param_grid_prefixed,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
        refit=True,
        error_score=np.nan
    )

    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    fitted_models[model_name] = best

    try:
        y_test_score = _get_proba_or_scores(best, X_test)
    except Exception:
        y_test_score = best.predict(X_test)

    try:
        test_auc = float(roc_auc_score(y_test, y_test_score))
    except Exception:
        test_auc = float("nan")
    try:
        test_acc = float(accuracy_score(y_test, best.predict(X_test)))
    except Exception:
        test_acc = float("nan")

    model_results[model_name] = {
        "best_params": gs.best_params_,
        "cv_best_score": float(gs.best_score_) if not np.isnan(gs.best_score_) else float("nan"),
        "test_auc": test_auc,
        "test_acc": test_acc
    }

    print(f"Best model (pipeline): {best}")
    print(f"CV best AUC: {gs.best_score_}")
    print(f"Test AUC: {test_auc:.4f}, Test ACC: {test_acc:.4f}\n")

#%%
##### EVALUATION PHASE #####
def _get_score(res, key):
    try:
        v = res.get(key)
        v = float(v)
        return None if math.isnan(v) else v
    except Exception:
        return None

def choose_best_model_name(fitted_models, model_results):
    if model_results:
        best = None
        best_score = -math.inf
        for name, res in model_results.items():
            s = _get_score(res, "test_auc")
            if s is not None and s > best_score:
                best_score, best = s, name
        if best:
            return best
        for name, res in model_results.items():
            s = _get_score(res, "cv_best_score")
            if s is not None and s > best_score:
                best_score, best = s, name
        if best:
            return best
    if fitted_models:
        return next(iter(fitted_models.keys()))
    raise RuntimeError("no models found")

def select_best(fitted_models, model_results, env=None):
    env = env or globals()
    best_name = choose_best_model_name(fitted_models, model_results)
    best_model = fitted_models[best_name]
    artifact_names = ["preprocessor", "fill_values", "low_card_cols", "high_card_cols", "feature_names", "num_columns"]
    artifacts = {n: (env[n].tolist() if hasattr(env.get(n), "tolist") else env[n])
                 for n in artifact_names if n in env}
    return {"best_name": best_name, "best_model": best_model, "artifacts": artifacts, "model_results": model_results}

def print_results(out):
    print("Best model:", out["best_name"])
    print("Model type:", type(out["best_model"]).__name__)
    if out["artifacts"]:
        print("Artifacts:", ", ".join(sorted(out["artifacts"].keys())))
    else:
        print("Artifacts: (none)")
    if out["model_results"]:
        for name, res in out["model_results"].items():
            print(f"{name}: test_auc={res.get('test_auc')!r}, cv_best_score={res.get('cv_best_score')!r}")
    else:
        print("model_results: (none)")

#%%
###### PRINTING AND PLOTTING RESULTS #####
print("\nModel results:")
for name, res in model_results.items():
    print(f"- {name:8s}: CV AUC={res.get('cv_best_score')!r}, Test AUC={res.get('test_auc')!r}, Test ACC={res.get('test_acc')!r}")

summary = pd.DataFrame([
    {"model": k, "cv_auc": v.get("cv_best_score"), "test_auc": v.get("test_auc")}
    for k, v in model_results.items()
]).set_index("model").sort_values("test_auc", ascending=False)

ax = summary.plot.bar(rot=0, figsize=(8,4), color=["#4C72B0", "#55A868"])
ax.set_ylim(0, 1)
ax.set_ylabel("AUC")
for p in ax.patches:
    h = p.get_height()
    if not np.isnan(h):
        ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width()/2, h), ha="center", va="bottom", fontsize=8)
plt.title("CV best AUC vs Test AUC per model")
plt.tight_layout()
plt.show()

if "logistic" in fitted_models:
    try:
        pipe = fitted_models["logistic"]
        scores = _get_proba_or_scores(pipe, X_test)
        auc = roc_auc_score(y_test, scores)
        fpr, tpr, _ = roc_curve(y_test, scores)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f"Logistic AUC = {auc:.4f}", lw=2)
        plt.plot([0,1], [0,1], "k--", lw=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC (Logistic) on test set")
        plt.legend(loc="lower right"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.show()
    except Exception as e:
        print("could not plot logistic ROC:", e)

#%%
##### PHASE 5: select, refit, and scoring #####
# Compute validation AUC for each fitted model and store it
for name, pipeline in fitted_models.items():
    try:
        val_scores = _get_proba_or_scores(pipeline, X_val)
        val_auc = float(roc_auc_score(y_val, val_scores))
    except Exception:
        val_auc = float("nan")
    if name not in model_results:
        model_results[name] = {}
    model_results[name]["val_auc"] = val_auc

def _get_sort_key_for_selection(name):
    try:
        v = model_results[name].get("val_auc")
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return (2, float(v))
    except Exception:
        pass
    try:
        v = model_results[name].get("cv_best_score")
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return (1, float(v))
    except Exception:
        pass
    try:
        v = model_results[name].get("test_auc")
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return (0, float(v))
    except Exception:
        pass
    return (-1, -math.inf)

best_model_name = max(fitted_models.keys(), key=_get_sort_key_for_selection)
best_model = fitted_models[best_model_name]
best_model_info = model_results.get(best_model_name, {})

print(f"\nChosen model: {best_model_name}")
print(" - val_auc:", best_model_info.get("val_auc"))
print(" - cv_best_score:", best_model_info.get("cv_best_score"))
print(" - earlier test_auc:", best_model_info.get("test_auc"))
print(" - best_params:", best_model_info.get("best_params"))

X_trainval = pd.concat([X_train, X_val], axis=0).sort_index()
y_trainval = pd.concat([y_train, y_val], axis=0).loc[X_trainval.index]

pipeline_trainval = clone(best_model)
pipeline_trainval.fit(X_trainval, y_trainval)

try:
    y_test_scores = _get_proba_or_scores(pipeline_trainval, X_test)
    honest_test_auc = roc_auc_score(y_test, y_test_scores)
except Exception:
    honest_test_auc = float("nan")
try:
    honest_test_acc = accuracy_score(y_test, pipeline_trainval.predict(X_test))
except Exception:
    honest_test_acc = float("nan")

print("\nHonest evaluation (refit on train+val, eval on test):")
print(f" Test AUC:  {honest_test_auc:.4f}")
print(f" Test ACC:  {honest_test_acc:.4f}")

X_all = pd.concat([X_train, X_val, X_test], axis=0).sort_index()
y_all = y.loc[X_all.index]

pipeline_all = clone(best_model)
pipeline_all.fit(X_all, y_all)
print("\nRefitted chosen pipeline on ALL labeled data (pipeline_all ready).")

try:
    base_path = Path(__file__).resolve().parent
except NameError:
    base_path = Path.cwd()

scoring_path = base_path / "DSC_scoring.xlsx"
df_scoring_raw = pd.read_excel(scoring_path, na_values=np.nan)
print(f"Loaded scoring dataset: {df_scoring_raw.shape[0]} rows, {df_scoring_raw.shape[1]} columns")

df_scoring = df_scoring_raw.copy()
for col in list(df_scoring.columns):
    if any(word in col.lower() for word in ["date", "dt", "timestamp"]):
        try:
            df_scoring[col] = pd.to_datetime(df_scoring[col], errors="raise", format="%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            df_scoring[col] = pd.to_datetime(df_scoring[col], errors="coerce")

datetime_cols_scoring = df_scoring.select_dtypes(include=["datetime", "datetime64", "datetimetz"]).columns.tolist()
new_cols_scoring = {}
for col in datetime_cols_scoring:
    new_cols_scoring[col + "_year"] = df_scoring[col].dt.year
    new_cols_scoring[col + "_month"] = df_scoring[col].dt.month
    new_cols_scoring[col + "_day"] = df_scoring[col].dt.day
if new_cols_scoring:
    datetime_features_df_scoring = pd.DataFrame(new_cols_scoring, index=df_scoring.index)
    df_scoring = pd.concat([df_scoring.drop(columns=datetime_cols_scoring), datetime_features_df_scoring], axis=1)

feature_cols_raw = [c for c in selected_columns.columns if c != target_feature]
df_scoring = df_scoring.reindex(columns=feature_cols_raw)
print("Scoring raw shape after reindex to training features:", df_scoring.shape)
print("Total missing cells (before imputation):", df_scoring.isna().sum().sum())

for c in num_cols:
    if c in df_scoring.columns:
        df_scoring[c] = pd.to_numeric(df_scoring[c], errors="coerce")

pipeline = pipeline_all
print("\nAttempting pipeline_all.predict_proba on scoring data (pipeline imputers will handle missing).")
try:
    scoring_proba = _get_proba_or_scores(pipeline, df_scoring)
    scoring_proba = np.asarray(scoring_proba).ravel()
    if np.isnan(scoring_proba).any():
        raise ValueError("predict_proba returned NaNs")
    print("pipeline_all.predict_proba succeeded without NaNs.")
except Exception as e:
    print("Direct pipeline.predict_proba failed or returned NaNs:", e)
    print("Falling back to manual fill from X_train then retrying.")

    fill_values_computed = {}
    for col in feature_cols_raw:
        if col in X_train.columns:
            if X_train[col].dtype.kind in 'iufc':
                fill_values_computed[col] = float(X_train[col].mean())
            else:
                modes = X_train[col].mode(dropna=True)
                fill_values_computed[col] = modes.iloc[0] if not modes.empty else ""
        else:
            fill_values_computed[col] = 0 if col in num_cols else ""

    df_scoring_filled = df_scoring.fillna(fill_values_computed)
    remaining_nans = df_scoring_filled.isna().sum().sum()
    print("Remaining NaNs after manual fill:", remaining_nans)

    try:
        if hasattr(pipeline, "named_steps") and "preproc" in pipeline.named_steps:
            if not is_pipeline_fitted(pipeline):
                raise RuntimeError("Pipeline preprocessor is not fitted. Expected a fitted pipeline (pipeline_all).")
            X_prepared = pipeline.named_steps["preproc"].transform(df_scoring_filled)
            model_step = pipeline.named_steps.get("model", pipeline)
            scoring_proba = _get_proba_or_scores(model_step, X_prepared)
        else:
            scoring_proba = _get_proba_or_scores(pipeline, df_scoring_filled)
        scoring_proba = np.asarray(scoring_proba).ravel()
        if np.isnan(scoring_proba).any():
            raise ValueError("scoring_proba contains NaNs after fallback")
        print("Fallback scoring succeeded.")
    except Exception as e2:
        print("Fallback scoring failed:", e2)
        raise RuntimeError("Scoring failed even after manual fill. Inspect df_scoring and pipeline preprocessing.") from e2

id_col_candidates = ["loan_id", "LoanID", "Loan_Id", "ID", "SampleID", "sample_id"]
id_col = next((c for c in id_col_candidates if c in df_scoring_raw.columns), None)
if id_col:
    ids = df_scoring_raw[id_col].values
else:
    ids = df_scoring_raw.index.values

predictions_df = pd.DataFrame({"SampleID": ids, "Predicted_Score": scoring_proba})
output_file = base_path / "outputZONDERSMOTE.xlsx"
predictions_df.to_excel(output_file, index=False)
print(f"Predictions saved to '{output_file}'")

try:
    joblib.dump(pipeline_all, base_path / "pipeline_all.joblib")
    print("Saved pipeline_all to disk.")
except Exception:
    pass