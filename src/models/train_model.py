"""Predictive model pipeline - next-day profitability bucket with imbalanced handling."""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier

# Imbalanced handling
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_profitability_model(
    X: pd.DataFrame,
    y: pd.Series,
    imbalance_strategy: str = "smote",
    random_state: int = 42,
    save_path: Optional[Path] = None,
) -> Tuple[Any, Dict[str, float], LabelEncoder]:
    """
    Train BINARY classification model to predict next-day profitability.
    Target: 'profit' (PnL >= 0) vs 'loss' (PnL < 0).
    Binary classification provides clearer signals and typically better results.
    Handles imbalanced data via SMOTE, class_weight, or undersampling.
    """
    config = _load_config()
    
    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=config["model"]["test_size"], random_state=random_state, stratify=y_enc
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle imbalance
    X_train_bal, y_train_bal = X_train_scaled, y_train
    use_class_weight = True

    if imbalance_strategy == "smote" and HAS_IMBLEARN:
        n_classes = len(np.unique(y_train))
        k = min(5, n_classes - 1) if n_classes > 1 else 1
        try:
            smote = SMOTE(random_state=random_state, k_neighbors=k)
            X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
            use_class_weight = False
        except ValueError:
            pass
    elif imbalance_strategy == "undersample" and HAS_IMBLEARN:
        try:
            rus = RandomUnderSampler(random_state=random_state)
            X_train_bal, y_train_bal = rus.fit_resample(X_train_scaled, y_train)
            use_class_weight = False
        except ValueError:
            pass

    # Compute class weights to emphasize minority (loss) class and improve recall
    class_weight_dict = None
    if use_class_weight:
        labels = np.unique(y_train)
        try:
            cw = compute_class_weight("balanced", classes=labels, y=y_train)
            class_weight_dict = dict(zip(labels, cw))
            # If binary, further amplify minority class weight to prioritize recall
            if len(class_weight_dict) == 2:
                # minority label has larger weight from compute_class_weight
                minority_label = max(class_weight_dict, key=class_weight_dict.get)
                class_weight_dict[minority_label] = float(class_weight_dict[minority_label]) * 2.0
        except Exception:
            class_weight_dict = "balanced"

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight=class_weight_dict if use_class_weight else None,
        max_depth=10,
    )
    clf.fit(X_train_bal, y_train_bal)
    y_pred = clf.predict(X_test_scaled)

    # Binary classification metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    # Core metrics
    acc = accuracy_score(y_test, y_pred)
    prec_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)

    metrics = {
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "accuracy": acc,
        "precision": prec_w,
        "recall": rec_w,
    }

    # Per-class recall (useful to check loss recall)
    try:
        labels = np.unique(y_test)
        _, recall_arr, _, support = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)
        # Map encoded labels back to class names via encoder
        label_map = {i: name for i, name in enumerate(le.classes_)}
        # Build recall dict keyed by class name
        recall_by_class = {label_map[int(lbl)]: float(r) for lbl, r in zip(labels, recall_arr) if int(lbl) < len(label_map)}
        metrics.update({f"recall_{k}": v for k, v in recall_by_class.items()})
    except Exception:
        pass

    # Add ROC-AUC for binary classification (use positive class proba if available)
    if len(np.unique(y_test)) == 2:
        try:
            # Determine positive class index (prefer 'profit' if present)
            pos_name = "profit" if "profit" in le.classes_ else le.classes_[-1]
            pos_idx = int(np.where(le.classes_ == pos_name)[0][0])
            y_proba = clf.predict_proba(X_test_scaled)[:, pos_idx]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            pass

    # Baseline: most-frequent predictor
    try:
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train_scaled, y_train)
        y_base = baseline.predict(X_test_scaled)
        base_acc = accuracy_score(y_test, y_base)
        base_f1 = f1_score(y_test, y_base, average="weighted", zero_division=0)
        base_rec_w = recall_score(y_test, y_base, average="weighted", zero_division=0)
        base_metrics = {"accuracy": base_acc, "f1_weighted": base_f1, "recall_weighted": base_rec_w}
        if len(np.unique(y_test)) == 2:
            try:
                # baseline predict_proba may exist
                if hasattr(baseline, "predict_proba"):
                    pos_name = "profit" if "profit" in le.classes_ else le.classes_[-1]
                    pos_idx = int(np.where(le.classes_ == pos_name)[0][0])
                    y_base_proba = baseline.predict_proba(X_test_scaled)[:, pos_idx]
                    base_metrics["roc_auc"] = float(roc_auc_score(y_test, y_base_proba))
            except Exception:
                pass
        metrics["baseline"] = base_metrics
    except Exception:
        pass
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        model_payload = {"model": clf, "scaler": scaler, "encoder": le, "imbalance": imbalance_strategy}
        # include feature names and importances when available
        try:
            model_payload["feature_names"] = list(X.columns)
        except Exception:
            model_payload["feature_names"] = None
        try:
            if hasattr(clf, "feature_importances_") and model_payload["feature_names"] is not None:
                importances = list(clf.feature_importances_)
                model_payload["feature_importances"] = dict(zip(model_payload["feature_names"], importances))
        except Exception:
            model_payload["feature_importances"] = None

        with open(save_path / "model.pkl", "wb") as f:
            pickle.dump(model_payload, f)
    
    return clf, metrics, le


def evaluate_model(
    clf: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate model and optionally save report."""
    y_pred = clf.predict(X_test)
    
    report = classification_report(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred),
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred)
    
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report).transpose().to_csv(out_path / "classification_report.csv")
        pd.DataFrame(cm).to_csv(out_path / "confusion_matrix.csv")
    
    return {"report": report, "confusion_matrix": cm}
