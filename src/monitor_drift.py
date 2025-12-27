import argparse
import json
import math
import os
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import joblib


def ks_2samp_pvalue(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    n1 = x.size
    n2 = y.size
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    data_all = np.concatenate([x_sorted, y_sorted])
    cdf1 = np.searchsorted(x_sorted, data_all, side="right") / n1
    cdf2 = np.searchsorted(y_sorted, data_all, side="right") / n2
    d = float(np.max(np.abs(cdf1 - cdf2)))

    en = math.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d

    if lam <= 0:
        return d, 1.0

    s = 0.0
    for j in range(1, 101):
        term = (-1) ** (j - 1) * math.exp(-2.0 * (j * j) * (lam * lam))
        s += term
        if abs(term) < 1e-10:
            break

    p = max(0.0, min(1.0, 2.0 * s))
    return d, p


def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)

    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(ref, quantiles)
    edges = np.unique(edges)
    if edges.size < 3:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_counts / max(1, ref_counts.sum())
    cur_pct = cur_counts / max(1, cur_counts.sum())

    ref_pct = np.clip(ref_pct, eps, 1)
    cur_pct = np.clip(cur_pct, eps, 1)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_categorical(ref: pd.Series, cur: pd.Series, eps: float = 1e-6) -> float:
    ref = ref.astype("object").fillna("__NA__")
    cur = cur.astype("object").fillna("__NA__")

    ref_counts = ref.value_counts(dropna=False)
    cur_counts = cur.value_counts(dropna=False)

    cats = ref_counts.index.union(cur_counts.index)

    ref_pct = (ref_counts.reindex(cats, fill_value=0) / max(1, len(ref))).to_numpy()
    cur_pct = (cur_counts.reindex(cats, fill_value=0) / max(1, len(cur))).to_numpy()

    ref_pct = np.clip(ref_pct, eps, 1)
    cur_pct = np.clip(cur_pct, eps, 1)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def is_high_cardinality_id_like(s: pd.Series, max_unique_ratio: float = 0.5) -> bool:
    if s.dtype.kind not in ("O", "U", "S"):
        return False
    n = len(s)
    if n == 0:
        return False
    uniq = s.nunique(dropna=True)
    return (uniq / n) > max_unique_ratio


def load_probability(model: Any, df: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)
        try:
            return np.asarray(proba)[:, 1]
        except Exception:
            pass

    pred = model.predict(df)
    pred = np.asarray(pred).reshape(-1)
    if np.any(pred < 0) or np.any(pred > 1):
        pred = (pred >= 1).astype(float)
    return np.clip(pred.astype(float), 0.0, 1.0)


def _write_html_placeholder(path: str, title: str, message: str) -> None:
    safe_msg = (message or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    pre {{ background: #f6f8fa; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>This report was generated as a placeholder.</p>
  <pre>{safe_msg}</pre>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/processed/train.csv")
    ap.add_argument("--current", default="data/processed/test.csv")
    ap.add_argument("--target", default="churn")
    ap.add_argument("--model-path", default="models/model.joblib")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--psi-threshold", type=float, default=0.2)
    ap.add_argument("--ks-p-threshold", type=float, default=0.05)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--ignore-cols", default="", help="Comma-separated columns to ignore")
    args = ap.parse_args()

    ignore_cols = [c.strip() for c in args.ignore_cols.split(",") if c.strip()]

    os.makedirs(args.out_dir, exist_ok=True)

    out_json = os.path.join(args.out_dir, "drift_metrics.json")
    out_html = os.path.join(args.out_dir, "data_drift_evidently.html")
    out_err = os.path.join(args.out_dir, "evidently_error.txt")

    # Ensure outputs exist even if we fail later (for DVC outs contract)
    if not os.path.exists(out_err):
        with open(out_err, "w", encoding="utf-8") as f:
            f.write("")
    if not os.path.exists(out_html):
        _write_html_placeholder(out_html, "Evidently Data Drift Report", "Not generated yet.")

    ref_df = pd.read_csv(args.reference)
    cur_df = pd.read_csv(args.current)

    if len(ref_df) == 0 or len(cur_df) == 0:
        raise SystemExit("Reference/current dataset is empty; cannot compute drift.")

    if args.target not in ref_df.columns or args.target not in cur_df.columns:
        raise SystemExit(f"Target column '{args.target}' not found in both datasets.")

    ref_X = ref_df.drop(columns=[args.target])
    cur_X = cur_df.drop(columns=[args.target])

    if not os.path.exists(args.model_path):
        raise SystemExit(f"Model file not found: {args.model_path}")
    model = joblib.load(args.model_path)

    ref_pred = load_probability(model, ref_X)
    cur_pred = load_probability(model, cur_X)

    ref_Xp = ref_X.copy()
    cur_Xp = cur_X.copy()
    ref_Xp["prediction"] = ref_pred
    cur_Xp["prediction"] = cur_pred

    columns: List[str] = []
    for c in ref_Xp.columns:
        if c in ignore_cols:
            continue
        if c not in cur_Xp.columns:
            continue
        if c.lower().endswith("id") or c.lower() in ("id", "customer_id"):
            continue
        if is_high_cardinality_id_like(ref_Xp[c]):
            continue
        columns.append(c)

    psi: Dict[str, float] = {}
    ks: Dict[str, Dict[str, float]] = {}

    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(ref_Xp[c])]
    cat_cols = [c for c in columns if not pd.api.types.is_numeric_dtype(ref_Xp[c])]

    for c in numeric_cols:
        psi[c] = psi_numeric(ref_Xp[c].to_numpy(), cur_Xp[c].to_numpy(), bins=args.bins)
        d, p = ks_2samp_pvalue(ref_Xp[c].to_numpy(), cur_Xp[c].to_numpy())
        ks[c] = {"D": float(d), "p_value": float(p)}

    for c in cat_cols:
        psi[c] = psi_categorical(ref_Xp[c], cur_Xp[c])

    drifted_psi = sorted([c for c, v in psi.items() if v > args.psi_threshold], key=lambda x: psi[x], reverse=True)
    drifted_ks = sorted([c for c, v in ks.items() if v["p_value"] < args.ks_p_threshold], key=lambda x: ks[x]["p_value"])

    prediction_psi = psi_numeric(ref_pred, cur_pred, bins=args.bins)

    alerts: List[Dict[str, Any]] = []
    if prediction_psi > args.psi_threshold:
        alerts.append({
            "severity": "warning",
            "type": "prediction_drift",
            "message": f"Prediction PSI={prediction_psi:.4f} > {args.psi_threshold}",
        })
    if len(drifted_psi) > 0:
        alerts.append({
            "severity": "warning",
            "type": "data_drift",
            "message": f"{len(drifted_psi)}/{len(columns)} columns drifted by PSI > {args.psi_threshold}",
        })

    summary = {
        "reference_rows": int(len(ref_df)),
        "current_rows": int(len(cur_df)),
        "psi_threshold": float(args.psi_threshold),
        "ks_p_threshold": float(args.ks_p_threshold),
        "columns_evaluated": int(len(columns)),
        "psi_max": float(max(psi.values()) if psi else 0.0),
        "psi_drifted_share": float(len(drifted_psi) / max(1, len(columns))),
        "ks_drifted_share_numeric": float(len(drifted_ks) / max(1, len(numeric_cols))),
        "prediction_psi": float(prediction_psi),
        "drifted_columns_psi": drifted_psi[:50],
        "drifted_columns_ks": drifted_ks[:50],
        "alerts": alerts,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "psi": psi, "ks": ks}, f, indent=2, ensure_ascii=False)

    # Evidently HTML report: ALWAYS ensure out_html exists (DVC outs requirement)
    evidently_error: Optional[str] = None
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report([DataDriftPreset(method="psi")], include_tests=True)
        eval_obj = report.run(cur_Xp, ref_Xp)
        eval_obj.save_html(out_html)

        # success -> empty error file
        with open(out_err, "w", encoding="utf-8") as f:
            f.write("")
    except Exception as e:
        evidently_error = str(e)
        with open(out_err, "w", encoding="utf-8") as f:
            f.write(evidently_error)
        _write_html_placeholder(out_html, "Evidently Data Drift Report (placeholder)", evidently_error)

    print(f"Drift metrics saved: {out_json}")
    print(f"Evidently HTML saved: {out_html}")
    print(f"Evidently error file: {out_err}")
    if alerts:
        print("ALERTS:")
        for a in alerts:
            print(f"- [{a['severity']}] {a['type']}: {a['message']}")
    if evidently_error:
        print(f"Evidently generation failed (placeholder HTML created): {evidently_error}")


if __name__ == "__main__":
    main()
