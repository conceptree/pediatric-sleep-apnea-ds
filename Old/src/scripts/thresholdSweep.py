import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import (precision_recall_curve, roc_curve, auc,
                             average_precision_score, classification_report)

def sweep_threshold(y_true, prob, steps=1001):
    thrs = np.linspace(0,1,steps)
    rows=[]
    for t in thrs:
        pred = (prob>=t).astype(int)
        tp = ((pred==1)&(y_true==1)).sum()
        fp = ((pred==1)&(y_true==0)).sum()
        tn = ((pred==0)&(y_true==0)).sum()
        fn = ((pred==0)&(y_true==1)).sum()
        prec = tp/(tp+fp+1e-9)
        rec  = tp/(tp+fn+1e-9)
        spec = tn/(tn+fp+1e-9)
        f1   = 2*prec*rec/(prec+rec+1e-9)
        rows.append(dict(thr=float(t), precision=float(prec), recall=float(rec),
                         specificity=float(spec), f1=float(f1), tp=int(tp), fp=int(fp),
                         tn=int(tn), fn=int(fn)))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model-probs", required=False,
                    help="CSV opcional com colunas [stem,prob_pos]; se não passar, usa prob_pos do features.parquet (se existir).")
    ap.add_argument("--subset-val", default="val")
    ap.add_argument("--objective", choices=["youden","max_f1","recall_at_precision","precision_at_recall"], default="youden")
    ap.add_argument("--target", type=float, default=0.90,
                    help="alvo para recall_at_precision (prec>=target) ou precision_at_recall (rec>=target)")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    args=ap.parse_args()

    df = pd.read_parquet(args.features)
    if args.model_probs:
        p = pd.read_csv(args.model_probs)
        df = df.merge(p[["stem","prob_pos"]], on="stem", how="left")
    if "prob_pos" not in df.columns:
        raise SystemExit("preciso da coluna prob_pos (gera com xgbCvPrediction.py ou trainModels.py com save preds).")
    mask = df["subset"]==args.subset_val
    y = df.loc[mask,"label"].astype(int).values
    prob = df.loc[mask,"prob_pos"].values
    sweep = sweep_threshold(y, prob)
    # escolher
    if args.objective=="youden":
        # maximize sens+spec-1 (sen=rec)
        sweep["youden"] = sweep["recall"] + sweep["specificity"] - 1
        best = sweep.sort_values("youden", ascending=False).iloc[0]
    elif args.objective=="max_f1":
        best = sweep.sort_values("f1", ascending=False).iloc[0]
    elif args.objective=="recall_at_precision":
        cand = sweep[sweep["precision"]>=args.target]
        best = cand.sort_values("recall", ascending=False).iloc[0] if len(cand) else sweep.iloc[sweep["precision"].idxmax()]
    else: # precision_at_recall
        cand = sweep[sweep["recall"]>=args.target]
        best = cand.sort_values("precision", ascending=False).iloc[0] if len(cand) else sweep.iloc[sweep["recall"].idxmax()]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.out_csv, index=False)
    meta = dict(best_threshold=float(best["thr"]), best_row=best.to_dict(),
                objective=args.objective, target=args.target, n_val=int(mask.sum()))
    Path(args.out_json).write_text(json.dumps(meta, indent=2))
    print(f"✅ threshold({args.objective})={meta['best_threshold']:.3f} salvo")
if __name__=="__main__":
    main()
