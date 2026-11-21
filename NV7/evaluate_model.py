import json
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# 1. Đường dẫn dữ liệu
V2_PRED_PATH = r"D:\NV7\y_test_y_pred_v2.csv"
V1_PRED_PATH = r"D:\NV6\04_Evaluation\y_test_y_pred.csv"

OUT_CM_PATH = r"D:\NV7\confusion_matrix_v2.csv"
OUT_EVAL_JSON = r"D:\NV7\evaluation_v2.json"
OUT_WRONG_CSV = r"D:\NV7\misclassified_samples_v2.csv"

labels = ["ham", "spam", "smishing"]


def evaluate_predictions(df, name="Model V2 (TF-IDF + RandomOverSampler + MultinomialNB)"):
    y_test = df["y_test"]
    y_pred = df["y_pred"]

    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="weighted", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="micro", zero_division=0
    )
    p_cls, r_cls, f1_cls, s_cls = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average=None, zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    print("=" * 65)
    print(f"KẾT QUẢ ĐÁNH GIÁ: {name}")
    print("=" * 65)
    print(f"Tổng số mẫu tập test : {len(df)}")
    print(f"Tổng số mẫu trong CM : {cm.sum()} (Khớp hoàn toàn: {cm.sum() == len(df)})")
    print(f"Accuracy             : {acc:.4f} ({acc * 100:.2f}%)")
    print("-" * 65)
    print(f"Macro Average        -> Precision: {p_macro:.4f} | Recall: {r_macro:.4f} | F1-score: {f1_macro:.4f}")
    print(f"Weighted Average     -> Precision: {p_weighted:.4f} | Recall: {r_weighted:.4f} | F1-score: {f1_weighted:.4f}")
    print(f"Micro Average        -> Precision: {p_micro:.4f} | Recall: {r_micro:.4f} | F1-score: {f1_micro:.4f}")
    print("-" * 65)
    print("CHI TIẾT TỪNG LỚP:")
    for idx, label in enumerate(labels):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        print(
            f"  - Lớp '{label:8s}' (Support={s_cls[idx]:3d}): "
            f"Precision={p_cls[idx]:.4f}, Recall={r_cls[idx]:.4f}, F1={f1_cls[idx]:.4f} | "
            f"Đúng(TP)={tp:3d}, Sai(FN)={fn:2d}, Báo nhầm vào lớp này(FP)={fp:2d}, TN={tn:4d}"
        )

    print("\nCONFUSION MATRIX (Hàng = Thực tế [y_test], Cột = Dự đoán [y_pred]):")
    print(cm_df)
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, labels=labels, digits=4, zero_division=0))

    return {
        "model": name,
        "total_test_samples": int(len(df)),
        "confusion_matrix_sum": int(cm.sum()),
        "matches_test_size": bool(cm.sum() == len(df)),
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "micro_precision": float(p_micro),
        "micro_recall": float(r_micro),
        "micro_f1": float(f1_micro),
        "per_class": {
            label: {
                "support": int(s_cls[i]),
                "precision": float(p_cls[i]),
                "recall": float(r_cls[i]),
                "f1_score": float(f1_cls[i]),
                "correct_tp": int(cm[i, i]),
                "missed_fn": int(cm[i, :].sum() - cm[i, i]),
                "false_alarm_fp": int(cm[:, i].sum() - cm[i, i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }, cm_df


def main():
    v2_df = pd.read_csv(V2_PRED_PATH)
    eval_v2, cm_df_v2 = evaluate_predictions(
        v2_df, "Mô hình V2 (TF-IDF + RandomOverSampler + MultinomialNB)"
    )

    cm_df_v2.to_csv(OUT_CM_PATH, encoding="utf-8-sig")

    if os.path.exists(V1_PRED_PATH):
        v1_df = pd.read_csv(V1_PRED_PATH)
        eval_v1, _ = evaluate_predictions(
            v1_df, "Mô hình V1 Baseline (TF-IDF + MultinomialNB không cân bằng dữ liệu)"
        )
        eval_v2["baseline_v1_comparison"] = eval_v1

        merged = v2_df.copy()
        if "id" in v1_df.columns:
            merged.insert(0, "id", v1_df["id"])
        merged["text_clean"] = v1_df["text_clean"]
        merged["y_pred_v1"] = v1_df["y_pred"]
        for col in ["probability_ham", "probability_smishing", "probability_spam"]:
            if col in v1_df.columns:
                merged[f"v1_{col}"] = v1_df[col]

        wrong_v2 = merged[merged["y_test"] != merged["y_pred"]].copy()
        wrong_v2.index.name = "test_index"
        wrong_v2.to_csv(OUT_WRONG_CSV, encoding="utf-8-sig")
        print(f"Tổng số mẫu dự đoán sai ở V2: {len(wrong_v2)} / {len(v2_df)}")
        print("\nDANH SÁCH 44 MẪU DỰ ĐOÁN SAI Ở MÔ HÌNH V2:")
        for idx, row in wrong_v2.iterrows():
            rid = row["id"] if "id" in row else idx
            print(
                f"[{idx:4d} | id={rid:4d}] Thực tế: {row['y_test']:8s} -> Dự đoán V2: {row['y_pred']:8s} "
                f"(V1: {row['y_pred_v1']:8s}) | Text: {row['text_clean']}"
            )

    with open(OUT_EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_v2, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
