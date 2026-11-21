"""
split_train_test.py - Chia train/test CÓ STRATIFY, tái lập được.

Cách chạy (từ thư mục gốc dự án):
    python split_train_test.py
    python split_train_test.py --test-size 0.2 --random-state 42

Đầu vào : data/processed/sms_phishing_clean.csv  (sinh ra từ notebook 02)
Đầu ra  : data/processed/train.csv, test.csv, split_info.json
"""
import argparse
import json
import platform
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

LABEL2ID = {"ham": 0, "spam": 1, "smishing": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/sms_phishing_clean.csv")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    a = ap.parse_args()

    clean = pd.read_csv(a.input, encoding="utf-8")
    X = clean[["id", "text_clean", "text_norm", "url", "email", "phone"]]
    y = clean["label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=a.test_size,
        stratify=y,                  # giữ nguyên tỷ lệ nhãn ở cả 2 tập
        random_state=a.random_state, # cố định để tái lập
        shuffle=True,
    )
    train = X_tr.assign(label=y_tr, label_id=y_tr.map(LABEL2ID)).reset_index(drop=True)
    test = X_te.assign(label=y_te, label_id=y_te.map(LABEL2ID)).reset_index(drop=True)

    # Kiểm tra rò rỉ: không tin nào nằm ở cả 2 tập
    overlap = set(train["text_clean"]) & set(test["text_clean"])
    assert not overlap, f"Có {len(overlap)} tin nằm ở cả train và test"

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False, encoding="utf-8")
    test.to_csv(out / "test.csv", index=False, encoding="utf-8")
    info = {"test_size": a.test_size, "random_state": a.random_state, "stratify": "label",
            "n_clean": len(clean), "n_train": len(train), "n_test": len(test),
            "python": platform.python_version(), "pandas": pd.__version__, "sklearn": sklearn.__version__}
    (out / "split_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(info, indent=2, ensure_ascii=False))
    print("\nTỷ lệ nhãn (%):")
    print(pd.DataFrame({"train": train["label"].value_counts(normalize=True) * 100,
                        "test": test["label"].value_counts(normalize=True) * 100}).round(2))


if __name__ == "__main__":
    main()
