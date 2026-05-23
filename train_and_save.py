"""
train_and_save.py (final)
=========================
Pipeline tối ưu thực tế:
  Đọc → Encode → Tất cả features → Scale → SMOTE → XGBoost mạnh → Lưu
"""

import argparse
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.simplefilter("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="dataset.csv")
    parser.add_argument("--output", default="model.pkl")
    args = parser.parse_args()

    print("=" * 55)
    print("   PIPELINE: DỰ ĐOÁN BỎ HỌC - TRAIN XGBOOST (final)")
    print("=" * 55)

    # 1. Đọc
    data = pd.read_csv(args.data, sep=";")
    data.columns = [c.replace(" ", "_") for c in data.columns]
    print(f"✅ Đã đọc: {data.shape[0]} dòng, {data.shape[1]} cột")

    # 2. Encode Target
    le = LabelEncoder()
    data["Target"] = le.fit_transform(data["Target"])
    print(f"✅ Nhãn: {list(le.classes_)} → {[0,1,2]}")

    # 3. Tất cả features
    X = data.drop(columns=["Target"])
    y = data["Target"]
    feature_cols = list(X.columns)
    print(f"✅ Dùng toàn bộ {len(feature_cols)} features")

    # 4. Split không stratify (giống notebook gốc)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 6. SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_s, y_train)
    print(f"✅ SMOTE: {X_res.shape}")

    # 7. XGBoost với tham số mạnh hơn
    print("\n🚀 Đang train XGBoost...")
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(X_res, y_res)

    # 8. Evaluate
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\n🎯 XGBoost Accuracy: {acc:.2f}%")
    print(classification_report(y_test, y_pred,
          target_names=["Dropout","Enrolled","Graduate"]))

    # 9. Lưu
    pipeline = {
        "model": model,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "scaler": scaler,
    }
    with open(args.output, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"💾 Đã lưu → {args.output}")
    print("\n✅ Chạy web: streamlit run app.py")


if __name__ == "__main__":
    main()
