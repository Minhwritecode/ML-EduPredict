"""
train_and_save.py (final + tuning + ensemble)
==============================================
Pipeline đầy đủ nhất:
  Đọc → Feature Engineering → Encode → Scale → SMOTE → 
  GridSearch Tuning → Voting Ensemble (XGBoost + LightGBM + RandomForest) → Lưu
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.simplefilter("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="dataset.csv")
    parser.add_argument("--output", default="model.pkl")
    args = parser.parse_args()

    print("=" * 60)
    print("   PIPELINE: DỰ ĐOÁN BỎ HỌC - VOTING ENSEMBLE")
    print("   XGBoost + LightGBM + Random Forest")
    print("=" * 60)

    # 1. Đọc dữ liệu
    data = pd.read_csv(args.data, sep=";")
    data.columns = [c.replace(" ", "_") for c in data.columns]
    print(f"✅ Đã đọc: {data.shape[0]} dòng, {data.shape[1]} cột")

    # 2. Feature Engineering — 5 cột mới
    data["pass_rate_1"]    = data["Curricular_units_1st_sem_(approved)"] / (data["Curricular_units_1st_sem_(enrolled)"] + 1)
    data["pass_rate_2"]    = data["Curricular_units_2nd_sem_(approved)"] / (data["Curricular_units_2nd_sem_(enrolled)"] + 1)
    data["grade_progress"] = data["Curricular_units_2nd_sem_(grade)"]    - data["Curricular_units_1st_sem_(grade)"]
    data["total_approved"] = data["Curricular_units_1st_sem_(approved)"] + data["Curricular_units_2nd_sem_(approved)"]
    data["avg_grade"]      = (data["Curricular_units_1st_sem_(grade)"]   + data["Curricular_units_2nd_sem_(grade)"]) / 2
    print(f"✅ Feature Engineering: tổng {data.shape[1]} cột")

    # 3. Encode Target
    le = LabelEncoder()
    data["Target"] = le.fit_transform(data["Target"])
    print(f"✅ Nhãn: {list(le.classes_)} → [0, 1, 2]")

    # 4. Tách X, y
    X = data.drop(columns=["Target"])
    y = data["Target"]
    feature_cols = list(X.columns)
    print(f"✅ Tổng features: {len(feature_cols)}")

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 7. SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_s, y_train)
    print(f"✅ SMOTE: {X_res.shape}")

    # 8. Khởi tạo 3 model với tham số tốt
    print("\n🚀 Khởi tạo 3 models...")

    xgb_model = XGBClassifier(
        n_estimators=400,
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

    lgb_model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    # 9. Voting Ensemble — soft voting = dùng xác suất trung bình
    print("🗳️  Tạo Voting Ensemble (soft voting)...")
    ensemble = VotingClassifier(
        estimators=[
            ("xgboost",      xgb_model),
            ("lightgbm",     lgb_model),
            ("randomforest",  rf_model),
        ],
        voting="soft",   # dùng xác suất, chính xác hơn hard voting
        weights=[3, 2, 1],  # XGBoost tin cậy nhất → trọng số cao hơn
    )

    print("\n⏳ Đang train Ensemble (mất khoảng 3-5 phút)...")
    ensemble.fit(X_res, y_res)
    print("✅ Train xong!")

    # 10. Đánh giá từng model riêng lẻ
    print("\n📊 Đánh giá từng model:")
    for name, m in [("XGBoost", xgb_model), ("LightGBM", lgb_model), ("RandomForest", rf_model)]:
        m.fit(X_res, y_res)
        acc = accuracy_score(y_test, m.predict(X_test_s)) * 100
        print(f"   {name}: {acc:.2f}%")

    # 11. Đánh giá Ensemble
    y_pred = ensemble.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\n🏆 Voting Ensemble Accuracy: {acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Dropout", "Enrolled", "Graduate"]))

    # 12. Lưu pipeline — lưu ensemble thay vì model đơn lẻ
    pipeline = {
        "model":         ensemble,
        "label_encoder": le,
        "feature_cols":  feature_cols,
        "scaler":        scaler,
    }
    with open(args.output, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n💾 Đã lưu → {args.output}")
    print("✅ Chạy web: streamlit run app.py")


if __name__ == "__main__":
    main()
