"""
train_and_save.py (final + tuning)
===================================
Pipeline đầy đủ với GridSearchCV tuning:
  Đọc → Feature Engineering → Encode → Scale → SMOTE → Tuning → XGBoost → Lưu
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.simplefilter("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="dataset.csv")
    parser.add_argument("--output", default="model.pkl")
    parser.add_argument("--tune",   action="store_true", help="Bật GridSearch tuning")
    args = parser.parse_args()

    print("=" * 60)
    print("   PIPELINE: DỰ ĐOÁN BỎ HỌC - TRAIN XGBOOST (tuning)")
    print("=" * 60)

    # 1. Đọc dữ liệu
    data = pd.read_csv(args.data, sep=";")
    data.columns = [c.replace(" ", "_") for c in data.columns]
    print(f"✅ Đã đọc: {data.shape[0]} dòng, {data.shape[1]} cột")

    # 2. Feature Engineering — tạo 5 cột mới
    data["pass_rate_1"]    = data["Curricular_units_1st_sem_(approved)"] / (data["Curricular_units_1st_sem_(enrolled)"] + 1)
    data["pass_rate_2"]    = data["Curricular_units_2nd_sem_(approved)"] / (data["Curricular_units_2nd_sem_(enrolled)"] + 1)
    data["grade_progress"] = data["Curricular_units_2nd_sem_(grade)"]    - data["Curricular_units_1st_sem_(grade)"]
    data["total_approved"] = data["Curricular_units_1st_sem_(approved)"] + data["Curricular_units_2nd_sem_(approved)"]
    data["avg_grade"]      = (data["Curricular_units_1st_sem_(grade)"]   + data["Curricular_units_2nd_sem_(grade)"]) / 2
    print(f"✅ Feature Engineering: tổng {data.shape[1]} cột (thêm 5 cột mới)")

    # 3. Encode Target
    le = LabelEncoder()
    data["Target"] = le.fit_transform(data["Target"])
    print(f"✅ Nhãn: {list(le.classes_)} → [0, 1, 2]")

    # 4. Tách X, y
    X = data.drop(columns=["Target"])
    y = data["Target"]
    feature_cols = list(X.columns)
    print(f"✅ Tổng features đưa vào model: {len(feature_cols)}")

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

    # 8. Tuning hoặc train thẳng
    print("\n🔍 Bắt đầu GridSearch tuning (khoảng 10-15 phút)...")
    print("   Đang thử các combinations tham số...\n")

    param_grid = {
        "n_estimators":  [200, 400, 600],
        "max_depth":     [5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample":     [0.8, 1.0],
        "colsample_bytree": [0.7, 0.9],
    }

    base_model = XGBClassifier(
        random_state=42,
        eval_metric="mlogloss",
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,       # dùng tất cả CPU core → nhanh hơn
        verbose=1,       # in tiến trình
    )

    grid_search.fit(X_res, y_res)

    print(f"\n✅ Tuning xong!")
    print(f"🏆 Tham số tốt nhất:")
    for k, v in grid_search.best_params_.items():
        print(f"   {k}: {v}")
    print(f"📊 CV Accuracy tốt nhất: {grid_search.best_score_*100:.2f}%")

    model = grid_search.best_estimator_

    # 9. Đánh giá trên test set
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\n🎯 Test Accuracy: {acc:.2f}%")
    print(classification_report(y_test, y_pred,
          target_names=["Dropout", "Enrolled", "Graduate"]))

    # 10. Lưu pipeline
    pipeline = {
        "model":         model,
        "label_encoder": le,
        "feature_cols":  feature_cols,
        "scaler":        scaler,
    }
    with open(args.output, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"💾 Đã lưu → {args.output}")
    print("\n✅ Chạy web: streamlit run app.py")


if __name__ == "__main__":
    main()
