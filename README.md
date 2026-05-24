# ML-EduPredict — Dự đoán Bỏ học & Kết quả Học tập Sinh viên

Dự án **Học máy (Machine Learning)** phân loại đa lớp: dự đoán sinh viên **Dropout** (bỏ học), **Enrolled** (đang học) hoặc **Graduate** (tốt nghiệp). Luồng chính: **phân tích + huấn luyện trong Jupyter Notebook** → xuất `model.pkl` → **ứng dụng Streamlit** (`app.py`) dự đoán và demo.

**Repository:** [https://github.com/Minhwritecode/ML-EduPredict](https://github.com/Minhwritecode/ML-EduPredict)

---

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [Bài toán & dữ liệu](#bài-toán--dữ-liệu)
3. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
4. [Quy trình trong Notebook (train chính)](#quy-trình-trong-notebook-train-chính)
5. [Xuất `model.pkl` từ Notebook](#xuất-modelpkl-từ-notebook)
6. [Cài đặt & chạy](#cài-đặt--chạy)
7. [Ứng dụng Streamlit (`app.py`)](#ứng-dụng-streamlit-apppy)
8. [Kịch bản demo](#kịch-bản-demo)
9. [Git & file không push](#git--file-không-push)
10. [Xử lý sự cố](#xử-lý-sự-cố)

---

## Tổng quan

| Thành phần | Vai trò |
|------------|---------|
| `dataset.csv` | 4.424 mẫu, 36 feature + `Target`, phân cách `;` |
| `Predict students' dropout and academic success.ipynb` | **EDA, xử lý, huấn luyện 7 mô hình, xuất `model.pkl`** |
| `Predict students' dropout and academic success.html` | Bản export notebook (xem không cần Jupyter) |
| `model.pkl` | Pipeline đã fit — **tạo từ notebook**, không commit Git |
| `app.py` | Web Streamlit: dự đoán, dashboard, lịch sử, báo cáo HTML |
| `Demo Scenario.text` | Kịch bản trình bày hội đồng / demo 12–25 phút |

**Luồng làm việc chuẩn:**

```
dataset.csv
    → Chạy notebook (train)
    → Lưu model.pkl (cell xuất ở cuối notebook)
    → streamlit run app.py
```

---

## Bài toán & dữ liệu

| Nhãn `Target` | Ý nghĩa | Tỷ lệ (~) |
|---------------|---------|-----------|
| `Graduate` | Tốt nghiệp | 49,9% |
| `Dropout` | Bỏ học | 32,1% |
| `Enrolled` | Đang học | 17,1% |

- **File:** `dataset.csv` — 4.424 dòng × 37 cột, không missing, không trùng hàng.
- **Nguồn:** bộ công khai *Predict students' dropout and academic success* (bối cảnh đại học Bồ Đào Nha).
- Sau khi đọc, notebook đổi tên cột: khoảng trắng → `_` (ví dụ `Marital status` → `Marital_status`).

---

## Cấu trúc thư mục

```
ML2026/
├── dataset.csv
├── Predict students' dropout and academic success.ipynb   # Train + phân tích
├── Predict students' dropout and academic success.html
├── app.py
├── model.pkl                    # Tạo bằng notebook (local, không push)
├── prediction_history.json      # Lịch sử app (local, không push)
├── Demo Scenario.text
├── README.md
├── .gitignore
└── venv/                        # Không push
```

---

## Quy trình trong Notebook (train chính)

File: `Predict students' dropout and academic success.ipynb` (~110 cells)

### Sơ đồ pipeline

```
[Dữ liệu thô]
    → Làm sạch, EDA, encode Target (LabelEncoder)
    → Xử lý outlier (IQR — 4 cột điểm)
    → Chi-square: loại 4 đặc trư không liên quan Target
    → StandardScaler (Z-score)
    → PCA (10 thành phần chính)
    → K-Means (3 cụm) → gắn nhãn cụm
    → Ma trận X_final (11 chiều: 10 PC + 1 cluster)
    → Train/Test split 80/20
    → SMOTE (chỉ trên train)
    → Huấn luyện & so sánh 7 classifier
```

### Bảy mô hình so sánh

| # | Mô hình |
|---|---------|
| 1 | Logistic Regression |
| 2 | Gaussian Naive Bayes |
| 3 | Decision Tree |
| 4 | Random Forest |
| 5 | **XGBoost** (thường dẫn đầu accuracy) |
| 6 | LightGBM |
| 7 | CatBoost |

Kết luận trong notebook: **XGBoost** phù hợp deploy (pattern phi tuyến, boosting). Có thể chọn mô hình khác nếu metric trên tập test của bạn tốt hơn.

### Các phần chính trong notebook

| Phần | Nội dung |
|------|----------|
| 1 | Thu thập & chuẩn bị dữ liệu |
| 2 | EDA — phân bố Target, biểu đồ, nhận định (nợ HP, giới tính, …) |
| 3 | Feature engineering / selection |
| 4 | Outlier IQR |
| 5 | Chi-square feature selection (bỏ 4 cột) |
| 6 | PCA + K-Means |
| 7 | SMOTE + train/test |
| 8 | Huấn luyện 7 mô hình + confusion matrix |
| 9 | Bảng so sánh accuracy + kết luận |
| 10 | (Tuỳ chọn) Demo Streamlit đơn giản trong notebook |

> **Lưu ý:** Pipeline notebook dùng **PCA + K-Means + SMOTE** trên không gian 11 chiều. `app.py` tự nhận pipeline này nếu `model.pkl` chứa `pca` và `kmeans` (sidebar hiển thị **PCA+KMeans**).

---

## Xuất `model.pkl` từ Notebook

Sau khi chạy xong các cell train (đặc biệt cell **XGBoost** hoặc mô hình bạn chọn deploy), thêm **một cell mới** ở cuối notebook và chạy:

```python
import pickle

# le, scaler, pca, kmeans, X_after_chi2, xgb_model đã có từ các cell trước
pipeline = {
    "model": xgb_model,                    # hoặc rf_model, lgb_model, ...
    "label_encoder": le,
    "feature_cols": list(X_after_chi2.columns),  # cột sau Chi-square (≈32 cột)
    "scaler": scaler,
    "pca": pca,
    "kmeans": kmeans,
}

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Đã lưu model.pkl — chạy: streamlit run app.py")
```

### Cấu trúc `model.pkl`

```python
{
    "model":         ...,              # classifier đã fit (vd. XGBClassifier)
    "label_encoder": LabelEncoder,     # Dropout / Enrolled / Graduate
    "feature_cols":  list[str],      # tên cột sau Chi-square, đúng thứ tự
    "scaler":        StandardScaler,
    "pca":           PCA,             # bắt buộc cho pipeline notebook
    "kmeans":        KMeans,
}
```

### Cách `app.py` dùng file này

1. Form → `DataFrame` 1 dòng (đủ trường nhập).
2. Bổ sung cột thiếu trong `feature_cols` = `0`.
3. `scaler.transform(df[feature_cols])`
4. `pca.transform` + `kmeans.predict` → vector 11 chiều.
5. `model.predict` / `predict_proba` → nhãn + xác suất.

**Không cần** script `train_and_save.py` riêng nếu bạn train hoàn toàn trong notebook.

---

## Cài đặt & chạy

### Yêu cầu

- Python **3.9+**
- RAM ≥ 8 GB (train XGBoost / CatBoost trong notebook)

### Cài thư viện

```bash
cd /đường/dẫn/ML2026
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install pandas numpy scikit-learn imbalanced-learn \
  xgboost lightgbm catboost matplotlib seaborn \
  streamlit altair jupyter
```

### Bước 1 — Train trong Notebook

```bash
jupyter notebook "Predict students' dropout and academic success.ipynb"
```

1. **Kernel → Restart & Run All** (lần đầu mất vài phút), **hoặc**
2. Chạy tuần tự từ đầu đến hết phần huấn luyện XGBoost (cell 95–97).
3. Chạy **cell xuất `model.pkl`** (mục trên).
4. Kiểm tra file `model.pkl` xuất hiện cùng thư mục project (~ vài MB đến vài chục MB tuỳ mô hình).

### Bước 2 — Chạy web

```bash
streamlit run app.py
```

Mở `http://localhost:8501` — sidebar: **✅ model.pkl đã load**, pipeline **PCA+KMeans**.

---

## Ứng dụng Streamlit (`app.py`)

### Ba trang (sidebar)

| Trang | Chức năng |
|-------|-----------|
| **🔮 Dự đoán** | Form, 5 preset demo, kết quả + giải thích + biểu đồ |
| **📊 Dashboard** | KPI phiên, % phân bố, top 3 nguy cơ Dropout |
| **📜 Lịch sử** | Timeline thẻ, so sánh 2 SV, CSV, `prediction_history.json` |

### Tính năng nổi bật

- Dropdown **5 ca demo** (Minh Anh, Văn Bình, Thu Hà, Lan Phương, Hoàng Nam) — khớp `Demo Scenario.text`
- Form **một khối**, step indicator 4 bước, tooltip từng trường
- Dark / Light mode, footer, gauge Dropout, heatmap HK1/HK2
- Giải thích “Vì sao mô hình dự đoán như vậy?”, tải báo cáo HTML
- Lịch sử lưu local qua F5 (`prediction_history.json`)

### 5 preset — kết quả tham chiếu

*(Sau khi export `model.pkl` mới từ notebook, chạy lại preset; xác suất có thể lệch ± vài %.)*

| Preset | Kết quả kỳ vọng |
|--------|-----------------|
| ⭐ Minh Anh | Graduate |
| ⚠️ Văn Bình | Dropout (0 tín chỉ) |
| 🔴 Thu Hà | Dropout (đăng ký nhưng qua 0 môn) |
| 📚 Lan Phương | Enrolled |
| 🚨 Hoàng Nam | Dropout (nợ HP + điểm thấp) |

---

## Kịch bản demo

Chi tiết lời dẫn, timeline 12/25 phút, checklist: xem **`Demo Scenario.text`**.

Tóm tắt:

1. Chuẩn bị: đã chạy notebook → có `model.pkl`.
2. `streamlit run app.py`
3. Demo từng preset → Dashboard → Lịch sử (so sánh Minh Anh vs Lan Phương).

---

## Git & file không push

`.gitignore` đã loại:

| File / thư mục | Lý do |
|----------------|--------|
| `venv/` | Môi trường local |
| `model.pkl`, `*.pkl` | File lớn, tạo lại bằng notebook |
| `prediction_history.json` | Dữ liệu demo local |
| `.env` | Bí mật |

### Push an toàn

```bash
git status
git add README.md app.py "Demo Scenario.text" .gitignore \
  "Predict students' dropout and academic success.ipynb"
git diff --cached --name-only    # KHÔNG có model.pkl / venv/
git commit -m "Cập nhật notebook train, app Streamlit và tài liệu"
git push origin main             # hoặc nhánh feature của bạn
```

---

## Xử lý sự cố

| Triệu chứng | Cách xử lý |
|-------------|------------|
| Sidebar ❌ thiếu `model.pkl` | Chạy notebook + cell xuất pickle |
| Mọi ca đều Dropout | `model.pkl` không khớp app — train lại và export đủ `pca`, `kmeans`, `feature_cols` |
| Xác suất khác tài liệu demo | Train lại; thứ hạng lớp (Graduate vs Dropout) quan trọng hơn số % chính xác |
| Notebook lỗi `catboost` | `pip install catboost` |
| Cột `Daytime/evening_attendance\t` | Giữ đúng tên có tab khi map từ form |
| `model.pkl` cũ (không có `pca`) | App dùng chế độ Scale+Ensemble — nên export lại từ notebook hiện tại |

---

## Tác giả

- **GitHub:** [Minhwritecode/ML-EduPredict](https://github.com/Minhwritecode/ML-EduPredict)
- Dự án học tập / đồ án **ML2026**.

Trích dẫn dataset gốc khi dùng trong báo cáo khoa học.

---

## Tóm tắt lệnh

```bash
# 1. Cài & kích hoạt venv
source venv/bin/activate

# 2. Train + export trong Jupyter
jupyter notebook "Predict students' dropout and academic success.ipynb"
# → Run All (hoặc đến XGBoost) → chạy cell lưu model.pkl

# 3. Web
streamlit run app.py
```
