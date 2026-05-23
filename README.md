# ML-EduPredict — Dự đoán Bỏ học & Kết quả Học tập Sinh viên

Dự án **Học máy (Machine Learning)** phân loại đa lớp nhằm dự đoán kết quả học tập của sinh viên đại học: **Bỏ học (Dropout)**, **Đang học (Enrolled)** hoặc **Tốt nghiệp (Graduate)**. Hệ thống gồm quy trình phân tích dữ liệu (Jupyter Notebook), script huấn luyện mô hình production và ứng dụng web **Streamlit** để dự đoán trực tiếp.

**Repository:** [https://github.com/Minhwritecode/ML-EduPredict](https://github.com/Minhwritecode/ML-EduPredict)

---

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [Bài toán & mục tiêu](#bài-toán--mục-tiêu)
3. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
4. [Bộ dữ liệu](#bộ-dữ-liệu)
5. [Quy trình xử lý & mô hình](#quy-trình-xử-lý--mô-hình)
6. [Cài đặt môi trường](#cài-đặt-môi-trường)
7. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
8. [Ứng dụng web Streamlit](#ứng-dụng-web-streamlit)
9. [Jupyter Notebook (phân tích & thử nghiệm)](#jupyter-notebook-phân-tích--thử-nghiệm)
10. [Mô hình đã lưu (`model.pkl`)](#mô-hình-đã-lưu-modelpkl)
11. [Lịch sử phát triển (Git)](#lịch-sử-phát-triển-git)
12. [Lưu ý kỹ thuật](#lưu-ý-kỹ-thuật)
13. [Tác giả & giấy phép](#tác-giả--giấy-phép)

---

## Tổng quan

| Thành phần | Mô tả |
|------------|--------|
| **Dữ liệu** | `dataset.csv` — 4.424 sinh viên, 36 đặc trưng + nhãn `Target` |
| **Notebook** | Phân tích EDA, feature engineering, PCA, K-Means, SMOTE, so sánh 7 thuật toán |
| **Huấn luyện** | `train_and_save.py` — pipeline production: Feature Engineering → Scale → SMOTE → **Voting Ensemble** |
| **Triển khai** | `app.py` — giao diện nhập liệu và dự đoán kèm biểu đồ xác suất |
| **Đầu ra** | `model.pkl` (~46 MB) — ensemble + scaler + label encoder + danh sách cột |

---

## Bài toán & mục tiêu

- **Loại bài toán:** Học máy có giám sát — **Phân loại đa lớp (Multi-class Classification)** với 3 nhãn.
- **Mục tiêu nghiệp vụ:** Nhận diện sớm sinh viên có nguy cơ bỏ học để nhà trường can thiệp kịp thời (hỗ trợ học phí, tư vấn học tập, v.v.).
- **Nhãn đầu ra (`Target`):**

| Nhãn | Ý nghĩa | Tỷ lệ (ước lượng) |
|------|---------|-------------------|
| `Graduate` | Tốt nghiệp | ~49,9% (2.209 mẫu) |
| `Dropout` | Bỏ học | ~32,1% (1.421 mẫu) |
| `Enrolled` | Đang theo học | ~17,1% (794 mẫu) |

Dữ liệu phản ánh bối cảnh giáo dục đại học (thường gắn với bộ dữ liệu công khai *Predict students' dropout and academic success* từ các trường ở Bồ Đào Nha), gồm thông tin hành chính, học tập hai học kỳ đầu và chỉ số kinh tế vĩ mô tại thời điểm nhập học.

---

## Cấu trúc thư mục

```
ML2026/
├── dataset.csv                                          # Bộ dữ liệu gốc (phân tách bằng ;)
├── train_and_save.py                                    # Huấn luyện & lưu model.pkl
├── app.py                                               # Ứng dụng Streamlit
├── model.pkl                                            # Pipeline đã huấn luyện (tạo sau khi train)
├── Predict students' dropout and academic success.ipynb # Notebook phân tích đầy đủ (110 cells)
├── Predict students' dropout and academic success.html  # Bản export HTML của notebook
├── venv/                                                # Virtual environment Python 3.9
└── README.md                                            # Tài liệu này
```

---

## Bộ dữ liệu

### Định dạng file

- **File:** `dataset.csv`
- **Dấu phân cách:** `;` (chấm phẩy)
- **Kích thước:** 4.424 dòng × 37 cột (36 feature + 1 target)
- **Giá trị thiếu:** Không có (đã kiểm tra trong notebook)
- **Trùng lặp:** Không có hàng trùng

### Cột dữ liệu (36 đặc trư đầu vào)

Sau khi đọc CSV, script huấn luyện chuẩn hóa tên cột: thay khoảng trắng bằng `_` (ví dụ `Marital status` → `Marital_status`).

| # | Tên cột (gốc) | Tên sau chuẩn hóa | Kiểu | Mô tả ngắn |
|---|---------------|-------------------|------|------------|
| 1 | Marital status | `Marital_status` | int | Tình trạng hôn nhân (1–6) |
| 2 | Application mode | `Application_mode` | int | Phương thức đăng ký tuyển sinh |
| 3 | Application order | `Application_order` | int | Thứ tự ưu tiên đăng ký (0–9) |
| 4 | Course | `Course` | int | Mã ngành/khóa học |
| 5 | Daytime/evening attendance | `Daytime/evening_attendance\t` | int | 1 = ban ngày, 0 = ban đêm |
| 6 | Previous qualification | `Previous_qualification` | int | Trình độ đầu vào |
| 7 | Previous qualification (grade) | `Previous_qualification_(grade)` | float | Điểm trình độ đầu vào (0–200) |
| 8 | Nacionality | `Nacionality` | int | Mã quốc tịch |
| 9 | Mother's qualification | `Mother's_qualification` | int | Trình độ học vấn của mẹ |
| 10 | Father's qualification | `Father's_qualification` | int | Trình độ học vấn của cha |
| 11 | Mother's occupation | `Mother's_occupation` | int | Nghề nghiệp mẹ |
| 12 | Father's occupation | `Father's_occupation` | int | Nghề nghiệp cha |
| 13 | Admission grade | `Admission_grade` | float | Điểm tuyển sinh (0–200) |
| 14 | Displaced | `Displaced` | int | Di dân (0/1) |
| 15 | Educational special needs | `Educational_special_needs` | int | Nhu cầu giáo dục đặc biệt (0/1) |
| 16 | Debtor | `Debtor` | int | Đang nợ học phí (0/1) |
| 17 | Tuition fees up to date | `Tuition_fees_up_to_date` | int | Học phí đóng đủ (0/1) |
| 18 | Gender | `Gender` | int | Giới tính (1 = nam, 0 = nữ) |
| 19 | Scholarship holder | `Scholarship_holder` | int | Có học bổng (0/1) |
| 20 | Age at enrollment | `Age_at_enrollment` | int | Tuổi khi nhập học |
| 21 | International | `International` | int | Sinh viên quốc tế (0/1) |
| 22–27 | Curricular units 1st sem (...) | `Curricular_units_1st_sem_(...)` | int/float | HK1: miễn, đăng ký, đánh giá, qua, điểm TB, không đánh giá |
| 28–33 | Curricular units 2nd sem (...) | `Curricular_units_2nd_sem_(...)` | int/float | HK2: tương tự HK1 |
| 34 | Unemployment rate | `Unemployment_rate` | float | Tỷ lệ thất nghiệp (%) |
| 35 | Inflation rate | `Inflation_rate` | float | Tỷ lệ lạm phát (%) |
| 36 | GDP | `GDP` | float | Tăng trưởng GDP (%) |
| — | **Target** | `Target` | string | `Dropout` / `Enrolled` / `Graduate` |

### Đặc trư được tạo thêm (Feature Engineering)

Trong `train_and_save.py` (và notebook), thêm **5 cột** từ dữ liệu học kỳ:

| Cột mới | Công thức / ý nghĩa |
|---------|---------------------|
| `pass_rate_1` | Tỷ lệ môn qua HK1 = approved₁ / (enrolled₁ + 1) |
| `pass_rate_2` | Tỷ lệ môn qua HK2 |
| `grade_progress` | Chênh điểm TB HK2 − HK1 |
| `total_approved` | Tổng môn qua hai học kỳ |
| `avg_grade` | Điểm trung bình hai học kỳ |

**Tổng số cột đầu vào khi huấn luyện:** 41 (36 gốc + 5 engineered).

---

## Quy trình xử lý & mô hình

Dự án có **hai pipeline** liên quan: pipeline **nghiên cứu** trong notebook và pipeline **triển khai** trong `train_and_save.py`.

### A. Pipeline production (`train_and_save.py`)

```
Đọc CSV (;)
    → Chuẩn hóa tên cột
    → Feature Engineering (5 cột)
    → LabelEncoder cho Target (Dropout/Enrolled/Graduate → 0/1/2)
    → Train/Test split (80/20, random_state=42)
    → StandardScaler (fit trên train)
    → SMOTE trên tập train (cân bằng lớp thiểu số)
    → VotingClassifier (soft voting)
           ├── XGBoost      (trọng số 3)
           ├── LightGBM     (trọng số 2)
           └── RandomForest (trọng số 1)
    → Đánh giá trên test set
    → Lưu model.pkl
```

#### Tham số các mô hình trong ensemble

**XGBoost (`XGBClassifier`):**
- `n_estimators=400`, `learning_rate=0.05`, `max_depth=7`
- `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=3`
- `gamma=0.1`, `reg_alpha=0.1`, `reg_lambda=1.0`

**LightGBM (`LGBMClassifier`):**
- `n_estimators=400`, `learning_rate=0.05`, `max_depth=7`
- `subsample=0.8`, `colsample_bytree=0.8`, `min_child_samples=20`

**Random Forest (`RandomForestClassifier`):**
- `n_estimators=300`, `max_depth=10`
- `min_samples_split=5`, `min_samples_leaf=2`

**Voting:** `voting="soft"` — lấy trung bình có trọng số xác suất của 3 mô hình.

#### Thời gian huấn luyện

Khoảng **3–5 phút** trên máy thông thường (phụ thuộc CPU/RAM).

### B. Pipeline nghiên cứu (Notebook)

Notebook thực hiện đầy đủ hơn:

1. **EDA:** phân phối, boxplot, heatmap tương quan, phân tích đơn/đa biến theo `Target`
2. **Xử lý ngoại lai (IQR):** trên 4 cột điểm (`Previous_qualification_(grade)`, `Admission_grade`, điểm HK1/HK2)
3. **Chi-square test:** loại 4 đặc trư không liên quan có ý nghĩa thống kê với target
4. **PCA:** giảm chiều xuống 10 thành phần chính
5. **K-Means:** gom 3 cụm, gắn nhãn cụm làm đặc trư bổ sung → ma trận **11 chiều**
6. **SMOTE** trên tập train
7. **So sánh 7 mô hình:** Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost

Kết luận trong notebook: **XGBoost** thường cho accuracy cao nhất trên pipeline PCA + K-Means; production chuyển sang **Voting Ensemble** trên toàn bộ 41 feature (không dùng PCA/K-Means) để tận dụng thông tin đầy đủ và ổn định hơn khi deploy.

---

## Cài đặt môi trường

### Yêu cầu

- **Python:** 3.9+ (dự án dùng `venv` với Python 3.9)
- **Hệ điều hành:** macOS / Linux / Windows
- **RAM:** khuyến nghị ≥ 8 GB khi train (ensemble + SMOTE)

### Cách 1: Dùng virtual environment có sẵn

```bash
cd /đường/dẫn/tới/ML2026
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate      # Windows
```

### Cách 2: Tạo môi trường mới

```bash
cd /đường/dẫn/tới/ML2026
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm streamlit altair
```

### Phiên bản thư viện chính (đã kiểm tra trong `venv`)

| Gói | Phiên bản |
|-----|-----------|
| pandas | 2.3.3 |
| numpy | 2.0.2 |
| scikit-learn | 1.6.1 |
| imbalanced-learn | 0.12.4 |
| xgboost | 2.1.4 |
| lightgbm | 4.6.0 |
| streamlit | 1.50.0 |
| altair | 5.5.0 |

Để xuất đầy đủ danh sách phụ thuộc:

```bash
pip freeze > requirements.txt
```

*(Notebook còn dùng thêm `matplotlib`, `seaborn`, `catboost` khi chạy phân tích — cài thêm nếu cần mở notebook.)*

---

## Hướng dẫn sử dụng

### Bước 1: Huấn luyện mô hình

```bash
python3 train_and_save.py --data dataset.csv --output model.pkl
```

| Tham số | Mặc định | Mô tả |
|---------|----------|--------|
| `--data` | `dataset.csv` | Đường dẫn file CSV |
| `--output` | `model.pkl` | File pickle đầu ra |

Khi chạy thành công, terminal in:
- Số dòng/cột sau feature engineering
- Kích thước tập sau SMOTE
- Accuracy từng mô hình đơn lẻ và **Voting Ensemble**
- `classification_report` (precision, recall, F1 theo lớp)

### Bước 2: Chạy ứng dụng web

```bash
streamlit run app.py
```

Trình duyệt mở (mặc định `http://localhost:8501`). Nếu chưa có `model.pkl`, sidebar hiển thị hướng dẫn chạy lại bước 1.

### Bước 3 (tùy chọn): Mở notebook phân tích

```bash
jupyter notebook "Predict students' dropout and academic success.ipynb"
```

Hoặc mở file `.html` đã export để xem kết quả không cần Jupyter.

---

## Ứng dụng web Streamlit

File: `app.py`

### Chức năng

- Form nhập **đầy đủ** thông tin sinh viên (cá nhân, hành chính, HK1, HK2, kinh tế vĩ mô)
- Dự đoán lớp: **Dropout** / **Enrolled** / **Graduate**
- Hiển thị **xác suất** từng lớp (bảng + biểu đồ cột Altair)
- Expander xem ma trận đặc trư đã gửi vào mô hình
- Cache model với `@st.cache_resource` để load `model.pkl` một lần

### Luồng dự đoán trong app

1. Thu thập input từ form → `DataFrame` 1 dòng
2. Bổ sung cột thiếu (nếu có) bằng `0`
3. `scaler.transform()` trên `feature_cols`
4. `ensemble.predict()` / `predict_proba()`
5. Map mã số → nhãn qua `label_encoder`

### Giao diện

- Layout **wide**, icon 🎓
- Màu theo lớp: đỏ (Dropout), vàng (Enrolled), xanh (Graduate)

> **Lưu ý:** Sidebar có thể ghi *"XGBoost"* và *"PCA + KMeans"* — đó là mô tả từ giai đoạn notebook. Mô hình thực tế trong `model.pkl` hiện tại là **Voting Ensemble (XGBoost + LightGBM + Random Forest)** với **StandardScaler + SMOTE**, không qua PCA/K-Means.

---

## Jupyter Notebook (phân tích & thử nghiệm)

| Thuộc tính | Giá trị |
|------------|---------|
| File | `Predict students' dropout and academic success.ipynb` |
| Số cell | 110 |
| Export HTML | `Predict students' dropout and academic success.html` (~1,7 MB) |

### Nội dung chính theo chương

| Phần | Nội dung |
|------|----------|
| 1 | Thu thập & làm sạch dữ liệu (null, duplicate, đổi tên cột) |
| 2 | EDA — phân bố target, biểu đồ liên tục/rời rạc, nhận định (nợ học phí, giới tính, v.v.) |
| 3 | Feature engineering & encoding target |
| 4 | Xử lý outlier IQR (4 cột điểm) |
| 5 | Chi-square feature selection |
| 6 | PCA (10 components) + K-Means (3 clusters) |
| 7 | Train/test split + SMOTE |
| 8 | Huấn luyện & so sánh 7 classifier |
| 9 | Bảng tổng hợp accuracy + demo Streamlit đơn giản |

### Một số insight từ EDA (trích notebook)

- Sinh viên **không đóng học phí đúng hạn** hoặc **đang nợ** có tỷ lệ Dropout cao hơn.
- **Nam** có xu hướng Dropout cao hơn trong tập dữ liệu.
- Điểm các học kỳ và tỷ lệ môn qua có tương quan mạnh với kết quả cuối (`Graduate` vs `Dropout`).

---

## Mô hình đã lưu (`model.pkl`)

File pickle là dictionary:

```python
{
    "model":         VotingClassifier,   # ensemble đã fit
    "label_encoder": LabelEncoder,       # classes_: Dropout, Enrolled, Graduate
    "feature_cols":  list[str],          # 41 tên cột, đúng thứ tự khi predict
    "scaler":        StandardScaler,     # đã fit trên train (trước SMOTE)
}
```

**Cách load thủ công:**

```python
import pickle
import pandas as pd

with open("model.pkl", "rb") as f:
    pipeline = pickle.load(f)

model = pipeline["model"]
scaler = pipeline["scaler"]
cols = pipeline["feature_cols"]
le = pipeline["label_encoder"]

# X_new: DataFrame 1 dòng, đủ 41 cột (hoặc bổ sung cột thiếu = 0)
X_scaled = scaler.transform(X_new[cols].values)
pred = model.predict(X_scaled)
label = le.classes_[pred[0]]
proba = model.predict_proba(X_scaled)[0]
```

**Kích thước file:** ~46 MB (do ensemble 3 mô hình lớn).

---

## Lịch sử phát triển (Git)

| Commit | Mô tả ngắn |
|--------|------------|
| `96020e0` | Basic Done — khung dự án cơ bản |
| `81c3b6a` | Add feature columns — feature engineering |
| `8fbac6f` | Tuning Parameter — tinh chỉnh hyperparameter |
| `6b5fdd4` | Compare Models — so sánh nhiều mô hình |

Remote: `origin` → `https://github.com/Minhwritecode/ML-EduPredict.git`

---

## Lưu ý kỹ thuật

1. **Không commit `venv/` hoặc `model.pkl` lên Git** nếu repo công khai (file model rất nặng). Người clone repo cần tự chạy `train_and_save.py`.
2. **Cột `Daytime/evening_attendance\t`:** tên cột trong CSV/model có ký tự tab `\t` do định dạng gốc — `app.py` và pipeline dùng đúng tên này khi predict.
3. **SMOTE chỉ áp dụng khi train**, không dùng khi predict trên web.
4. **Input web** là giá trị số (mã danh mục), không phải text — người dùng cần biết mã ngành, mã nghề nghiệp, v.v. hoặc dùng giá trị mặc định trên form.
5. **Đánh giá mô hình:** accuracy trên test set in ra khi train; nên xem thêm precision/recall cho lớp `Enrolled` (thiểu số) vì accuracy tổng có thể che bias.
6. **Notebook vs Production:** kết quả số liệu giữa notebook (PCA+KMeans, 7 models) và `train_and_save.py` (41 features, ensemble) **không hoàn toàn giống nhau** — dùng `model.pkl` từ script production cho `app.py`.

---

## Tác giả & giấy phép

- **GitHub:** [Minhwritecode/ML-EduPredict](https://github.com/Minhwritecode/ML-EduPredict)
- Dự án phục vụ mục đích học tập / đồ án môn Học máy (ML2026).

Nếu sử dụng bộ dữ liệu gốc trong công trình khoa học, nên trích dẫn nguồn dataset *Predict students' dropout and academic success* (UCI / tác giả gốc) theo quy định của nhà cung cấp dữ liệu.

---

## Tóm tắt lệnh nhanh

```bash
# Clone (nếu chưa có)
git clone https://github.com/Minhwritecode/ML-EduPredict.git
cd ML-EduPredict

# Môi trường
python3 -m venv venv && source venv/bin/activate
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm streamlit altair

# Train
python3 train_and_save.py --data dataset.csv

# Chạy web
streamlit run app.py
```
