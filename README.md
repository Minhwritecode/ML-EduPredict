# ML-EduPredict — Dự đoán Bỏ học & Kết quả Học tập Sinh viên

Dự án **Học máy (Machine Learning)** phân loại đa lớp nhằm dự đoán kết quả học tập của sinh viên đại học: **Bỏ học (Dropout)**, **Đang học (Enrolled)** hoặc **Tốt nghiệp (Graduate)**. Hệ thống gồm quy trình phân tích dữ liệu (Jupyter Notebook), script huấn luyện mô hình production và **ứng dụng web Streamlit** đầy đủ tính năng demo / triển khai.

**Repository:** [https://github.com/Minhwritecode/ML-EduPredict](https://github.com/Minhwritecode/ML-EduPredict)

---

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [Bài toán & mục tiêu](#bài-toán--mục-tiêu)
3. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
4. [Bộ dữ liệu](#bộ-dữ-liệu)
5. [Quy trình xử lý & mô hình](#quy-trình-xử-lý--mô-hình)
6. [Cài đặt môi trường](#cài-đặt-môi-trường)
7. [Hướng dẫn sử dụng nhanh](#hướng-dẫn-sử-dụng-nhanh)
8. [Ứng dụng web Streamlit (`app.py`)](#ứng-dụng-web-streamlit-apppy)
9. [Kịch bản demo (`Demo Scenario.text`)](#kịch-bản-demo-demo-scenariotext)
10. [Jupyter Notebook](#jupyter-notebook-phân-tích--thử-nghiệm)
11. [Mô hình đã lưu (`model.pkl`)](#mô-hình-đã-lưu-modelpkl)
12. [Lưu ý kỹ thuật & xử lý sự cố](#lưu-ý-kỹ-thuật--xử-lý-sự-cố)
13. [Tác giả](#tác-giả)

---

## Tổng quan

| Thành phần | Mô tả |
|------------|--------|
| **Dữ liệu** | `dataset.csv` — 4.424 sinh viên, 36 đặc trưng gốc + nhãn `Target` |
| **Notebook** | EDA, feature engineering, PCA, K-Means, SMOTE, so sánh 7 thuật toán |
| **Huấn luyện** | `train_and_save.py` — Scale → SMOTE → **Voting Ensemble** (XGB + LGBM + RF) |
| **Triển khai** | `app.py` — Streamlit: dự đoán, dashboard, lịch sử, báo cáo HTML |
| **Đầu ra** | `model.pkl` (~46 MB) — ensemble + scaler + label encoder + `feature_cols` |
| **Demo** | `Demo Scenario.text` — kịch bản trình bày 5 ca + timeline 12/25 phút |

---

## Bài toán & mục tiêu

- **Loại bài toán:** Phân loại đa lớp (3 nhãn).
- **Mục tiêu nghiệp vụ:** Nhận diện sớm sinh viên có nguy cơ bỏ học để can thiệp (học phí, cố vấn, tutoring).

| Nhãn | Ý nghĩa | Tỷ lệ (dataset) |
|------|---------|-----------------|
| `Graduate` | Tốt nghiệp | ~49,9% |
| `Dropout` | Bỏ học | ~32,1% |
| `Enrolled` | Đang theo học | ~17,1% |

Nguồn dữ liệu: bộ công khai *Predict students' dropout and academic success* (bối cảnh giáo dục đại học Bồ Đào Nha).

---

## Cấu trúc thư mục

```
ML2026/
├── dataset.csv                    # Dữ liệu gốc (sep ;)
├── train_and_save.py              # Huấn luyện → model.pkl
├── app.py                         # Ứng dụng Streamlit (UI + predict)
├── model.pkl                      # Pipeline đã train (không commit Git)
├── prediction_history.json        # Lịch sử dự đoán local (tự tạo, không commit)
├── Demo Scenario.text             # Kịch bản demo chi tiết cho hội đồng
├── requirements.txt               # Dependencies Python
├── .gitignore                     # venv/, model.pkl, prediction_history.json, ...
├── README.md                      # Tài liệu này
├── Predict students' dropout and academic success.ipynb
├── Predict students' dropout and academic success.html
└── venv/                          # Virtual environment (không commit)
```

---

## Bộ dữ liệu

| Thuộc tính | Giá trị |
|------------|---------|
| File | `dataset.csv` |
| Phân cách | `;` |
| Kích thước | 4.424 × 37 (36 feature + `Target`) |
| Missing / duplicate | Không |

Sau khi đọc, script chuẩn hóa tên cột: khoảng trắng → `_` (ví dụ `Marital status` → `Marital_status`).

### Nhóm đặc trư (36 cột gốc)

| Nhóm | Ví dụ cột |
|------|-----------|
| Hành chính / tuyển sinh | `Marital_status`, `Application_mode`, `Course`, `Admission_grade` |
| Gia đình / xã hội | `Mother's_qualification`, `Father's_occupation`, … |
| Sinh viên | `Debtor`, `Tuition_fees_up_to_date`, `Gender`, `Age_at_enrollment`, … |
| Học kỳ 1 & 2 | `Curricular_units_1st_sem_(enrolled/approved/grade/…)` |
| Kinh tế vĩ mô | `Unemployment_rate`, `Inflation_rate`, `GDP` |

### Feature engineering (bắt buộc khi predict)

| Cột mới | Công thức |
|---------|-----------|
| `pass_rate_1` | `approved_1 / (enrolled_1 + 1)` |
| `pass_rate_2` | `approved_2 / (enrolled_2 + 1)` |
| `grade_progress` | `grade_2 − grade_1` |
| `total_approved` | `approved_1 + approved_2` |
| `avg_grade` | `(grade_1 + grade_2) / 2` |

**Tổng cột đầu vào mô hình:** 41 (36 + 5).  
`app.py` **phải** tính 5 cột này trước `scaler.transform` — nếu thiếu, mọi ca có thể bị dự đoán sai (ví dụ luôn Dropout).

---

## Quy trình xử lý & mô hình

### Pipeline production (`train_and_save.py`)

```
dataset.csv (;)
  → Chuẩn hóa tên cột
  → Feature Engineering (5 cột)
  → LabelEncoder(Target)
  → Train/Test 80/20 (random_state=42)
  → StandardScaler (fit train)
  → SMOTE (chỉ train)
  → VotingClassifier (soft, weights [3,2,1])
        ├── XGBoost
        ├── LightGBM
        └── RandomForest
  → Đánh giá test + lưu model.pkl
```

**Soft voting:** trung bình có trọng số xác suất 3 mô hình con.

**Thời gian train:** khoảng 3–5 phút (CPU).

### Pipeline nghiên cứu (Notebook)

EDA → IQR outlier → Chi-square → **PCA (10)** + **K-Means (3)** → SMOTE → so sánh 7 classifier.  
Kết luận notebook: XGBoost mạnh trên pipeline PCA; **production** dùng full 41 feature + ensemble để deploy ổn định hơn.

---

## Cài đặt môi trường

**Yêu cầu:** Python 3.9+, RAM ≥ 8 GB khi train.

```bash
cd /đường/dẫn/ML2026
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Hoặc cài tay:

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm streamlit altair
```

*(Notebook thêm: `matplotlib`, `seaborn`, `jupyter`, `catboost` nếu cần.)*

---

## Hướng dẫn sử dụng nhanh

```bash
# 1. Huấn luyện (bắt buộc trước khi chạy web)
python3 train_and_save.py --data dataset.csv --output model.pkl

# 2. Chạy ứng dụng
streamlit run app.py
# → http://localhost:8501
```

| Lệnh train | Mặc định | Mô tả |
|--------------|----------|--------|
| `--data` | `dataset.csv` | File CSV |
| `--output` | `model.pkl` | File pickle |

---

## Ứng dụng web Streamlit (`app.py`)

### Ba trang (sidebar)

| Trang | Chức năng |
|-------|-----------|
| **🔮 Dự đoán** | Form nhập liệu, preset demo, kết quả + biểu đồ |
| **📊 Dashboard** | KPI phiên, % Dropout/Enrolled/Graduate, biểu đồ, top 3 nguy cơ |
| **📜 Lịch sử** | Timeline thẻ, bảng, so sánh 2 SV, xuất CSV, xóa lịch sử |

### Tính năng UI (theo mức)

**Mức cơ bản**

- Hero banner, layout wide
- **5 ca demo** — dropdown *"Chọn ca demo nhanh"* (điền form tự động, không cần gõ từng ô)
- Form **một khối** + một nút submit (không chia tab trong form → tránh lỗi submit Streamlit)
- Chỉ số trực tiếp: tỷ lệ qua HK1/HK2, điểm TB, tuổi
- Kết quả màu theo lớp, gauge nguy cơ Dropout, donut/cột xác suất
- Balloons (Graduate), toast (Dropout > 60%)

**Mức 2 — “Pro”**

| # | Tính năng |
|---|-----------|
| 6 | **Giải thích dự đoán** — lý do từ form + xác suất model |
| 7 | **Dashboard** — tổng quan phiên demo |
| 8 | **Step indicator** — 4 bước: Hành chính → Cá nhân → Học tập → Kinh tế |
| 9 | **Tooltip `help=`** — mọi trường form |
| 10 | **Dark / Light mode** — toggle sidebar |
| 11 | **Footer** — `ML-EduPredict · ML2026 · Dataset 4.424 mẫu` |
| 12 | **Empty state Lịch sử** + nút *Thử ca demo Minh Anh* |

**Mức 3 — Demo / đồ án nâng cao**

| # | Tính năng |
|---|-----------|
| 13 | **So sánh 2 sinh viên** (A vs B) — bar chart xác suất trong Lịch sử |
| 14 | **Timeline dạng thẻ** — viền màu theo lớp |
| 15 | **Heatmap mini HK1 vs HK2** — điểm TB + tỷ lệ qua môn |
| 16 | **Tải báo cáo HTML** — sau mỗi lần dự đoán |
| 17 | **`prediction_history.json`** — lịch sử giữ qua F5 (local, không backend) |

### 5 ca demo có sẵn (dropdown)

| Preset | Kết quả kỳ vọng | Dropout % (tham chiếu) |
|--------|-----------------|-------------------------|
| ⭐ Minh Anh (Tốt nghiệp) | **Graduate** | ~6,8% |
| ⚠️ Văn Bình (Bỏ học sớm) | **Dropout** | ~84,7% |
| 🔴 Thu Hà (Thất bại học thuật) | **Dropout** | ~99,5% |
| 📚 Lan Phương (Đang học) | **Enrolled** | ~28,4% |
| 🚨 Hoàng Nam (Cảnh báo sớm) | **Dropout** | ~97,4% |

*Xác suất có thể lệch ±1–2% nếu train lại `model.pkl`.*

**Cách demo nhanh:** chọn preset → **🔮 Dự đoán kết quả** → đọc khung kết quả + phần *Vì sao mô hình dự đoán như vậy?*

### Luồng predict trong `app.py` (quan trọng)

```
Form submit
  → DataFrame 1 dòng (36 cột gốc)
  → Feature engineering (5 cột)     ← BẮT BUỘC
  → Bổ sung cột thiếu = 0
  → scaler.transform(feature_cols)
  → model.predict / predict_proba
  → label_encoder → Dropout | Enrolled | Graduate
  → Lưu prediction_history.json + session_state.history
```

```python
# Trích logic cốt lõi (app.py)
df["pass_rate_1"] = df["Curricular_units_1st_sem_(approved)"] / (df["Curricular_units_1st_sem_(enrolled)"] + 1)
df["pass_rate_2"] = df["Curricular_units_2nd_sem_(approved)"] / (df["Curricular_units_2nd_sem_(enrolled)"] + 1)
df["grade_progress"] = df["Curricular_units_2nd_sem_(grade)"] - df["Curricular_units_1st_sem_(grade)"]
df["total_approved"] = df["Curricular_units_1st_sem_(approved)"] + df["Curricular_units_2nd_sem_(approved)"]
df["avg_grade"] = (df["Curricular_units_1st_sem_(grade)"] + df["Curricular_units_2nd_sem_(grade)"]) / 2
X_scaled = scaler.transform(df[feature_cols].values)
pred = model.predict(X_scaled)
```

> **Sidebar:** hiển thị `Scale+Ensemble` khi `model.pkl` không chứa PCA/KMeans. Notebook dùng PCA+KMeans; **app production** dùng ensemble trên 41 feature.

### Gợi ý so sánh trong Lịch sử

- Chọn **Minh Anh** + **Lan Phương** → minh họa cùng đăng ký 6 môn nhưng khác tỷ lệ qua / điểm → Graduate vs Enrolled.
- Chọn **Văn Bình** + **Thu Hà** → hai kiểu Dropout (0 tín chỉ vs đăng ký nhưng qua 0 môn).

---

## Kịch bản demo (`Demo Scenario.text`)

File text đi kèm repo — **hướng dẫn trình bày** cho giảng viên / hội đồng:

- Checklist chuẩn bị (`model.pkl`, `streamlit run app.py`)
- Timeline **12 phút** (rút gọn) và **25 phút** (đầy đủ)
- 5 ca với bối cảnh, lời dẫn, xác suất tham chiếu
- Luồng mới: preset dropdown, Dashboard, Lịch sử, so sánh A vs B
- Q&A, xử lý sự cố, phụ lục ánh xạ Form ↔ Dataset

**Mở file:** `Demo Scenario.text` trong editor hoặc in PDF.

---

## Jupyter Notebook (phân tích & thử nghiệm)

| File | Mô tả |
|------|--------|
| `.ipynb` | ~110 cells — EDA đến so sánh 7 model |
| `.html` | Bản export xem không cần Jupyter |

Insight EDA (trích): nợ học phí / chưa đóng đủ ↔ Dropout cao; điểm và tỷ lệ qua môn ↔ Graduate.

---

## Mô hình đã lưu (`model.pkl`)

```python
{
    "model":         VotingClassifier,   # đã fit
    "label_encoder": LabelEncoder,       # classes_: Dropout, Enrolled, Graduate
    "feature_cols":  list[str],          # 41 cột, đúng thứ tự
    "scaler":        StandardScaler,
}
```

**Load & predict thủ công:**

```python
import pickle
import pandas as pd

with open("model.pkl", "rb") as f:
    p = pickle.load(f)

# df: 1 dòng, đủ 41 cột SAU feature engineering
X = p["scaler"].transform(df[p["feature_cols"]].values)
label = p["label_encoder"].classes_[p["model"].predict(X)[0]]
proba = p["model"].predict_proba(X)[0]
```

**Không commit** `model.pkl` lên Git (file lớn). Clone repo → tự chạy `train_and_save.py`.

---

## Lưu ý kỹ thuật & xử lý sự cố

| Vấn đề | Nguyên nhân / cách xử lý |
|--------|---------------------------|
| Mọi ca đều **Dropout** | Thiếu 5 cột feature engineering trước predict — đã fix trong `app.py` |
| Sidebar ❌ thiếu model | Chạy `python3 train_and_save.py` |
| Xác suất lệch vài % so tài liệu | Train lại model; thứ hạng lớp vẫn đúng |
| Cột `Daytime/evening_attendance\t` | Tên có ký tự tab — giữ nguyên khi map từ form |
| SMOTE | Chỉ khi train, không dùng lúc predict |
| `prediction_history.json` | Tự tạo khi dự đoán; xóa bằng nút trong Lịch sử |
| Notebook ≠ Production | Dùng `model.pkl` từ `train_and_save.py` cho `app.py` |

### Git — không đưa lên remote

- `venv/`
- `model.pkl`
- `prediction_history.json`
- `.env`

---

## Tác giả

- **GitHub:** [Minhwritecode/ML-EduPredict](https://github.com/Minhwritecode/ML-EduPredict)
- Dự án học tập / đồ án **ML2026**.

Trích dẫn dataset gốc *Predict students' dropout and academic success* khi dùng trong báo cáo khoa học.

---

## Tóm tắt lệnh

```bash
git clone https://github.com/Minhwritecode/ML-EduPredict.git
cd ML-EduPredict
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 train_and_save.py --data dataset.csv
streamlit run app.py
```
