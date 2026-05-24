"""
app.py — Streamlit Web App: Dự đoán Bỏ học Sinh viên
Chạy: streamlit run app.py
"""

import json
import os
import pickle
import uuid
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

HISTORY_FILE = "prediction_history.json"
MINH_ANH_PRESET = "⭐ Minh Anh (Tốt nghiệp)"

st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Model (model.pkl) ─────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline(path="model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


MODEL_LOADED = False
pca = kmeans = None
USE_PCA_PIPELINE = False

try:
    pipeline = load_pipeline()
    model = pipeline["model"]
    label_encoder = pipeline["label_encoder"]
    feature_cols = pipeline["feature_cols"]
    scaler = pipeline["scaler"]
    pca = pipeline.get("pca")
    kmeans = pipeline.get("kmeans")
    USE_PCA_PIPELINE = pca is not None and kmeans is not None
    MODEL_LOADED = True
except FileNotFoundError:
    pipeline = None

CLASS_INFO = {
    "Dropout":  {"emoji": "⚠️", "color": "#FF4B4B", "bg": "#FFF0F0"},
    "Enrolled": {"emoji": "📚", "color": "#F59E0B", "bg": "#FFFBEB"},
    "Graduate": {"emoji": "🎓", "color": "#10B981", "bg": "#F0FFF4"},
}
CLASS_COLORS = {"Dropout": "#FF4B4B", "Enrolled": "#F59E0B", "Graduate": "#10B981"}

DEMO_PRESET_OPTIONS = [
    "— Tự nhập —",
    MINH_ANH_PRESET,
    "⚠️ Văn Bình (Bỏ học sớm)",
    "🔴 Thu Hà (Thất bại học thuật)",
    "📚 Lan Phương (Đang học)",
    "🚨 Hoàng Nam (Cảnh báo sớm)",
]

DEMO_PRESETS = {
    MINH_ANH_PRESET: {
        "student_name": "Minh Anh",
        "marital_status": 1, "application_mode": 15, "application_order": 1, "course": 9254,
        "daytime": 1, "prev_qual": 1, "prev_qual_grade": 160.0, "nationality": 1,
        "mothers_qual": 1, "fathers_qual": 3, "mothers_occ": 3, "fathers_occ": 3,
        "displaced": 1, "special_needs": 0, "debtor": 0, "tuition_ok": 0,
        "gender": 1, "scholarship": 0, "age": 19, "international": 0, "admission_grade": 142.5,
        "cu1_credited": 0, "cu1_enrolled": 6, "cu1_evals": 6, "cu1_approved": 6,
        "cu1_grade": 14.0, "cu1_no_eval": 0,
        "cu2_credited": 0, "cu2_enrolled": 6, "cu2_evals": 6, "cu2_approved": 6,
        "cu2_grade": 13.7, "cu2_no_eval": 0,
        "unemployment": 13.9, "inflation": -0.3, "gdp": 0.79,
    },
    "⚠️ Văn Bình (Bỏ học sớm)": {
        "student_name": "Văn Bình",
        "marital_status": 1, "application_mode": 17, "application_order": 5, "course": 171,
        "daytime": 1, "prev_qual": 1, "prev_qual_grade": 122.0, "nationality": 1,
        "mothers_qual": 19, "fathers_qual": 12, "mothers_occ": 5, "fathers_occ": 9,
        "displaced": 1, "special_needs": 0, "debtor": 0, "tuition_ok": 1,
        "gender": 1, "scholarship": 0, "age": 20, "international": 0, "admission_grade": 127.3,
        "cu1_credited": 0, "cu1_enrolled": 0, "cu1_evals": 0, "cu1_approved": 0,
        "cu1_grade": 0.0, "cu1_no_eval": 0,
        "cu2_credited": 0, "cu2_enrolled": 0, "cu2_evals": 0, "cu2_approved": 0,
        "cu2_grade": 0.0, "cu2_no_eval": 0,
        "unemployment": 10.8, "inflation": 1.4, "gdp": 1.74,
    },
    "🔴 Thu Hà (Thất bại học thuật)": {
        "student_name": "Thu Hà",
        "marital_status": 1, "application_mode": 1, "application_order": 5, "course": 9070,
        "daytime": 1, "prev_qual": 1, "prev_qual_grade": 122.0, "nationality": 1,
        "mothers_qual": 29, "fathers_qual": 29, "mothers_occ": 9, "fathers_occ": 9,
        "displaced": 0, "special_needs": 0, "debtor": 0, "tuition_ok": 0,
        "gender": 1, "scholarship": 0, "age": 19, "international": 0, "admission_grade": 124.8,
        "cu1_credited": 0, "cu1_enrolled": 6, "cu1_evals": 0, "cu1_approved": 0,
        "cu1_grade": 0.0, "cu1_no_eval": 0,
        "cu2_credited": 0, "cu2_enrolled": 6, "cu2_evals": 0, "cu2_approved": 0,
        "cu2_grade": 0.0, "cu2_no_eval": 0,
        "unemployment": 10.8, "inflation": 1.4, "gdp": 1.74,
    },
    "📚 Lan Phương (Đang học)": {
        "student_name": "Lan Phương",
        "marital_status": 1, "application_mode": 18, "application_order": 1, "course": 9238,
        "daytime": 1, "prev_qual": 1, "prev_qual_grade": 137.0, "nationality": 1,
        "mothers_qual": 19, "fathers_qual": 29, "mothers_occ": 5, "fathers_occ": 8,
        "displaced": 0, "special_needs": 0, "debtor": 0, "tuition_ok": 1,
        "gender": 0, "scholarship": 0, "age": 18, "international": 0, "admission_grade": 137.4,
        "cu1_credited": 0, "cu1_enrolled": 6, "cu1_evals": 10, "cu1_approved": 1,
        "cu1_grade": 12.0, "cu1_no_eval": 0,
        "cu2_credited": 0, "cu2_enrolled": 6, "cu2_evals": 14, "cu2_approved": 2,
        "cu2_grade": 11.0, "cu2_no_eval": 0,
        "unemployment": 10.8, "inflation": 1.4, "gdp": 1.74,
    },
    "🚨 Hoàng Nam (Cảnh báo sớm)": {
        "student_name": "Hoàng Nam",
        "marital_status": 1, "application_mode": 1, "application_order": 1, "course": 9500,
        "daytime": 1, "prev_qual": 1, "prev_qual_grade": 120.0, "nationality": 1,
        "mothers_qual": 19, "fathers_qual": 29, "mothers_occ": 5, "fathers_occ": 8,
        "displaced": 0, "special_needs": 0, "debtor": 1, "tuition_ok": 0,
        "gender": 1, "scholarship": 0, "age": 22, "international": 0, "admission_grade": 115.0,
        "cu1_credited": 0, "cu1_enrolled": 6, "cu1_evals": 8, "cu1_approved": 3,
        "cu1_grade": 9.5, "cu1_no_eval": 1,
        "cu2_credited": 0, "cu2_enrolled": 6, "cu2_evals": 8, "cu2_approved": 2,
        "cu2_grade": 8.0, "cu2_no_eval": 2,
        "unemployment": 15.5, "inflation": 2.8, "gdp": -1.7,
    },
}

FORM_DEFAULTS = {
    "student_name": "Sinh viên mới",
    "marital_status": 1, "application_mode": 1, "application_order": 1, "course": 9500,
    "daytime": 1, "prev_qual": 1, "prev_qual_grade": 130.0, "nationality": 1,
    "mothers_qual": 19, "fathers_qual": 29, "mothers_occ": 5, "fathers_occ": 8,
    "displaced": 0, "special_needs": 0, "debtor": 0, "tuition_ok": 1,
    "gender": 1, "scholarship": 0, "age": 19, "international": 0, "admission_grade": 130.0,
    "cu1_credited": 0, "cu1_enrolled": 6, "cu1_evals": 6, "cu1_approved": 5,
    "cu1_grade": 12.0, "cu1_no_eval": 0,
    "cu2_credited": 0, "cu2_enrolled": 6, "cu2_evals": 6, "cu2_approved": 5,
    "cu2_grade": 12.0, "cu2_no_eval": 0,
    "unemployment": 10.8, "inflation": 1.4, "gdp": 1.74,
}

FIELD_HELP = {
    "marital_status": "Tình trạng hôn nhân khi nhập học (mã theo dataset).",
    "application_mode": "Kênh / hình thức đăng ký tuyển sinh.",
    "application_order": "Thứ tự ưu tiên nguyện vọng (0–9).",
    "course": "Mã ngành đào tạo (Course ID trong dataset).",
    "daytime": "1 = Ban ngày, 0 = Ban đêm.",
    "prev_qual": "Trình độ văn bằng trước khi vào đại học.",
    "prev_qual_grade": "Điểm trung bình bằng cấp trước (0–200).",
    "nationality": "Mã quốc tịch (1 = Bồ Đào Nha trong dataset gốc).",
    "mothers_qual": "Trình độ học vấn của mẹ.",
    "fathers_qual": "Trình độ học vấn của cha.",
    "mothers_occ": "Mã nghề nghiệp mẹ.",
    "fathers_occ": "Mã nghề nghiệp cha.",
    "displaced": "Sinh viên di dời nơi cư trú để học.",
    "special_needs": "Có nhu cầu giáo dục đặc biệt hay không.",
    "debtor": "Sinh viên đang nợ học phí (1 = Có).",
    "tuition_ok": "Đã đóng đủ học phí (1 = Có).",
    "gender": "1 = Nam, 0 = Nữ.",
    "scholarship": "Đang nhận học bổng (1 = Có).",
    "age": "Tuổi tại thời điểm nhập học.",
    "international": "Sinh viên quốc tế (1 = Có).",
    "admission_grade": "Điểm xét tuyển đầu vào (0–200).",
    "cu1_credited": "Số môn được công nhận tín chỉ HK1.",
    "cu1_enrolled": "Tổng môn đăng ký học kỳ 1.",
    "cu1_evals": "Số lần đánh giá / thi HK1.",
    "cu1_approved": "Số môn đạt yêu cầu tín chỉ học kỳ 1.",
    "cu1_grade": "Điểm trung bình các môn HK1 (thang 0–20).",
    "cu1_no_eval": "Môn HK1 chưa được đánh giá.",
    "cu2_credited": "Số môn được công nhận tín chỉ HK2.",
    "cu2_enrolled": "Tổng môn đăng ký học kỳ 2.",
    "cu2_evals": "Số lần đánh giá / thi HK2.",
    "cu2_approved": "Số môn đạt yêu cầu tín chỉ học kỳ 2.",
    "cu2_grade": "Điểm trung bình các môn HK2 (thang 0–20).",
    "cu2_no_eval": "Môn HK2 chưa được đánh giá.",
    "unemployment": "Tỷ lệ thất nghiệp vùng tại thời điểm nhập học (%).",
    "inflation": "Tỷ lệ lạm phát (%).",
    "gdp": "Tăng trưởng GDP (%).",
}


# ── Session init ──────────────────────────────────────────────────────────────
def load_history_from_disk():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []
    return []


def persist_history_to_disk():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


if "history" not in st.session_state:
    st.session_state.history = load_history_from_disk()
if "page" not in st.session_state:
    st.session_state.page = "predict"
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "_last_demo_preset" not in st.session_state:
    st.session_state._last_demo_preset = DEMO_PRESET_OPTIONS[0]
if "_do_reset" not in st.session_state:
    st.session_state._do_reset = False
if "_apply_minh_anh" not in st.session_state:
    st.session_state._apply_minh_anh = False
for _k, _v in FORM_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── UI helpers ────────────────────────────────────────────────────────────────
def inject_theme_css(theme: str) -> None:
    dark = theme == "dark"
    bg = "#0f172a" if dark else "#f8fafc"
    card = "#1e293b" if dark else "#ffffff"
    text = "#e2e8f0" if dark else "#1e293b"
    muted = "#94a3b8" if dark else "#64748b"
    border = "#334155" if dark else "#e2e8f0"
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
.stApp {{ background: {bg}; color: {text}; }}
.hero {{
    background: linear-gradient(120deg, #1e3a5f 0%, #2563eb 50%, #7c3aed 100%);
    border-radius: 14px; padding: 22px 30px; margin-bottom: 1.2rem;
    color: #fff; box-shadow: 0 6px 24px rgba(37,99,235,.22);
}}
.hero h1 {{ margin: 0; font-size: 1.7rem; color: #fff !important; }}
.hero p  {{ margin: 6px 0 0; opacity: .9; color: #e0e7ff !important; }}
div[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}}
.live-metric {{
    background: {card}; border: 1px solid {border}; border-radius: 12px;
    padding: 14px 18px; text-align: center;
}}
.live-metric .val {{ font-size: 1.5rem; font-weight: 700; color: {text}; }}
.live-metric .lbl {{ font-size: .75rem; color: {muted}; text-transform: uppercase; }}
.tip-card, .reason-card {{
    background: {"#1e3a5f" if dark else "#eff6ff"};
    border: 1px solid {"#3b82f6" if dark else "#bfdbfe"};
    border-radius: 12px; padding: 18px 24px; margin: 12px 0;
}}
.tip-card h4, .reason-card h4 {{ margin: 0 0 10px; color: {"#93c5fd" if dark else "#1d4ed8"}; }}
.steps {{
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
    margin: 12px 0 20px; padding: 12px 16px;
    background: {card}; border: 1px solid {border}; border-radius: 12px;
}}
.step {{
    padding: 6px 14px; border-radius: 20px; font-size: .85rem; font-weight: 600;
    background: {"#334155" if dark else "#e2e8f0"}; color: {muted};
}}
.step.active {{
    background: linear-gradient(90deg, #2563eb, #7c3aed); color: #fff;
}}
.hist-card {{
    border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
    border-left: 5px solid; background: {card};
}}
.sem-box {{
    border-radius: 10px; padding: 16px; text-align: center;
    border: 1px solid {border}; background: {card};
}}
.app-footer {{
    text-align: center; padding: 20px; margin-top: 40px;
    color: {muted}; font-size: .85rem; border-top: 1px solid {border};
}}
.empty-state {{
    text-align: center; padding: 40px 24px;
    background: {card}; border-radius: 16px; border: 2px dashed {border};
}}
</style>
""",
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        '<div class="app-footer">ML-EduPredict · ML2026 · Dataset 4.424 mẫu · Powered by model.pkl</div>',
        unsafe_allow_html=True,
    )


def render_step_indicator():
    st.markdown("""
<div class="steps">
  <span class="step active">1 · Hành chính</span>
  <span>→</span>
  <span class="step active">2 · Cá nhân</span>
  <span>→</span>
  <span class="step active">3 · Học tập</span>
  <span>→</span>
  <span class="step active">4 · Kinh tế</span>
</div>
<p style="font-size:.8rem;color:#64748b;margin:-8px 0 12px 0;">
Form một khối — cuộn xuống để điền đủ 4 nhóm thông tin.</p>
""", unsafe_allow_html=True)


def apply_demo_preset(preset_name: str) -> None:
    data = DEMO_PRESETS.get(preset_name)
    if data:
        for key, val in data.items():
            st.session_state[key] = val


def save_to_history(entry: dict) -> None:
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:100]
    persist_history_to_disk()


def get_risk_level(dropout_pct: float) -> tuple[str, str]:
    if dropout_pct < 30:
        return "Thấp", "#10B981"
    if dropout_pct <= 60:
        return "Trung bình", "#F59E0B"
    return "Cao", "#EF4444"


def explain_prediction(
    pred_label: str,
    dropout_pct: float,
    debtor: int,
    tuition_ok: int,
    avg_grade: float,
    cu1_e: int,
    cu1_a: int,
    cu2_e: int,
    cu2_a: int,
    cu1_g: float,
    cu2_g: float,
    scholarship: int,
) -> list[str]:
    reasons = []
    pr1 = cu1_a / (cu1_e + 1) * 100 if cu1_e >= 0 else 0
    pr2 = cu2_a / (cu2_e + 1) * 100 if cu2_e >= 0 else 0

    if cu1_e == 0 and cu2_e == 0:
        reasons.append("Không đăng ký môn học ở cả hai học kỳ đầu — tín hiệu bỏ học sớm mạnh.")
    if cu1_e > 0 and cu1_a == 0:
        reasons.append("Học kỳ 1 có đăng ký nhưng không qua môn nào.")
    if cu2_e > 0 and cu2_a == 0:
        reasons.append("Học kỳ 2 có đăng ký nhưng không qua môn nào.")
    if avg_grade < 10:
        reasons.append(f"Điểm trung bình hai kỳ thấp ({avg_grade:.1f}/20).")
    elif avg_grade >= 13:
        reasons.append(f"Điểm trung bình tốt ({avg_grade:.1f}/20) — ủng hộ hướng tốt nghiệp.")
    if pr1 < 50 or pr2 < 50:
        reasons.append(f"Tỷ lệ qua môn thấp (HK1: {pr1:.0f}%, HK2: {pr2:.0f}%).")
    elif pr1 >= 85 and pr2 >= 85:
        reasons.append(f"Tỷ lệ qua môn cao (HK1: {pr1:.0f}%, HK2: {pr2:.0f}%).")
    if debtor:
        reasons.append("Đang nợ học phí — yếu tố tăng nguy cơ Dropout trong dữ liệu huấn luyện.")
    if not tuition_ok:
        reasons.append("Chưa đóng đủ học phí đúng hạn.")
    if scholarship:
        reasons.append("Có học bổng — thường liên quan kết quả học tập ổn định hơn.")
    if cu2_g < cu1_g - 1:
        reasons.append("Điểm HK2 giảm so với HK1 — xu hướng học tập đi xuống.")
    elif cu2_g > cu1_g + 0.5:
        reasons.append("Điểm HK2 cải thiện so với HK1.")

    if pred_label == "Dropout":
        reasons.append(f"Mô hình ưu tiên **Dropout** với xác suất {dropout_pct:.1f}%.")
    elif pred_label == "Graduate":
        reasons.append(f"Mô hình ưu tiên **Graduate** — xác suất Dropout chỉ {dropout_pct:.1f}%.")
    else:
        reasons.append("Kết quả **Enrolled**: đang học nhưng chưa đủ tín hiệu tốt nghiệp hoặc bỏ học.")

    return reasons[:8]


def render_prediction_reasons(reasons: list[str]) -> None:
    items = "".join(f"<li>{r}</li>" for r in reasons)
    st.markdown(
        f'<div class="reason-card"><h4>🔎 Vì sao mô hình dự đoán như vậy?</h4>'
        f"<ul>{items}</ul>"
        f"<p style='font-size:.8rem;margin:8px 0 0;opacity:.8'>"
        f"Dựa trên quy tắc diễn giải từ form bạn nhập + xác suất từ <code>model.pkl</code>.</p></div>",
        unsafe_allow_html=True,
    )


def render_live_metrics() -> None:
    cu1_e = st.session_state.get("cu1_enrolled", 6)
    cu1_a = st.session_state.get("cu1_approved", 5)
    cu2_e = st.session_state.get("cu2_enrolled", 6)
    cu2_a = st.session_state.get("cu2_approved", 5)
    age = st.session_state.get("age", 19)
    g1 = st.session_state.get("cu1_grade", 12.0)
    g2 = st.session_state.get("cu2_grade", 12.0)
    pr1 = cu1_a / (cu1_e + 1) * 100
    pr2 = cu2_a / (cu2_e + 1) * 100
    avg = (g1 + g2) / 2

    st.markdown("##### 📈 Chỉ số trực tiếp")
    st.caption(
        "Tính từ form (không phải ML): tỷ lệ qua môn, điểm TB, tuổi. "
        "Cập nhật khi chọn ca demo hoặc sau khi Dự đoán."
    )
    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl in zip(
        [m1, m2, m3, m4],
        [f"{pr1:.0f}%", f"{pr2:.0f}%", f"{avg:.1f}", str(age)],
        ["Tỷ lệ qua HK1", "Tỷ lệ qua HK2", "Điểm TB 2 kỳ", "Tuổi nhập học"],
    ):
        with col:
            st.markdown(
                f'<div class="live-metric"><div class="val">{val}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )


def render_semester_heatmap(cu1_g, cu2_g, cu1_e, cu1_a, cu2_e, cu2_a):
    pr1 = cu1_a / (cu1_e + 1) * 100 if cu1_e else 0
    pr2 = cu2_a / (cu2_e + 1) * 100 if cu2_e else 0

    def heat_color(val, is_pct=False):
        if is_pct:
            return "#10B981" if val >= 70 else ("#F59E0B" if val >= 40 else "#EF4444")
        return "#10B981" if val >= 12 else ("#F59E0B" if val >= 9 else "#EF4444")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="sem-box" style="border-top:4px solid {heat_color(cu1_g)}">'
            f"<b>HK1</b><br>Điểm TB: <span style='font-size:1.4rem'>{cu1_g:.1f}</span><br>"
            f"Tỷ lệ qua: {pr1:.0f}%</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="sem-box" style="border-top:4px solid {heat_color(cu2_g)}">'
            f"<b>HK2</b><br>Điểm TB: <span style='font-size:1.4rem'>{cu2_g:.1f}</span><br>"
            f"Tỷ lệ qua: {pr2:.0f}%</div>",
            unsafe_allow_html=True,
        )


def render_dropout_gauge(dropout_pct: float) -> None:
    level, color = get_risk_level(dropout_pct)
    pct = min(max(dropout_pct, 0), 100)
    gauge_df = pd.DataFrame([
        {"part": "rest", "value": 100 - pct, "order": 1},
        {"part": "risk", "value": pct, "order": 2},
    ])
    gauge = (
        alt.Chart(gauge_df)
        .mark_arc(innerRadius=70, outerRadius=110)
        .encode(
            theta=alt.Theta("value:Q", stack=True),
            color=alt.Color(
                "part:N",
                scale=alt.Scale(domain=["risk", "rest"], range=[color, "#E5E7EB"]),
                legend=None,
            ),
            order=alt.Order("order:Q"),
        )
        .properties(height=220, width=220)
    )
    g_col, t_col = st.columns([1, 2])
    with g_col:
        st.altair_chart(gauge, use_container_width=True)
    with t_col:
        st.markdown(f"### Nguy cơ bỏ học: **{level}**")
        st.markdown(f"**{pct:.1f}%** xác suất Dropout (từ model.pkl)")


def render_intervention_tips(debtor, tuition_ok, avg_grade, cu1_a, cu1_e, cu2_a, cu2_e) -> None:
    tips = []
    if debtor:
        tips.append("Hỗ trợ trả góp / miễn giảm học phí")
    if not tuition_ok:
        tips.append("Nhắc đóng học phí đúng hạn")
    if avg_grade < 10:
        tips.append("Xếp lớp củng cố / tutoring")
    if (cu1_a + cu2_a) < (cu1_e + cu2_e):
        tips.append("Cố vấn học tập — tỷ lệ qua môn thấp")
    if not tips:
        tips.append("Duy trì chương trình hiện tại — theo dõi định kỳ")
    items = "".join(f"<li>{t}</li>" for t in tips)
    st.markdown(
        f'<div class="tip-card"><h4>💡 Gợi ý can thiệp</h4><ul>{items}</ul></div>',
        unsafe_allow_html=True,
    )


def build_html_report(entry: dict) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Báo cáo - {entry['student_name']}</title>
<style>body{{font-family:Arial,sans-serif;max-width:720px;margin:40px auto;padding:20px}}
h1{{color:#2563eb}}.badge{{padding:8px 16px;border-radius:8px;color:#fff;font-weight:bold}}
</style></head><body>
<h1>🎓 Báo cáo dự đoán — {entry['student_name']}</h1>
<p><b>Thời gian:</b> {entry['timestamp']}</p>
<p><b>Kết quả:</b> <span class="badge" style="background:{CLASS_COLORS.get(entry['pred_label'],'#333')}">
{entry['pred_label']}</span></p>
<h3>Xác suất</h3>
<ul>
<li>Dropout: {entry['dropout_pct']}%</li>
<li>Enrolled: {entry['enrolled_pct']}%</li>
<li>Graduate: {entry['graduate_pct']}%</li>
</ul>
<h3>Học tập</h3>
<p>Điểm TB: {entry['avg_grade']} · Tuổi: {entry['age']} · Ngành: {entry['course']}</p>
<p><small>ML-EduPredict · model.pkl · ML2026</small></p>
</body></html>"""


def render_dashboard_page():
    st.subheader("📊 Dashboard — Tổng quan phiên")
    hist = st.session_state.history
    if not hist:
        st.info("Chưa có dữ liệu. Hãy dự đoán ít nhất một lần ở trang **Dự đoán**.")
        return

    n = len(hist)
    n_drop = sum(1 for x in hist if x["pred_label"] == "Dropout")
    n_enr = sum(1 for x in hist if x["pred_label"] == "Enrolled")
    n_grad = sum(1 for x in hist if x["pred_label"] == "Graduate")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng dự đoán", n)
    m2.metric("% Dropout", f"{n_drop / n * 100:.0f}%")
    m3.metric("% Enrolled", f"{n_enr / n * 100:.0f}%")
    m4.metric("% Graduate", f"{n_grad / n * 100:.0f}%")

    df_hist = pd.DataFrame(hist)
    c1, c2 = st.columns(2)
    with c1:
        dist = df_hist["pred_label"].value_counts().reset_index()
        dist.columns = ["Kết quả", "Số lượng"]
        st.altair_chart(
            alt.Chart(dist).mark_bar().encode(
                x="Kết quả", y="Số lượng",
                color=alt.Color("Kết quả", scale=alt.Scale(
                    domain=list(CLASS_COLORS.keys()), range=list(CLASS_COLORS.values()))),
            ).properties(height=280, title="Phân bố kết quả"),
            use_container_width=True,
        )
    with c2:
        st.altair_chart(
            alt.Chart(df_hist).mark_bar().encode(
                x=alt.X("student_name", sort="-y"),
                y=alt.Y("dropout_pct", title="Dropout %"),
                color=alt.Color("pred_label", scale=alt.Scale(
                    domain=list(CLASS_COLORS.keys()), range=list(CLASS_COLORS.values()))),
            ).properties(height=280, title="Nguy cơ Dropout theo SV"),
            use_container_width=True,
        )

    st.markdown("##### 🚨 Top 3 sinh viên nguy cơ cao nhất")
    top = sorted(hist, key=lambda x: x["dropout_pct"], reverse=True)[:3]
    for i, e in enumerate(top, 1):
        st.markdown(
            f"**{i}. {e['student_name']}** — Dropout **{e['dropout_pct']}%** "
            f"({e['pred_label']}) · {e['timestamp']}"
        )


def render_history_page():
    st.subheader("📜 Lịch sử dự đoán")

    if not st.session_state.history:
        st.markdown("""
<div class="empty-state">
  <h3>📭 Chưa có lịch sử dự đoán</h3>
  <p>Hãy thử dự đoán một sinh viên để xem timeline và so sánh tại đây.</p>
</div>
""", unsafe_allow_html=True)
        if st.button("⭐ Thử ca demo Minh Anh", type="primary", use_container_width=True):
            st.session_state._apply_minh_anh = True
            st.session_state.page = "predict"
            st.rerun()
        return

    hist = st.session_state.history
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng", len(hist))
    m2.metric("Dropout", sum(1 for x in hist if x["pred_label"] == "Dropout"))
    m3.metric("Enrolled", sum(1 for x in hist if x["pred_label"] == "Enrolled"))
    m4.metric("Graduate", sum(1 for x in hist if x["pred_label"] == "Graduate"))

    labels = [f"{x['student_name']} ({x['timestamp']})" for x in hist]
    st.markdown("##### ⚖️ So sánh 2 sinh viên (A vs B)")
    picked = st.multiselect("Chọn tối đa 2 bản ghi", labels, max_selections=2)
    if len(picked) == 2:
        e1 = hist[labels.index(picked[0])]
        e2 = hist[labels.index(picked[1])]
        cmp = pd.DataFrame({
            "Lớp": ["Dropout", "Enrolled", "Graduate"],
            "A": [e1["dropout_pct"], e1["enrolled_pct"], e1["graduate_pct"]],
            "B": [e2["dropout_pct"], e2["enrolled_pct"], e2["graduate_pct"]],
        })
        cmp_long = cmp.melt("Lớp", var_name="Sinh viên", value_name="Xác suất (%)")
        cmp_long["Sinh viên"] = cmp_long["Sinh viên"].map({"A": e1["student_name"], "B": e2["student_name"]})
        st.altair_chart(
            alt.Chart(cmp_long).mark_bar().encode(
                x="Lớp", y="Xác suất (%)", color="Sinh viên", column="Sinh viên",
            ).properties(height=300, title="So sánh xác suất"),
            use_container_width=True,
        )

    st.markdown("##### 🗂️ Timeline thẻ")
    for e in hist:
        color = CLASS_COLORS.get(e["pred_label"], "#64748b")
        st.markdown(
            f'<div class="hist-card" style="border-color:{color}">'
            f"<b>{e['student_name']}</b> · <span style='color:{color}'>{e['pred_label']}</span> · "
            f"{e['timestamp']}<br>"
            f"Dropout {e['dropout_pct']}% · Điểm TB {e['avg_grade']} · Tuổi {e['age']}"
            f"</div>",
            unsafe_allow_html=True,
        )

    df_hist = pd.DataFrame([{
        "Thời gian": x["timestamp"], "Tên SV": x["student_name"], "Kết quả": x["pred_label"],
        "Dropout %": x["dropout_pct"], "Enrolled %": x["enrolled_pct"], "Graduate %": x["graduate_pct"],
        "Điểm TB": x["avg_grade"], "Tuổi": x["age"], "Ngành": x["course"],
    } for x in hist])

    st.dataframe(df_hist, use_container_width=True, hide_index=True)
    b1, b2 = st.columns(2)
    with b1:
        st.download_button("⬇️ Xuất CSV", df_hist.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"history_{datetime.now():%Y%m%d}.csv", use_container_width=True)
    with b2:
        if st.button("🗑️ Xóa lịch sử", use_container_width=True):
            st.session_state.history = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 Student Dropout")
    st.caption("Dự đoán kết quả học tập sinh viên")
    st.divider()

    theme_label = st.toggle("🌙 Dark mode", value=(st.session_state.theme == "dark"))
    st.session_state.theme = "dark" if theme_label else "light"
    st.divider()

    menu = st.radio(
        "Menu",
        ["🔮 Dự đoán", "📊 Dashboard", "📜 Lịch sử"],
        label_visibility="collapsed",
    )
    if "Dashboard" in menu:
        st.session_state.page = "dashboard"
    elif "Lịch sử" in menu:
        st.session_state.page = "history"
    else:
        st.session_state.page = "predict"

    st.divider()
    if MODEL_LOADED:
        st.success("✅ model.pkl đã load")
        st.metric("Pipeline", "PCA+KMeans" if USE_PCA_PIPELINE else "Scale+Ensemble")
        if st.session_state.history:
            st.metric("Dự đoán phiên này", len(st.session_state.history))
    else:
        st.error("❌ Thiếu model.pkl")
        st.code("python3 train_and_save.py", language="bash")

inject_theme_css(st.session_state.theme)

st.markdown("""
<div class="hero">
    <h1>🎓 Dự đoán Bỏ học &amp; Kết quả Học tập Sinh viên</h1>
    <p>Nhập thông tin sinh viên · Dự đoán bằng <b>model.pkl</b> (Voting Ensemble)</p>
</div>
""", unsafe_allow_html=True)

# ── Routing ───────────────────────────────────────────────────────────────────
if st.session_state.page == "dashboard":
    render_dashboard_page()
    render_footer()
    st.stop()

if st.session_state.page == "history":
    render_history_page()
    render_footer()
    st.stop()

# ── Trang DỰ ĐOÁN ────────────────────────────────────────────────────────────
if st.session_state._apply_minh_anh:
    apply_demo_preset(MINH_ANH_PRESET)
    st.session_state._last_demo_preset = MINH_ANH_PRESET
    if "demo_preset" in st.session_state:
        del st.session_state["demo_preset"]
    st.session_state._apply_minh_anh = False

if st.session_state._do_reset:
    for k, v in FORM_DEFAULTS.items():
        st.session_state[k] = v
    st.session_state._last_demo_preset = DEMO_PRESET_OPTIONS[0]
    if "demo_preset" in st.session_state:
        del st.session_state["demo_preset"]
    st.session_state._do_reset = False

pc1, pc2 = st.columns([2, 1])
with pc1:
    selected_preset = st.selectbox(
        "📌 Chọn ca demo nhanh",
        DEMO_PRESET_OPTIONS,
        key="demo_preset",
        help="Điền sẵn form từ Demo Scenario.text",
    )
with pc2:
    st.write("")
    if st.button("🔄 Đặt lại mặc định", use_container_width=True):
        st.session_state._do_reset = True
        st.rerun()

if selected_preset != st.session_state._last_demo_preset:
    if selected_preset in DEMO_PRESETS:
        apply_demo_preset(selected_preset)
    st.session_state._last_demo_preset = selected_preset
    st.rerun()

st.text_input("👤 Tên sinh viên (tuỳ chọn)", key="student_name", help="Chỉ để hiển thị — không ảnh hưởng model.pkl.")
render_live_metrics()
render_step_indicator()
st.divider()

with st.form("predict_form"):
    st.subheader("📋 Thông tin cá nhân & Hành chính")
    c1, c2, c3 = st.columns(3)
    with c1:
        marital_status = st.selectbox(
            "Tình trạng hôn nhân", [1, 2, 3, 4, 5, 6],
            format_func=lambda x: {1: "Độc thân", 2: "Kết hôn", 3: "Goá", 4: "Ly hôn", 5: "Chung sống", 6: "Tách thân"}[x],
            key="marital_status", help=FIELD_HELP["marital_status"],
        )
        application_mode = st.selectbox(
            "Phương thức đăng ký", list(range(1, 19)), key="application_mode",
            help=FIELD_HELP["application_mode"],
        )
        application_order = st.slider(
            "Thứ tự đăng ký", 0, 9, key="application_order", help=FIELD_HELP["application_order"],
        )
        course = st.number_input("Mã ngành học", 1, 9999, key="course", help=FIELD_HELP["course"])
    with c2:
        daytime = st.radio(
            "Ca học", [1, 0], format_func=lambda x: "Ban ngày" if x == 1 else "Ban đêm",
            key="daytime", help=FIELD_HELP["daytime"],
        )
        prev_qual = st.selectbox(
            "Trình độ đầu vào", list(range(1, 18)), key="prev_qual", help=FIELD_HELP["prev_qual"],
        )
        prev_qual_grade = st.slider(
            "Điểm đầu vào (0–200)", 0.0, 200.0, step=0.5, key="prev_qual_grade",
            help=FIELD_HELP["prev_qual_grade"],
        )
        nationality = st.number_input(
            "Quốc tịch (mã số)", 1, 109, key="nationality", help=FIELD_HELP["nationality"],
        )
    with c3:
        mothers_qual = st.selectbox(
            "Trình độ mẹ", list(range(1, 30)), key="mothers_qual", help=FIELD_HELP["mothers_qual"],
        )
        fathers_qual = st.selectbox(
            "Trình độ cha", list(range(1, 30)), key="fathers_qual", help=FIELD_HELP["fathers_qual"],
        )
        mothers_occ = st.number_input(
            "Nghề nghiệp mẹ (mã)", 0, 195, key="mothers_occ", help=FIELD_HELP["mothers_occ"],
        )
        fathers_occ = st.number_input(
            "Nghề nghiệp cha (mã)", 0, 195, key="fathers_occ", help=FIELD_HELP["fathers_occ"],
        )

    st.divider()
    st.subheader("👤 Đặc điểm sinh viên")
    c4, c5, c6 = st.columns(3)
    with c4:
        displaced = st.radio(
            "Di dân?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có",
            key="displaced", help=FIELD_HELP["displaced"],
        )
        special_needs = st.radio(
            "Nhu cầu đặc biệt?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có",
            key="special_needs", help=FIELD_HELP["special_needs"],
        )
        debtor = st.radio("Đang nợ học phí?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có", key="debtor", help=FIELD_HELP["debtor"])
    with c5:
        tuition_ok = st.radio("Học phí đầy đủ?", [1, 0], format_func=lambda x: "Có" if x == 1 else "Không", key="tuition_ok", help=FIELD_HELP["tuition_ok"])
        gender = st.radio(
            "Giới tính", [1, 0], format_func=lambda x: "Nam" if x == 1 else "Nữ",
            key="gender", help=FIELD_HELP["gender"],
        )
        scholarship = st.radio("Có học bổng?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có", key="scholarship", help=FIELD_HELP["scholarship"])
    with c6:
        age = st.slider("Tuổi khi nhập học", 17, 70, key="age", help=FIELD_HELP["age"])
        international = st.radio(
            "Sinh viên quốc tế?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có",
            key="international", help=FIELD_HELP["international"],
        )
        admission_grade = st.slider("Điểm tuyển sinh (0–200)", 0.0, 200.0, step=0.5, key="admission_grade", help=FIELD_HELP["admission_grade"])

    st.divider()
    st.subheader("📊 Kết quả học tập theo kỳ")
    c7, c8 = st.columns(2)
    with c7:
        st.markdown("**Học kỳ 1**")
        cu1_credited = st.number_input(
            "Môn được miễn (HK1)", 0, 20, key="cu1_credited", help=FIELD_HELP["cu1_credited"],
        )
        cu1_enrolled = st.number_input("Môn đăng ký (HK1)", 0, 30, key="cu1_enrolled", help=FIELD_HELP["cu1_enrolled"])
        cu1_evals = st.number_input("Số lần đánh giá (HK1)", 0, 45, key="cu1_evals", help=FIELD_HELP["cu1_evals"])
        cu1_approved = st.number_input("Môn qua (HK1)", 0, 30, key="cu1_approved", help=FIELD_HELP["cu1_approved"])
        cu1_grade = st.slider("Điểm TB HK1 (0–20)", 0.0, 20.0, step=0.1, key="cu1_grade", help=FIELD_HELP["cu1_grade"])
        cu1_no_eval = st.number_input(
            "Môn không đánh giá (HK1)", 0, 20, key="cu1_no_eval", help=FIELD_HELP["cu1_no_eval"],
        )
    with c8:
        st.markdown("**Học kỳ 2**")
        cu2_credited = st.number_input(
            "Môn được miễn (HK2)", 0, 20, key="cu2_credited", help=FIELD_HELP["cu2_credited"],
        )
        cu2_enrolled = st.number_input("Môn đăng ký (HK2)", 0, 30, key="cu2_enrolled", help=FIELD_HELP["cu2_enrolled"])
        cu2_evals = st.number_input("Số lần đánh giá (HK2)", 0, 45, key="cu2_evals", help=FIELD_HELP["cu2_evals"])
        cu2_approved = st.number_input("Môn qua (HK2)", 0, 30, key="cu2_approved", help=FIELD_HELP["cu2_approved"])
        cu2_grade = st.slider("Điểm TB HK2 (0–20)", 0.0, 20.0, step=0.1, key="cu2_grade", help=FIELD_HELP["cu2_grade"])
        cu2_no_eval = st.number_input(
            "Môn không đánh giá (HK2)", 0, 20, key="cu2_no_eval", help=FIELD_HELP["cu2_no_eval"],
        )

    st.divider()
    st.subheader("🌍 Kinh tế vĩ mô")
    c9, c10, c11 = st.columns(3)
    with c9:
        unemployment = st.slider(
            "Tỷ lệ thất nghiệp (%)", 0.0, 25.0, step=0.1, key="unemployment",
            help=FIELD_HELP["unemployment"],
        )
    with c10:
        inflation = st.slider(
            "Tỷ lệ lạm phát (%)", -5.0, 10.0, step=0.1, key="inflation", help=FIELD_HELP["inflation"],
        )
    with c11:
        gdp = st.slider(
            "GDP tăng trưởng (%)", -10.0, 10.0, step=0.01, key="gdp", help=FIELD_HELP["gdp"],
        )

    submitted = st.form_submit_button("🔮 Dự đoán kết quả", use_container_width=True, type="primary")

# ══════════════════════════════════════════════════════════════════════════════
# LOGIC PREDICT — GIỮ NGUYÊN (model.pkl + feature engineering)
# ══════════════════════════════════════════════════════════════════════════════
if submitted:
    if not MODEL_LOADED:
        st.error("Model chưa load. Chạy: python3 train_and_save.py --data dataset.csv")
    else:
        raw = {
            "Marital_status": marital_status,
            "Application_mode": application_mode,
            "Application_order": application_order,
            "Course": course,
            "Daytime/evening_attendance\t": daytime,
            "Previous_qualification": prev_qual,
            "Previous_qualification_(grade)": prev_qual_grade,
            "Nacionality": nationality,
            "Mother's_qualification": mothers_qual,
            "Father's_qualification": fathers_qual,
            "Mother's_occupation": mothers_occ,
            "Father's_occupation": fathers_occ,
            "Admission_grade": admission_grade,
            "Displaced": displaced,
            "Educational_special_needs": special_needs,
            "Debtor": debtor,
            "Tuition_fees_up_to_date": tuition_ok,
            "Gender": gender,
            "Scholarship_holder": scholarship,
            "Age_at_enrollment": age,
            "International": international,
            "Curricular_units_1st_sem_(credited)": cu1_credited,
            "Curricular_units_1st_sem_(enrolled)": cu1_enrolled,
            "Curricular_units_1st_sem_(evaluations)": cu1_evals,
            "Curricular_units_1st_sem_(approved)": cu1_approved,
            "Curricular_units_1st_sem_(grade)": cu1_grade,
            "Curricular_units_1st_sem_(without_evaluations)": cu1_no_eval,
            "Curricular_units_2nd_sem_(credited)": cu2_credited,
            "Curricular_units_2nd_sem_(enrolled)": cu2_enrolled,
            "Curricular_units_2nd_sem_(evaluations)": cu2_evals,
            "Curricular_units_2nd_sem_(approved)": cu2_approved,
            "Curricular_units_2nd_sem_(grade)": cu2_grade,
            "Curricular_units_2nd_sem_(without_evaluations)": cu2_no_eval,
            "Unemployment_rate": unemployment,
            "Inflation_rate": inflation,
            "GDP": gdp,
        }

        df = pd.DataFrame([raw])

        df["pass_rate_1"] = df["Curricular_units_1st_sem_(approved)"] / (
            df["Curricular_units_1st_sem_(enrolled)"] + 1
        )
        df["pass_rate_2"] = df["Curricular_units_2nd_sem_(approved)"] / (
            df["Curricular_units_2nd_sem_(enrolled)"] + 1
        )
        df["grade_progress"] = (
            df["Curricular_units_2nd_sem_(grade)"] - df["Curricular_units_1st_sem_(grade)"]
        )
        df["total_approved"] = (
            df["Curricular_units_1st_sem_(approved)"] + df["Curricular_units_2nd_sem_(approved)"]
        )
        df["avg_grade"] = (
            df["Curricular_units_1st_sem_(grade)"] + df["Curricular_units_2nd_sem_(grade)"]
        ) / 2

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        X_scaled = scaler.transform(df[feature_cols].values)
        if USE_PCA_PIPELINE:
            X_pca = pca.transform(X_scaled)
            cluster = kmeans.predict(X_scaled)
            X_final = np.hstack((X_pca, cluster.reshape(-1, 1)))
        else:
            X_final = X_scaled

        pred_code = model.predict(X_final)[0]
        proba = model.predict_proba(X_final)[0]
        classes = label_encoder.classes_
        pred_label = classes[pred_code]

        avg_grade = (cu1_grade + cu2_grade) / 2
        idx = {c: i for i, c in enumerate(classes)}
        history_entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "student_name": st.session_state.get("student_name") or "Sinh viên",
            "pred_label": pred_label,
            "course": int(course),
            "age": int(age),
            "avg_grade": round(avg_grade, 2),
            "dropout_pct": round(float(proba[idx["Dropout"]] * 100), 1),
            "enrolled_pct": round(float(proba[idx["Enrolled"]] * 100), 1),
            "graduate_pct": round(float(proba[idx["Graduate"]] * 100), 1),
        }
        save_to_history(history_entry)

        dropout_pct = float(proba[idx["Dropout"]] * 100)

        if pred_label == "Graduate":
            st.balloons()
        if dropout_pct > 60:
            st.toast("⚠️ Cần can thiệp sớm!", icon="⚠️")

        st.divider()
        info = CLASS_INFO[pred_label]
        msg = {
            "Dropout": "⚠️ Sinh viên này có nguy cơ cao bỏ học. Cần can thiệp sớm!",
            "Enrolled": "📚 Sinh viên đang tiếp tục theo học.",
            "Graduate": "🎉 Sinh viên có khả năng tốt nghiệp thành công!",
        }
        st.markdown(f"""
        <div style="background:{info['bg']};border-left:6px solid {info['color']};
            border-radius:12px;padding:24px 32px;margin-bottom:16px;">
            <h2 style="color:{info['color']};margin:0 0 8px 0;">
                {info['emoji']} Kết quả: <strong>{pred_label}</strong>
                <span style="font-size:1rem;color:#666;"> — {st.session_state.get("student_name", "")}</span>
            </h2>
            <p style="margin:0;color:#444;">{msg[pred_label]}</p>
        </div>
        """, unsafe_allow_html=True)

        reasons = explain_prediction(
            pred_label, dropout_pct, debtor, tuition_ok, avg_grade,
            cu1_enrolled, cu1_approved, cu2_enrolled, cu2_approved,
            cu1_grade, cu2_grade, scholarship,
        )
        render_prediction_reasons(reasons)

        st.subheader("📉 So sánh học kỳ")
        render_semester_heatmap(cu1_grade, cu2_grade, cu1_enrolled, cu1_approved, cu2_enrolled, cu2_approved)

        st.subheader("🎯 Mức nguy cơ bỏ học")
        render_dropout_gauge(dropout_pct)
        render_intervention_tips(debtor, tuition_ok, avg_grade, cu1_approved, cu1_enrolled, cu2_approved, cu2_enrolled)

        st.subheader("📊 Xác suất từng lớp")
        prob_df = pd.DataFrame({
            "Lớp": classes,
            "Xác suất (%)": (proba * 100).round(2),
        }).sort_values("Xác suất (%)", ascending=False)

        col_bar, col_donut, col_table = st.columns([1.3, 1, 0.9])
        with col_bar:
            st.altair_chart(
                alt.Chart(prob_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("Lớp", sort=None),
                    y=alt.Y("Xác suất (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("Lớp", scale=alt.Scale(
                        domain=["Dropout", "Enrolled", "Graduate"],
                        range=["#FF4B4B", "#F59E0B", "#10B981"]), legend=None),
                    tooltip=["Lớp", "Xác suất (%)"],
                ).properties(height=280, title="Biểu đồ cột"),
                use_container_width=True,
            )
        with col_donut:
            st.altair_chart(
                alt.Chart(prob_df).mark_arc(innerRadius=55, outerRadius=95).encode(
                    theta=alt.Theta("Xác suất (%)", stack=True),
                    color=alt.Color("Lớp", scale=alt.Scale(
                        domain=["Dropout", "Enrolled", "Graduate"],
                        range=["#FF4B4B", "#F59E0B", "#10B981"])),
                    tooltip=["Lớp", alt.Tooltip("Xác suất (%)", format=".1f")],
                ).properties(height=280, title="Donut"),
                use_container_width=True,
            )
        with col_table:
            st.dataframe(
                prob_df.style.format({"Xác suất (%)": "{:.2f}%"}).highlight_max(
                    subset=["Xác suất (%)"], color="#d4edda"),
                use_container_width=True, hide_index=True,
            )

        st.download_button(
            "📄 Tải báo cáo HTML",
            build_html_report(history_entry).encode("utf-8"),
            file_name=f"report_{history_entry['id']}.html",
            mime="text/html",
            use_container_width=True,
        )

        with st.expander("🔍 Xem dữ liệu đầu vào"):
            st.dataframe(df[feature_cols], use_container_width=True)

        st.success("✅ Đã lưu lịch sử (file prediction_history.json)")

render_footer()
