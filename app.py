"""
app.py — Streamlit Web App: Dự đoán Bỏ học Sinh viên
Cách chạy: streamlit run app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓", layout="wide")

@st.cache_resource
def load_pipeline(path="model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    pipeline      = load_pipeline()
    model         = pipeline["model"]
    label_encoder = pipeline["label_encoder"]
    feature_cols  = pipeline["feature_cols"]
    scaler        = pipeline["scaler"]
    MODEL_LOADED  = True
except FileNotFoundError:
    MODEL_LOADED  = False

CLASS_INFO = {
    "Dropout":  {"emoji": "⚠️", "color": "#FF4B4B", "bg": "#FFF0F0"},
    "Enrolled": {"emoji": "📚", "color": "#F59E0B", "bg": "#FFFBEB"},
    "Graduate": {"emoji": "🎓", "color": "#10B981", "bg": "#F0FFF4"},
}

with st.sidebar:
    st.title("🎓 Student Dropout")
    st.caption("Dự đoán kết quả học tập sinh viên")
    st.divider()
    if MODEL_LOADED:
        st.success("✅ Model đã sẵn sàng")
        st.metric("Thuật toán", "XGBoost")
        st.metric("Pipeline", "PCA + KMeans + SMOTE")
        st.divider()
        st.info("Nhập thông tin sinh viên rồi nhấn **Dự đoán**.")
    else:
        st.error("❌ Chưa tìm thấy model.pkl")
        st.warning("Chạy lệnh:\n```\npython3 train_and_save.py --data dataset.csv\n```")

st.title("🎓 Dự đoán Bỏ học & Kết quả Học tập Sinh viên")
st.markdown("Nhập thông tin sinh viên. Mô hình **XGBoost** phân loại: **Bỏ học**, **Đang học**, **Tốt nghiệp**.")
st.divider()

with st.form("predict_form"):
    st.subheader("📋 Thông tin cá nhân & Hành chính")
    c1, c2, c3 = st.columns(3)
    with c1:
        marital_status    = st.selectbox("Tình trạng hôn nhân", [1,2,3,4,5,6],
                            format_func=lambda x:{1:"Độc thân",2:"Kết hôn",3:"Goá",4:"Ly hôn",5:"Chung sống",6:"Tách thân"}[x])
        application_mode  = st.selectbox("Phương thức đăng ký", list(range(1,19)))
        application_order = st.slider("Thứ tự đăng ký", 0, 9, 1)
        course            = st.number_input("Mã ngành học", 1, 9999, 9500)
    with c2:
        daytime           = st.radio("Ca học", [1,0], format_func=lambda x:"Ban ngày" if x==1 else "Ban đêm")
        prev_qual         = st.selectbox("Trình độ đầu vào", list(range(1,18)))
        prev_qual_grade   = st.slider("Điểm đầu vào (0–200)", 0.0, 200.0, 130.0, step=0.5)
        nationality       = st.number_input("Quốc tịch (mã số)", 1, 109, 1)
    with c3:
        mothers_qual      = st.selectbox("Trình độ mẹ", list(range(1,30)))
        fathers_qual      = st.selectbox("Trình độ cha", list(range(1,30)))
        mothers_occ       = st.number_input("Nghề nghiệp mẹ (mã)", 0, 195, 5)
        fathers_occ       = st.number_input("Nghề nghiệp cha (mã)", 0, 195, 5)

    st.divider()
    st.subheader("👤 Đặc điểm sinh viên")
    c4, c5, c6 = st.columns(3)
    with c4:
        displaced     = st.radio("Di dân?",           [0,1], format_func=lambda x:"Không" if x==0 else "Có")
        special_needs = st.radio("Nhu cầu đặc biệt?", [0,1], format_func=lambda x:"Không" if x==0 else "Có")
        debtor        = st.radio("Đang nợ học phí?",  [0,1], format_func=lambda x:"Không" if x==0 else "Có")
    with c5:
        tuition_ok    = st.radio("Học phí đầy đủ?",   [1,0], format_func=lambda x:"Có" if x==1 else "Không")
        gender        = st.radio("Giới tính",          [1,0], format_func=lambda x:"Nam" if x==1 else "Nữ")
        scholarship   = st.radio("Có học bổng?",       [0,1], format_func=lambda x:"Không" if x==0 else "Có")
    with c6:
        age           = st.slider("Tuổi khi nhập học", 17, 70, 19)
        international = st.radio("Sinh viên quốc tế?", [0,1], format_func=lambda x:"Không" if x==0 else "Có")
        admission_grade = st.slider("Điểm tuyển sinh (0–200)", 0.0, 200.0, 130.0, step=0.5)

    st.divider()
    st.subheader("📊 Kết quả học tập theo kỳ")
    c7, c8 = st.columns(2)
    with c7:
        st.markdown("**Học kỳ 1**")
        cu1_credited = st.number_input("Môn được miễn (HK1)",      0, 20, 0)
        cu1_enrolled = st.number_input("Môn đăng ký (HK1)",        0, 30, 6)
        cu1_evals    = st.number_input("Số lần đánh giá (HK1)",    0, 45, 6)
        cu1_approved = st.number_input("Môn qua (HK1)",            0, 30, 5)
        cu1_grade    = st.slider("Điểm TB HK1 (0–20)", 0.0, 20.0, 12.0, step=0.1)
        cu1_no_eval  = st.number_input("Môn không đánh giá (HK1)", 0, 20, 0)
    with c8:
        st.markdown("**Học kỳ 2**")
        cu2_credited = st.number_input("Môn được miễn (HK2)",      0, 20, 0)
        cu2_enrolled = st.number_input("Môn đăng ký (HK2)",        0, 30, 6)
        cu2_evals    = st.number_input("Số lần đánh giá (HK2)",    0, 45, 6)
        cu2_approved = st.number_input("Môn qua (HK2)",            0, 30, 5)
        cu2_grade    = st.slider("Điểm TB HK2 (0–20)", 0.0, 20.0, 12.0, step=0.1)
        cu2_no_eval  = st.number_input("Môn không đánh giá (HK2)", 0, 20, 0)

    st.divider()
    st.subheader("🌍 Kinh tế vĩ mô")
    c9, c10, c11 = st.columns(3)
    with c9:
        unemployment = st.slider("Tỷ lệ thất nghiệp (%)", 0.0, 25.0, 10.8, step=0.1)
    with c10:
        inflation    = st.slider("Tỷ lệ lạm phát (%)", -5.0, 10.0, 1.4, step=0.1)
    with c11:
        gdp          = st.slider("GDP tăng trưởng (%)", -10.0, 10.0, 1.74, step=0.01)

    submitted = st.form_submit_button("🔮 Dự đoán kết quả", use_container_width=True, type="primary")

if submitted:
    if not MODEL_LOADED:
        st.error("Model chưa load. Chạy train_and_save.py trước.")
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

        # Bổ sung cột thiếu nếu có
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Áp dụng đúng pipeline: Scale → PCA → KMeans → Predict
        X_scaled  = scaler.transform(df[feature_cols].values)
        X_final = scaler.transform(df[feature_cols].values)

        pred_code = model.predict(X_final)[0]
        proba     = model.predict_proba(X_final)[0] 
        classes    = label_encoder.classes_
        pred_label = classes[pred_code]

        st.divider()
        info = CLASS_INFO[pred_label]
        msg = {
            "Dropout":  "⚠️ Sinh viên này <b>có nguy cơ cao bỏ học</b>. Cần can thiệp sớm!",
            "Enrolled": "📚 Sinh viên đang <b>tiếp tục theo học</b> bình thường.",
            "Graduate": "🎉 Sinh viên có khả năng <b>tốt nghiệp thành công</b>!",
        }
        st.markdown(f"""
        <div style="background:{info['bg']};border-left:6px solid {info['color']};
                    border-radius:12px;padding:24px 32px;margin-bottom:24px;">
            <h2 style="color:{info['color']};margin:0 0 8px 0;">
                {info['emoji']} Kết quả: <strong>{pred_label}</strong>
            </h2>
            <p style="font-size:16px;margin:0;color:#444;">{msg[pred_label]}</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📊 Xác suất từng lớp")
        prob_df = pd.DataFrame({
            "Lớp": classes,
            "Xác suất (%)": (proba * 100).round(2),
        }).sort_values("Xác suất (%)", ascending=False)

        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            import altair as alt
            chart = (
                alt.Chart(prob_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Lớp", sort=None, axis=alt.Axis(labelFontSize=13)),
                    y=alt.Y("Xác suất (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("Lớp",
                        scale=alt.Scale(
                            domain=["Dropout","Enrolled","Graduate"],
                            range=["#FF4B4B","#F59E0B","#10B981"]),
                        legend=None),
                    tooltip=["Lớp","Xác suất (%)"],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)
        with col_table:
            st.dataframe(
                prob_df.style.format({"Xác suất (%)": "{:.2f}%"})
                             .highlight_max(subset=["Xác suất (%)"], color="#d4edda"),
                use_container_width=True, hide_index=True,
            )

        with st.expander("🔍 Xem dữ liệu đầu vào"):
            st.dataframe(df[feature_cols], use_container_width=True)
