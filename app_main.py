import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
df = pd.read_csv('student_habits_performance.csv')
df = df.dropna()  # Drop missing values to match notebook preprocessing
model = joblib.load('best_model.pkl')

# Function to predict grade
def predict_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

# Main header
st.markdown('<h1 class="main-header">📊 Student Performance Analysis & Prediction Dashboard</h1>', unsafe_allow_html=True)

# Navigation with tabs
tab1, tab2, tab3 = st.tabs(["📈 Data Overview", "📊 Visualizations", "🔮 Prediction"])

with tab1:
    st.header("Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Average Exam Score", f"{df['exam_score'].mean():.1f}")
    with col3:
        st.metric("Highest Score", df['exam_score'].max())

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    with st.expander("View Full Dataset"):
        st.dataframe(df, use_container_width=True)

with tab2:
    st.header("Advanced Visualizations")

    # Distribution plots
    st.subheader("Score Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df['exam_score'], bins=20, ax=ax, color='#4CAF50')
        ax.set_title('Exam Score Distribution')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df, y='exam_score', ax=ax, color='#FF9800')
        ax.set_title('Exam Score Box Plot')
        st.pyplot(fig)

    # Categorical comparisons
    st.subheader("Performance by Categories")
    categories = ['gender', 'part_time_job', 'parental_education_level', 'internet_quality']
    cols = st.columns(2)
    for i, cat in enumerate(categories):
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df, x=cat, y='exam_score', ax=ax, palette='Set2')
            ax.set_title(f'Exam Score by {cat.replace("_", " ").title()}')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Feature Correlations")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)

with tab3:
    st.header("Performance Prediction")

    st.markdown("### Enter Student Details for Prediction")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider('📚 Study Hours per Day', 0.0, 12.0, 2.0)
        attendance = st.slider('📅 Attendance Percentage', 0.0, 100.0, 80.0)
        mental_health = st.slider('🧠 Mental Health Rating (1-10)', 1, 10)

    with col2:
        sleep_hours = st.slider('😴 Sleep Hours per Night', 0.0, 12.0, 7.0, 0.5)
        part_time_job = st.selectbox('💼 Part-time Job', ['No', 'Yes'])

    ptj_encoded = 1 if part_time_job == 'Yes' else 0

    if st.button('🔮 Predict Performance', type='primary'):
        with st.spinner('Analyzing...'):
            input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])
            prediction = model.predict(input_data)[0]
            prediction = max(0, min(100, prediction))
            grade = predict_grade(prediction)

        st.success(f'🎯 Predicted Performance Score: **{prediction:.2f}**')

        # Progress bar for score
        st.progress(prediction / 100)
        st.caption(f"Score visualization: {prediction:.1f}/100")

# Footer
st.markdown("---")
st.markdown("© 2026 AnkurBhowmik. All rights reserved. | Data Science Project")