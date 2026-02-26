import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from langchain_community.llms import Ollama

# ==============================
# PAGE CONFIG (MUST BE FIRST)
# ==============================
st.set_page_config(
    page_title="StatBot Pro",
    page_icon="📊",
    layout="wide"
)

# ==============================
# CUSTOM PROFESSIONAL UI
# ==============================
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        h1 {
            color: #1f4e79;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 StatBot Pro")
st.markdown("### AI-Powered CSV Data Analyst")
st.markdown("Upload your dataset and ask intelligent questions.")

# ==============================
# SESSION STATE FOR CHAT
# ==============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.markdown("---")

    question = st.text_input("💬 Ask a question about your data")

    if question:
        st.session_state.chat_history.append(("User", question))

        llm = Ollama(model="phi")

        with st.spinner("Analyzing..."):

            q = question.lower()
            answer = ""

            try:
                # ==============================
                # BASIC ANALYTICS LOGIC
                # ==============================

                if "total" in q and "sales" in q:
                    result = df["Sales"].sum()
                    answer = f"Total Sales = {result}"

                elif "average" in q and "sales" in q:
                    result = df["Sales"].mean()
                    answer = f"Average Sales = {result}"

                elif "highest" in q and "sales" in q:
                    result = df.loc[df["Sales"].idxmax()]
                    answer = f"Highest Sales Record:\n{result}"

                elif "plot" in q or "chart" in q:
                    df.groupby("Region")["Sales"].sum().plot(kind="bar")
                    plt.title("Sales by Region")
                    plt.xticks(rotation=45)
                    st.pyplot(plt)
                    answer = "Bar chart generated successfully."

                else:
                    # Fallback to AI explanation
                    prompt = f"""
                    You are a data analyst.
                    Here is the dataset preview:
                    {df.head()}

                    Question: {question}
                    Answer clearly and concisely.
                    """
                    answer = llm(prompt)

            except Exception as e:
                answer = f"Error: {str(e)}"

        st.session_state.chat_history.append(("AI", answer))

    # ==============================
    # DISPLAY CHAT HISTORY
    # ==============================
    for role, msg in st.session_state.chat_history:
        if role == "User":
            st.markdown(f"🧑 **You:** {msg}")
        else:
            st.markdown(f"🤖 **StatBot:** {msg}")

    # ==============================
    # DOWNLOAD REPORT
    # ==============================
    if st.session_state.chat_history:
        report_content = "StatBot Pro Analysis Report\n"
        report_content += "================================\n\n"

        for role, msg in st.session_state.chat_history:
            report_content += f"{role}: {msg}\n\n"

        st.download_button(
            label="📥 Download Analysis Report",
            data=report_content,
            file_name="statbot_report.txt",
            mime="text/plain"
        )