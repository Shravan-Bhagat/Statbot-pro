import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_community.llms import Ollama

# ===============================
# PAGE CONFIG (MUST BE FIRST)
# ===============================
st.set_page_config(
    page_title="StatBot Pro",
    page_icon="📊",
    layout="wide"
)

# ===============================
# FORCE LIGHT THEME + DARK TEXT
# ===============================
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            background-color: #f5f7fa !important;
            color: #000000 !important;
        }

        .stTextInput input {
            color: black !important;
            background-color: white !important;
        }

        .stTextInput label {
            color: black !important;
        }

        .stMarkdown, .stDataFrame, .stTable {
            color: black !important;
        }

        .chat-user {
            background-color: #d1e7ff;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
            color: black;
        }

        .chat-bot {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            border: 1px solid #dcdcdc;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 StatBot Pro")
st.markdown("### AI-Powered CSV Data Analyst")
st.markdown("Upload your dataset and ask intelligent questions.")

# ===============================
# CHAT MEMORY
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.markdown("---")

    question = st.text_input("💬 Ask something about your data")

    if question:
        st.session_state.chat_history.append(("User", question))

        llm = Ollama(model="phi")

        with st.spinner("Analyzing..."):
            q = question.lower()
            answer = ""

            try:
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
                    prompt = f"""
                    You are a professional data analyst.
                    Here is dataset preview:
                    {df.head()}

                    Question: {question}
                    Give clear answer.
                    """
                    answer = llm(prompt)

            except Exception as e:
                answer = f"Error: {str(e)}"

        st.session_state.chat_history.append(("AI", answer))

    # ===============================
    # DISPLAY CHAT WITH BUBBLES
    # ===============================
    for role, msg in st.session_state.chat_history:
        if role == "User":
            st.markdown(f"<div class='chat-user'><b>You:</b> {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'><b>StatBot:</b> {msg}</div>", unsafe_allow_html=True)

    # ===============================
    # DOWNLOAD REPORT
    # ===============================
    if st.session_state.chat_history:
        report = "StatBot Pro Analysis Report\n"
        report += "=================================\n\n"

        for role, msg in st.session_state.chat_history:
            report += f"{role}: {msg}\n\n"

        st.download_button(
            label="📥 Download Analysis Report",
            data=report,
            file_name="statbot_report.txt",
            mime="text/plain"
        )