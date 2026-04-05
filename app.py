import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_community.llms import Ollama

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="StatBot Pro",
    page_icon="📊",
    layout="wide"
)

# ===============================
# UI STYLE
# ===============================
st.markdown("""
<style>

html, body {
    background-color: #0e1117;
    color: white;
}

/* Header */
.header {
    text-align: center;
    padding: 30px;
    border-radius: 15px;
    background: linear-gradient(90deg,#4facfe,#00f2fe);
    color: white;
    margin-bottom: 30px;
}

/* Upload box */
.upload-box {
    border: 2px dashed #4facfe;
    padding: 25px;
    border-radius: 12px;
    background-color: #1c1f26;
    text-align: center;
    margin-bottom: 20px;
}

/* Dataset card */
.data-card {
    background: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 25px;
}

/* User message */
.user-msg {
    background-color: #3b82f6;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: white;
}

/* Bot message */
.bot-msg {
    background-color: #1c1f26;
    border: 1px solid #374151;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 12px;
    color: white;
}

/* Input box */
.stTextInput input {
    background-color: #1c1f26;
    color: white;
    border: 1px solid #374151;
    border-radius: 8px;
}

/* Buttons */
.stButton button {
    background-color: #374151;
    color: white;
    border-radius: 8px;
}

.stDownloadButton button {
    background-color: #4facfe;
    color: white;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown("""
<div class="header">
<h1>📊 StatBot Pro</h1>
<h4>AI Powered CSV Data Analyst</h4>
<p>Upload your dataset and ask intelligent questions instantly</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# SESSION STATE
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===============================
# CLEAR CHAT BUTTON
# ===============================
if st.button("🗑 Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# ===============================
# FILE UPLOADER
# ===============================
st.markdown('<div class="upload-box">Upload your CSV dataset below</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# ===============================
# MAIN LOGIC
# ===============================
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.markdown('<div class="data-card">', unsafe_allow_html=True)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.markdown('</div>', unsafe_allow_html=True)

    question = st.text_input("💬 Ask something about your data")

    if question:

        st.session_state.chat_history.append(("User", question))

        llm = Ollama(model="phi")

        with st.spinner("🤖 AI analyzing your data..."):

            q = question.lower()
            answer = ""

            try:

                if "total" in q and "sales" in q:
                    result = df["Sales"].sum()
                    answer = f"📈 Total Sales = {result}"

                elif "average" in q and "sales" in q:
                    result = df["Sales"].mean()
                    answer = f"📊 Average Sales = {result}"

                elif "maximum" in q and "sales" in q:
                    result = df["Sales"].max()
                    answer = f"📊 Maximum Sales = {result}"

                elif "minimum" in q and "sales" in q:
                    result = df["Sales"].min()
                    answer = f"📉 Minimum Sales = {result}"

                elif "highest" in q and "sales" in q:
                    result = df.loc[df["Sales"].idxmax()]
                    answer = f"🏆 Highest Sales Record:\n\n{result}"

                elif "plot" in q or "chart" in q:

                    fig, ax = plt.subplots()

                    df.groupby("Region")["Sales"].sum().plot(
                        kind="bar",
                        ax=ax,
                        color="#4facfe"
                    )

                    plt.title("Sales by Region")
                    plt.xticks(rotation=30)

                    st.pyplot(fig)

                    answer = "📊 Chart generated successfully."

                else:

                    prompt = f"""
You are a professional data analyst.

Dataset preview:
{df.head()}

Question:
{question}

Give a short clear answer.
"""

                    answer = llm.invoke(prompt)

            except Exception as e:
                answer = f"❌ Error: {str(e)}"

        st.session_state.chat_history.append(("AI", answer))

# ===============================
# CHAT DISPLAY
# ===============================
for role, msg in st.session_state.chat_history:

    if role == "User":
        st.markdown(
            f"<div class='user-msg'>👤 <b>You:</b> {msg}</div>",
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"<div class='bot-msg'>🤖 <b>StatBot:</b> {msg}</div>",
            unsafe_allow_html=True
        )

# ===============================
# DOWNLOAD REPORT
# ===============================
if st.session_state.chat_history:

    report = "StatBot Pro Analysis Report\n"
    report += "===========================\n\n"

    for role, msg in st.session_state.chat_history:
        report += f"{role}: {msg}\n\n"

    st.download_button(
        "📥 Download Analysis Report",
        report,
        file_name="statbot_report.txt"
    )