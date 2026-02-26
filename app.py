import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_community.llms import Ollama

st.set_page_config(page_title="StatBot Pro", layout="wide")
st.title("📊 StatBot Pro – Local CSV AI")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview")
    st.dataframe(df.head())

    question = st.text_input("Ask your question (example: total Sales, plot Sales by Region)")

    if question:
        llm = Ollama(model="phi")

        with st.spinner("Analyzing..."):

            q = question.lower()

            # SIMPLE AUTO LOGIC (no LangChain agent)
            try:
                if "total" in q and "sales" in q:
                    result = df["Sales"].sum()
                    st.write("Total Sales:", result)

                elif "average" in q and "sales" in q:
                    result = df["Sales"].mean()
                    st.write("Average Sales:", result)

                elif "plot" in q or "chart" in q:
                    df.groupby("Region")["Sales"].sum().plot(kind="bar")
                    plt.title("Sales by Region")
                    st.pyplot(plt)

                else:
                    # fallback to Ollama text answer
                    prompt = f"Here is a dataframe:\n{df.head()}\n\nQuestion: {question}"
                    answer = llm(prompt)
                    st.write(answer)

            except Exception as e:
                st.error(str(e))