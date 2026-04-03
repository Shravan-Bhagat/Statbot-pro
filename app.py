import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — faster rendering
import matplotlib.pyplot as plt
from langchain_community.llms import Ollama
from components import (
    load_css,
    render_header,
    render_upload_area,
    render_stats_panel,
    render_data_card,
    render_chat_history,
    render_download_report,
    validate_csv_file,
    sanitize_dataframe,
)

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="StatBot Pro",
    page_icon="⚡",
    layout="wide",
)

# ===============================
# LOAD FRONTEND
# ===============================
load_css()
render_header()

# ===============================
# SESSION STATE
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===============================
# CACHED FUNCTIONS
# ===============================

@st.cache_data(show_spinner="Loading dataset...")
def load_csv(file):
    """Load, cache, validate, and sanitize the CSV."""
    df = pd.read_csv(file, low_memory=False, on_bad_lines="skip")
    df = sanitize_dataframe(df)
    return df


@st.cache_resource
def get_llm():
    """Create the LLM once and reuse across queries."""
    return Ollama(model="llama3", timeout=60)


@st.cache_data(show_spinner=False, max_entries=200)
def build_data_summary(df_hash, df_shape, describe_str, col_info_str, cat_info_str, sample_str):
    """Build a compact data summary string. Cached by content hash."""
    parts = [
        f"Dataset: {df_shape[0]:,} rows × {df_shape[1]} columns",
        f"Columns: {col_info_str}",
        f"Statistics (all rows):\n{describe_str}",
    ]
    if cat_info_str:
        parts.append(f"Categorical columns:\n{cat_info_str}")
    parts.append(f"Sample rows:\n{sample_str}")
    return "\n\n".join(parts)


def prepare_summary_inputs(df):
    """Prepare hashable inputs for the cached build_data_summary."""
    import hashlib
    # Create a content hash from shape + column names + first/last rows
    hash_input = f"{df.shape}_{list(df.columns)}_{df.head(1).to_string()}_{df.tail(1).to_string()}"
    df_hash = hashlib.md5(hash_input.encode()).hexdigest()

    col_info = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)

    # Compact describe — limit to 6 columns max for prompt size
    num_cols = df.select_dtypes(include="number").columns[:6]
    describe_str = df[num_cols].describe().to_string() if len(num_cols) > 0 else "No numeric columns"

    # Categorical info — top 3 values for first 4 cat columns
    cat_cols = df.select_dtypes(include="object").columns[:4]
    cat_parts = []
    for col in cat_cols:
        top = df[col].value_counts().head(3).to_dict()
        cat_parts.append(f"  {col} ({df[col].nunique()} unique): {top}")
    cat_info = "\n".join(cat_parts)

    # Compact sample — first 3 + last 2 rows
    sample = pd.concat([df.head(3), df.tail(2)]).to_string(index=False, max_cols=8)

    return df_hash, df.shape, describe_str, col_info, cat_info, sample


@st.cache_data(show_spinner=False, max_entries=500)
def ask_llm_cached(prompt_hash, prompt):
    """Cache LLM responses — same question = instant answer."""
    llm = get_llm()
    return llm.invoke(prompt)


def find_numeric_column(df, query):
    """Dynamically find which numeric column the user is asking about."""
    q = query.lower()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        if col.lower() in q:
            return col

    for col in numeric_cols:
        for word in col.lower().replace("_", " ").split():
            if len(word) > 2 and word in q:
                return col

    return numeric_cols[0] if numeric_cols else None


def handle_query(df, question, data_summary):
    """Process the user's question and return an answer."""
    q = question.lower().strip()
    chart_generated = False

    # Security: reject extremely long queries
    if len(question) > 2000:
        return "❌ Query too long. Please keep your question under 2000 characters.", chart_generated

    try:
        numeric_col = find_numeric_column(df, question)

        # --- Quick math handlers (FULL dataset, no LLM needed = instant) ---
        if numeric_col:
            if any(kw in q for kw in ["total", "sum"]):
                result = df[numeric_col].sum()
                return f"📈 Total {numeric_col} = **{result:,.2f}**", chart_generated

            if any(kw in q for kw in ["average", "mean", "avg"]):
                result = df[numeric_col].mean()
                return f"📊 Average {numeric_col} = **{result:,.2f}**", chart_generated

            if any(kw in q for kw in ["maximum", "max", "highest", "largest", "top"]):
                result = df[numeric_col].max()
                return f"🏆 Maximum {numeric_col} = **{result:,.2f}**", chart_generated

            if any(kw in q for kw in ["minimum", "min", "lowest", "smallest"]):
                result = df[numeric_col].min()
                return f"📉 Minimum {numeric_col} = **{result:,.2f}**", chart_generated

            if "median" in q:
                result = df[numeric_col].median()
                return f"📊 Median {numeric_col} = **{result:,.2f}**", chart_generated

            if "count" in q and any(kw in q for kw in ["unique", "distinct"]):
                result = df[numeric_col].nunique()
                return f"🔢 Unique values in {numeric_col} = **{result:,}**", chart_generated

            if any(kw in q for kw in ["std", "standard deviation", "deviation"]):
                result = df[numeric_col].std()
                return f"📐 Std Dev of {numeric_col} = **{result:,.2f}**", chart_generated

        # --- Rows/shape queries ---
        if any(kw in q for kw in ["how many rows", "row count", "number of rows", "total rows"]):
            return f"📋 Dataset has **{len(df):,}** rows and **{len(df.columns)}** columns.", chart_generated

        if any(kw in q for kw in ["columns", "fields", "column names"]) and any(kw in q for kw in ["what", "list", "show", "all"]):
            cols = ", ".join(df.columns.tolist())
            return f"📋 Columns: {cols}", chart_generated

        # --- Chart handler ---
        if any(kw in q for kw in ["plot", "chart", "graph", "visualize", "bar"]):
            cat_cols = df.select_dtypes(include="object").columns.tolist()
            group_col = cat_cols[0] if cat_cols else None

            if group_col and numeric_col:
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_facecolor("#0d1117")
                ax.set_facecolor("#161b22")

                data = df.groupby(group_col)[numeric_col].sum().sort_values(ascending=False).head(15)
                ax.bar(data.index, data.values, color="#58a6ff", edgecolor="#1c2333", linewidth=0.5)

                ax.set_title(f"{numeric_col} by {group_col}", color="#e6edf3", fontsize=13, fontweight="bold", pad=12)
                ax.tick_params(colors="#8b949e", rotation=35, labelsize=9)
                for spine in ax.spines.values():
                    spine.set_color("#21262d")
                ax.yaxis.label.set_color("#8b949e")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                chart_generated = True
                return f"📊 Chart: {numeric_col} by {group_col}", chart_generated
            else:
                return "❌ Need at least one categorical and one numeric column for chart.", chart_generated

        # --- LLM handler (full dataset awareness, compact prompt) ---
        prompt = f"""You are a data analyst. Answer based on this dataset summary.

{data_summary}

Question: {question}

Answer concisely with specific numbers."""

        prompt_hash = hash(prompt)
        answer = ask_llm_cached(prompt_hash, prompt)
        return answer, chart_generated

    except Exception as e:
        return f"❌ Error: {str(e)}", chart_generated


# ===============================
# FILE UPLOAD
# ===============================
render_upload_area()
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], label_visibility="collapsed")

# ===============================
# MAIN LOGIC
# ===============================
if uploaded_file:
    # Security: validate file before processing
    is_valid, error_msg = validate_csv_file(uploaded_file)
    if not is_valid:
        st.error(error_msg)
        st.stop()

    df = load_csv(uploaded_file)

    # Prepare data summary (cached)
    summary_inputs = prepare_summary_inputs(df)
    data_summary = build_data_summary(*summary_inputs)

    # Stats dashboard
    render_stats_panel(df)

    # Data preview
    render_data_card(df, title=f"Dataset Preview — showing 10 of {len(df):,} rows", num_rows=10)

    # Chat history (above input)
    render_chat_history(st.session_state.chat_history)

    # Clear chat & download (below chat)
    if st.session_state.chat_history:
        col1, col2 = st.columns([1, 1])
        with col1:
            render_download_report(st.session_state.chat_history)
        with col2:
            if st.button("🗑 Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    # Chat input (bottom, auto-clears)
    question = st.chat_input("Ask anything about your data...")

    if question:
        st.session_state.chat_history.append(("User", question))

        with st.spinner("⚡ Analyzing..."):
            answer, _ = handle_query(df, question, data_summary)

        st.session_state.chat_history.append(("AI", answer))
        st.rerun()