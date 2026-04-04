"""
StatBot Pro — UI Components
ChatGPT-inspired UI rendering with security hardening.
"""

import streamlit as st
import html
import os
import re


# ============================================
# SECURITY UTILITIES
# ============================================

def sanitize_html(text: str) -> str:
    """Escape HTML entities to prevent XSS injection in chat messages."""
    return html.escape(str(text))


def sanitize_filename(name: str) -> str:
    """Sanitize uploaded filename to prevent path traversal."""
    # Strip directory components and dangerous characters
    name = os.path.basename(name)
    name = re.sub(r'[^\w\s\-.]', '', name)
    return name


def validate_csv_file(uploaded_file, max_size_mb: int = 150, max_rows: int = 2_000_000):
    """Validate an uploaded CSV file for security and size limits.
    Returns (is_valid, error_message)."""
    # Check file size
    uploaded_file.seek(0, 2)  # Seek to end
    size_bytes = uploaded_file.tell()
    uploaded_file.seek(0)  # Reset to start
    size_mb = size_bytes / (1024 * 1024)

    if size_mb > max_size_mb:
        return False, f"File too large ({size_mb:.1f} MB). Maximum is {max_size_mb} MB."

    if size_mb == 0:
        return False, "File is empty."

    # Check extension
    name = getattr(uploaded_file, 'name', '')
    if not name.lower().endswith('.csv'):
        return False, "Only CSV files are accepted."

    return True, ""


def sanitize_dataframe(df):
    """Sanitize DataFrame column names to prevent injection."""
    import pandas as pd
    # Clean column names: remove special chars, limit length
    clean_cols = {}
    for col in df.columns:
        clean = re.sub(r'[^\w\s\-.]', '', str(col)).strip()
        clean = clean[:50]  # Limit column name length
        if not clean:
            clean = "unnamed"
        clean_cols[col] = clean
    return df.rename(columns=clean_cols)


# ============================================
# UI COMPONENTS
# ============================================

def load_css():
    """Load the external CSS stylesheet."""
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header():
    """Render the hero header."""
    st.markdown("""
    <div class="hero-header">
        <h1>⚡ StatBot Pro</h1>
        <div class="subtitle">AI-Powered Data Analyst</div>
        <div class="tagline">Upload any CSV · Ask anything · Get instant insights</div>
        <div class="security-badge">🔒 Secure · Validated · Cached</div>
    </div>
    """, unsafe_allow_html=True)


def render_upload_area():
    """Render the styled upload prompt."""
    st.markdown("""
    <div class="upload-area">
        <span class="upload-icon">📄</span>
        <div class="upload-text">Drop your CSV dataset below</div>
        <div class="upload-hint">Up to 150 MB · Supports millions of rows</div>
    </div>
    """, unsafe_allow_html=True)


def render_stats_panel(df):
    """Render quick-stats dashboard showing dataset overview."""
    rows, cols = df.shape
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    missing = int(df.isnull().sum().sum())

    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{rows:,}</div>
            <div class="stat-label">Rows</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{cols}</div>
            <div class="stat-label">Columns</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(numeric_cols)}</div>
            <div class="stat-label">Numeric</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(cat_cols)}</div>
            <div class="stat-label">Categorical</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{missing}</div>
            <div class="stat-label">Missing</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_data_card(df, title="📄 Dataset Preview", num_rows=10):
    """Render the dataset preview inside a card."""
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-title">{sanitize_html(title)}</div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df.head(num_rows), use_container_width=True)


def render_chat_message(role, message):
    """Render a single chat message in ChatGPT style."""
    safe_msg = sanitize_html(message).replace("\n", "<br>")

    if role == "User":
        st.markdown(f"""
        <div class="chat-msg user-msg">
            <div class="chat-avatar">You</div>
            <div class="chat-bubble">
                <div class="chat-msg-label">You</div>
                {safe_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-msg bot-msg">
            <div class="chat-avatar">⚡</div>
            <div class="chat-bubble">
                <div class="chat-msg-label">StatBot</div>
                {safe_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_chat_history(chat_history):
    """Render the full chat history in a centered container."""
    if not chat_history:
        return
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for role, msg in chat_history:
        render_chat_message(role, msg)
    st.markdown('</div>', unsafe_allow_html=True)


def render_thinking():
    """Show a thinking/typing indicator."""
    st.markdown("""
    <div class="thinking-indicator">
        <div class="thinking-dots">
            <span></span><span></span><span></span>
        </div>
        StatBot is analyzing...
    </div>
    """, unsafe_allow_html=True)


def render_download_report(chat_history):
    """Render the download report button."""
    if not chat_history:
        return

    report = "━" * 40 + "\n"
    report += "  StatBot Pro — Analysis Report\n"
    report += "━" * 40 + "\n\n"

    for role, msg in chat_history:
        prefix = "You" if role == "User" else "StatBot"
        report += f"{prefix}:\n{msg}\n\n"

    st.download_button(
        "📥 Download Report",
        report,
        file_name="statbot_report.txt",
        mime="text/plain"
    )
