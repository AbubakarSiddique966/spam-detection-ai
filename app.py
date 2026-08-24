"""
SpamGuard AI - Streamlit Frontend

Modern UI for the FastAPI-based SMS Spam Detection backend.

The Streamlit app does NOT run the ML model.
It communicates with the deployed FastAPI API over HTTP.
"""

import streamlit as st
import requests


# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = "https://spam-detection-ai.fastapicloud.dev"

PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
HEALTH_ENDPOINT = f"{API_BASE_URL}/"

REQUEST_TIMEOUT = 15


EXAMPLE_MESSAGES = {
    "🎯 Classic Spam": (
        "URGENT! Your mobile number has WON a £2000 cash prize. "
        "Call 09061701461 now to claim before it expires!"
    ),
    "🎁 Prize / Lottery": (
        "Congratulations! You have won a free prize. "
        "Click this link to claim your reward."
    ),
    "📅 Legitimate Appointment": (
        "Hi, this is a reminder that your dentist appointment "
        "is scheduled for tomorrow at 10:30 AM."
    ),
    "💬 Normal Conversation": (
        "Hey, are we still on for lunch tomorrow? "
        "Let me know what time works for you."
    ),
}


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# SESSION STATE
# ============================================================================

if "sms_text" not in st.session_state:
    st.session_state.sms_text = ""

if "result" not in st.session_state:
    st.session_state.result = None

if "error" not in st.session_state:
    st.session_state.error = None


# ============================================================================
# CUSTOM CSS
# ============================================================================

def load_css():

    st.markdown(
        """
        <style>

        /* ================================================================
           GLOBAL
        ================================================================ */

        .stApp {
            background: linear-gradient(
                180deg,
                #f7f8fc 0%,
                #eef1f8 100%
            );
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1100px;
        }

        html,
        body,
        [class*="css"] {
            font-family:
                "Inter",
                "Segoe UI",
                -apple-system,
                BlinkMacSystemFont,
                sans-serif;
        }


        /* ================================================================
           HERO
        ================================================================ */

        .hero-title {
            font-size: 2.7rem;
            font-weight: 800;
            color: #1a1a2e;
            margin-bottom: 0.2rem;
            letter-spacing: -0.03em;
        }

        .hero-subtitle {
            font-size: 1.15rem;
            font-weight: 600;
            color: #4b5eaa;
            margin-bottom: 0.6rem;
        }

        .hero-description {
            font-size: 0.98rem;
            color: #5b5f77;
            max-width: 720px;
            line-height: 1.6;
        }


        /* ================================================================
           CARDS
        ================================================================ */

        .info-card {
            background: #ffffff;
            border-radius: 14px;
            padding: 1.2rem 1rem;
            text-align: center;
            border: 1px solid #eaebf3;
            box-shadow: 0 2px 10px rgba(30, 34, 90, 0.05);
            height: 100%;
        }

        .info-card-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #8a8fa8;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }

        .info-card-value {
            font-size: 1.05rem;
            font-weight: 700;
            color: #1a1a2e;
        }


        /* ================================================================
           RESULT CARDS
        ================================================================ */

        .result-card {
            border-radius: 16px;
            padding: 1.6rem 1.8rem;
            margin-top: 1rem;
            border: 1px solid;
        }

        .result-spam {
            background: linear-gradient(
                135deg,
                #fff1f1 0%,
                #ffe4e4 100%
            );
            border-color: #f5b5b5;
        }

        .result-ham {
            background: linear-gradient(
                135deg,
                #f0fbf4 0%,
                #e2f8ea 100%
            );
            border-color: #a9e3bf;
        }

        .result-title {
            font-size: 1.5rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }

        .result-title-spam {
            color: #c62828;
        }

        .result-title-ham {
            color: #1f8b4c;
        }

        .result-explanation {
            font-size: 0.98rem;
            color: #3d3d4e;
            margin-bottom: 0.6rem;
        }

        .result-meta {
            font-size: 0.85rem;
            color: #6c6f88;
            border-top: 1px solid rgba(0, 0, 0, 0.06);
            padding-top: 0.6rem;
            margin-top: 0.6rem;
        }


        /* ================================================================
           SECTION HEADERS
        ================================================================ */

        .section-header {
            font-size: 1.25rem;
            font-weight: 700;
            color: #1a1a2e;
            margin-top: 2rem;
            margin-bottom: 0.6rem;
        }


        /* ================================================================
           BUTTONS
        ================================================================ */

        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            font-size: 1rem;
            padding: 0.65rem 1.4rem;
            transition: all 0.15s ease;
        }


        /* ================================================================
           TEXT AREA
        ================================================================ */

        textarea {
            border-radius: 12px !important;
        }


        /* ================================================================
           SIDEBAR
        ================================================================ */

        section[data-testid="stSidebar"] {
            background: #1a1a2e;
        }

        section[data-testid="stSidebar"] * {
            color: #eaeaf5 !important;
        }

        section[data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.15);
        }


        /* ================================================================
           STATUS
        ================================================================ */

        .status-badge {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .status-online {
            background: rgba(46, 204, 113, 0.15);
            color: #2ecc71;
        }

        .status-offline {
            background: rgba(231, 76, 60, 0.15);
            color: #e74c3c;
        }


        /* ================================================================
           CHARACTER COUNTER
        ================================================================ */

        .char-counter {
            font-size: 0.8rem;
            color: #8a8fa8;
            text-align: right;
            margin-top: -0.6rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# API FUNCTIONS
# ============================================================================

def check_api_status() -> bool:
    """
    Check whether the deployed FastAPI backend is reachable.
    """

    try:
        response = requests.get(
            HEALTH_ENDPOINT,
            timeout=10,
        )

        return response.status_code == 200

    except requests.exceptions.RequestException:
        return False


def predict_spam(text: str):
    """
    Send SMS text to the FastAPI /predict endpoint.

    Returns:
        result_dict, error_message
    """

    try:

        response = requests.post(
            PREDICT_ENDPOINT,
            json={
                "text": text
            },
            timeout=REQUEST_TIMEOUT,
        )

    except requests.exceptions.ConnectionError:

        return None, (
            "🔴 Could not connect to the Spam Detection API. "
            "Please try again in a few moments."
        )

    except requests.exceptions.Timeout:

        return None, (
            "⏳ The API request timed out. "
            "Please try again."
        )

    except requests.exceptions.RequestException as error:

        return None, (
            f"⚠️ Network error occurred: {error}"
        )


    # ------------------------------------------------------------------------
    # Successful response
    # ------------------------------------------------------------------------

    if response.status_code == 200:

        try:
            data = response.json()

        except ValueError:

            return None, (
                "⚠️ The API returned an invalid response."
            )

        if "prediction" not in data:

            return None, (
                "⚠️ The API response is missing "
                "the prediction field."
            )

        return data, None


    # ------------------------------------------------------------------------
    # Client error
    # ------------------------------------------------------------------------

    if response.status_code == 400:

        return None, (
            "⚠️ Invalid request. "
            "Please check the SMS message."
        )


    # ------------------------------------------------------------------------
    # Server error
    # ------------------------------------------------------------------------

    if response.status_code >= 500:

        return None, (
            "🔴 The Spam Detection API encountered "
            "an internal error."
        )


    # ------------------------------------------------------------------------
    # Other status codes
    # ------------------------------------------------------------------------

    return None, (
        f"⚠️ Unexpected API response: "
        f"HTTP {response.status_code}"
    )


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():

    st.markdown(
        """
        <div class="hero-title">
            📩 SpamGuard AI
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-subtitle">
            AI-Powered SMS Spam Detection
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-description">
            SpamGuard AI uses an XGBoost machine-learning model
            trained on TF-IDF text features to classify SMS messages
            as <b>Spam</b> or <b>Ham</b>.
            Enter an SMS message below and let the model analyze it.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")


def render_info_cards():

    st.markdown(
        """
        <div class="section-header">
            📊 Model Overview
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = [
        ("Model", "XGBoost"),
        ("Features", "TF-IDF"),
        ("Task", "Binary Classification"),
        ("Backend", "FastAPI"),
    ]

    columns = st.columns(4)

    for column, (label, value) in zip(columns, cards):

        with column:

            st.markdown(
                f"""
                <div class="info-card">

                    <div class="info-card-label">
                        {label}
                    </div>

                    <div class="info-card-value">
                        {value}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


def render_examples():

    st.markdown(
        """
        <div class="section-header">
            💡 Try an Example
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(len(EXAMPLE_MESSAGES))

    for column, (label, message) in zip(
        columns,
        EXAMPLE_MESSAGES.items()
    ):

        with column:

            if st.button(
                label,
                use_container_width=True,
                key=f"example_{label}",
            ):

                st.session_state.sms_text = message
                st.session_state.result = None
                st.session_state.error = None

                st.rerun()


def render_input():

    st.markdown(
        """
        <div class="section-header">
            ✉️ Enter your SMS message
        </div>
        """,
        unsafe_allow_html=True,
    )

    text = st.text_area(
        "Enter your SMS message",
        value=st.session_state.sms_text,
        placeholder=(
            "Congratulations! You have won a free prize. "
            "Click this link to claim your reward."
        ),
        height=140,
        label_visibility="collapsed",
        key="sms_input_area",
    )

    st.session_state.sms_text = text

    character_count = len(text)

    st.markdown(
        f"""
        <div class="char-counter">
            {character_count} characters
        </div>
        """,
        unsafe_allow_html=True,
    )

    analyze_clicked = st.button(
        "🔍 Analyze Message",
        use_container_width=True,
    )

    return text, analyze_clicked


def display_result(result: dict):

    prediction = str(
        result.get("prediction", "")
    ).strip().lower()

    input_text = result.get(
        "input_text",
        st.session_state.sms_text,
    )

    # ============================================================
    # SPAM RESULT
    # ============================================================

    if prediction == "spam":

        st.markdown(
            """
            <div class="result-card result-spam">

                <div class="result-title result-title-spam">
                    🚨 SPAM DETECTED
                </div>

                <div class="result-explanation">
                    This message appears to be a spam message.
                </div>

                <div class="result-meta">
                    <b>Prediction:</b> Spam
                    &nbsp; | &nbsp;
                    <b>Model:</b> XGBoost + TF-IDF
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # ============================================================
    # HAM RESULT
    # ============================================================

    elif prediction == "ham":

        st.markdown(
            """
            <div class="result-card result-ham">

                <div class="result-title result-title-ham">
                    ✅ HAM / SAFE
                </div>

                <div class="result-explanation">
                    This message appears to be a legitimate SMS.
                </div>

                <div class="result-meta">
                    <b>Prediction:</b> Ham
                    &nbsp; | &nbsp;
                    <b>Model:</b> XGBoost + TF-IDF
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # ============================================================
    # UNKNOWN RESULT
    # ============================================================

    else:

        st.warning(
            f"⚠️ Unknown prediction returned by API: `{prediction}`"
        )

    # ============================================================
    # ANALYZED MESSAGE
    # ============================================================

    with st.expander("View analyzed message"):
        st.write(input_text)


def render_sidebar(api_online: bool):

    with st.sidebar:

        st.markdown("## 📩 SpamGuard AI")

        st.caption(
            "AI-Powered SMS Spam Detection"
        )

        st.markdown("---")


        # --------------------------------------------------------------------
        # About
        # --------------------------------------------------------------------

        st.markdown("### ℹ️ About")

        st.write(
            "SpamGuard AI classifies SMS messages as "
            "**Spam** or **Ham** using an XGBoost machine-learning "
            "model served through a FastAPI backend."
        )


        # --------------------------------------------------------------------
        # Technology Stack
        # --------------------------------------------------------------------

        st.markdown("### 🛠️ Technology Stack")

        st.markdown(
            """
            - Python
            - FastAPI
            - Streamlit
            - XGBoost
            - TF-IDF
            - Scikit-learn
            """
        )


        # --------------------------------------------------------------------
        # API Status
        # --------------------------------------------------------------------

        st.markdown("### 🔌 API Status")

        if api_online:

            st.markdown(
                """
                <span class="status-badge status-online">
                    🟢 API Online
                </span>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <span class="status-badge status-offline">
                    🔴 API Offline
                </span>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "The production API is currently unavailable. "
                "Please try again later."
            )


        # --------------------------------------------------------------------
        # API Information
        # --------------------------------------------------------------------

        st.markdown("### 🌐 API Information")

        st.code(
            f"""
Base URL:
{API_BASE_URL}

Predict:
POST /predict

Docs:
{API_BASE_URL}/docs
            """,
            language="text",
        )


        st.markdown("---")

        st.caption(
            "Built with Streamlit · "
            "Powered by FastAPI + XGBoost"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    # Load CSS
    load_css()


    # Check production API
    api_online = check_api_status()


    # Sidebar
    render_sidebar(api_online)


    # Main UI
    render_header()

    render_info_cards()

    render_examples()

    text, analyze_clicked = render_input()


    # ------------------------------------------------------------------------
    # ANALYZE BUTTON
    # ------------------------------------------------------------------------

    if analyze_clicked:

        st.session_state.result = None
        st.session_state.error = None


        # Empty input
        if not text or not text.strip():

            st.session_state.error = (
                "⚠️ Please enter an SMS message "
                "before analyzing."
            )


        # API offline
        elif not api_online:

            st.session_state.error = (
                "🔴 The Spam Detection API is currently "
                "unavailable. Please try again in a few moments."
            )


        # API online
        else:

            with st.spinner(
                "🤖 Analyzing your message..."
            ):

                result, error = predict_spam(
                    text.strip()
                )

            st.session_state.result = result
            st.session_state.error = error


    # ------------------------------------------------------------------------
    # DISPLAY ERROR
    # ------------------------------------------------------------------------

    if st.session_state.error:

        st.error(
            st.session_state.error
        )


    # ------------------------------------------------------------------------
    # DISPLAY RESULT
    # ------------------------------------------------------------------------

    elif st.session_state.result:

        display_result(
            st.session_state.result
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()