# app/ui/search_app.py — multipage demo entry point.

import os
import sys

import streamlit as st

# Add the project root to the Python path so 'core' / 'ingestion_pipeline'
# imports work when Streamlit runs this file directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.logger import setup_logging  # noqa: E402

setup_logging()

st.set_page_config(
    page_title="Semantic Video Search — Demo",
    layout="wide",
    page_icon="🎬",
)


def _page(relative_path: str, title: str, icon: str) -> "st.Page":
    return st.Page(
        os.path.join(os.path.dirname(__file__), "pages", relative_path),
        title=title,
        icon=icon,
    )


navigation = st.navigation(
    {
        "Demo": [
            _page("home.py", "Home", "🏠"),
            _page("submit.py", "1 · Submit", "📤"),
            _page("pipeline.py", "2 · Pipeline", "⚙️"),
            _page("search.py", "3 · Search", "🔍"),
        ],
    }
)
navigation.run()
