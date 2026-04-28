"""Kōjin — entry point.

Sets the page config, applies the global theme, and dispatches to the two
pages of the app via Streamlit's modern multipage navigation:

- **Bento Maker** (default landing) — composes nutritionally-optimised bentos.
- **Exploration des ingrédients** — natural-language SQL queries over the
  products catalogue, powered by LangChain + Amazon Bedrock + DuckDB.

The pages live in ``app_pages/`` (not ``pages/``) so that Streamlit's legacy
multipage auto-discovery doesn't add the entry script as an extra page.
"""

import streamlit as st

from kojin_common import apply_theme

st.set_page_config(
    page_title="Kōjin — コージン",
    page_icon="◯",
    layout="wide",
)

apply_theme()

bento_maker = st.Page(
    "app_pages/bento_maker.py",
    title="Bento Maker",
    default=True,
)
exploration = st.Page(
    "app_pages/exploration.py",
    title="Exploration des ingrédients",
)

navigation = st.navigation([bento_maker, exploration])
navigation.run()
