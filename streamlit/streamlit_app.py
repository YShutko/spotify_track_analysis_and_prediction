import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# Optional Spotify API use
try:
    import spotipy
    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False


st.set_page_config(
    page_title="Spotify Popularity Project",
    layout="wide",
    page_icon="🎧",
)


# ---------------------------
# 1. DATA LOADING
# ---------------------------

def map_macro_genre(g: str) -> str:
    g = str(g).lower()
    if "pop" in g:
        return "Pop"
    elif "rock" in g:
        return "Rock"
    elif "hip hop" in g or "rap" in g or "trap" in g:
        return "Hip-Hop/Rap"
    elif "r&b" in g or "soul" in g:
        return "R&B/Soul"
    elif any(x in g for x in ["electro", "techno", "house", "edm", "dance"]):
        return "Electronic/Dance"
    elif any(x in g for x in ["metal", "hardcore"]):
        return "Metal/Hardcore"
    elif any(x in g for x in ["jazz", "blues"]):
        return "Jazz/Blues"
    elif any(x in g for x in ["classical", "orchestra", "piano"]):
        return "Classical"
    elif any(x in g for x in ["latin", "reggaeton", "sertanejo", "samba"]):
        return "Latin"
    elif "country" in g:
        return "Country"
    elif any(x in g for x in ["folk", "singer-songwriter"]):
        return "Folk"
    elif any(x in g for x in ["indie", "alternative"]):
        return "Indie/Alternative"
    return "Other"


@st.cache_data
def load_data(filename="spotify_cleaned_data.csv"):
    here = os.path.dirname(__file__)
    search_paths = [
        os.path.join(here, filename),
        filename,
    ]

    for path in search_paths:
        if os.path.isfile(path):
            st.success(f"Loaded dataset: {os.path.relpath(path)}")
            df = pd.read_csv(path)

            if "track_genre" in df.columns:
                df["macro_genre"] = df["track_genre"].apply(map_macro_genre)
            else:
                df["macro_genre"] = "Other"

            if "explicit" in df.columns:
                df["explicit"] = df["explicit"].astype(bool)

            return df

    st.error("❌ CSV file not found!")
    raise FileNotFoundError


# ---------------------------
# 2. MODEL LOADING
# ---------------------------

@st.cache_resource
def load_models():
    REPO = "YShutko/spotify-popularity-models"
    model_files = {
        "Random Forest": "random_forest_model.pkl",
        "XGBoost (Tuned)": "xgb_model_best.pkl",
        "Linear Regression": "linear_regression_model.pkl",
    }

    models = {}
    for name, f in model_files.items():
        try:
            p = hf_hub_download(repo_id=REPO, filename=f)
            models[name] = joblib.load(p)
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
    return models


# ---------------------------
# 3. MAIN APP
# ---------------------------

def main():
    df = load_data()

    st.sidebar.title("🎧 Spotify Popularity App")
    theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

    global_min_pop = st.sidebar.slider("Global min popularity", 0, 100, 0)

    if theme == "Dark":
        st.markdown("<style>.stApp{background:#0e1117;color:white}</style>", unsafe_allow_html=True)

    df_filtered = df[df["popularity"] >= global_min_pop]

    tab1, tab2, tab3 = st.tabs(["📁 Dataset", "🤖 Prediction", "🎶 Playlist Builder"])

    # ---------------------------
    # TAB 1 — DATASET
    # ---------------------------
    with tab1:
        st.title("📁 Dataset Overview")
        st.dataframe(df_filtered, use_container_width=True)

    # ---------------------------
    # TAB 2 — ML PREDICTION
    # ---------------------------
    with tab2:
        st.title("🤖 Predict Popularity")

        models = load_models()
        if not models:
            st.error("No models could be loaded.")
            return

        model_choice = st.selectbox("Choose a model", list(models.keys()))
        energy = st.slider("Energy", 0.0, 1.0, 0.6)
        dance = st.slider("Danceability", 0.0, 1.0, 0.6)

        if st.button("Predict"):
            sample = pd.DataFrame([{
                "energy": energy,
                "danceability": dance,
                "instrumentalness": 0,
                "acousticness": 0,
                "liveness": 0,
                "valence": 0.5,
                "tempo": 120,
                "loudness": -8,
                "duration_min": 3,
            }])
            pred = models[model_choice].predict(sample)[0]
            st.success(f"Predicted popularity: {pred:.1f}")

    # ---------------------------
    # TAB 3 — PLAYLIST BUILDER
    # ---------------------------
    with tab3:
        st.title("🎶 Playlist Builder with Filters")

        # PERSONAL PLAYLIST (session state)
        if "my_playlist" not in st.session_state:
            st.session_state.my_playlist = pd.DataFrame()

        st.subheader("🔎 Filter tracks by audio features")

        # AVAILABLE SLIDER FILTERS
        numeric_cols = [
            "danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "duration_min"
        ]

        filters = {}
        for col in numeric_cols:
            if col in df_filtered.columns:
                min_val, max_val = float(df_filtered[col].min()), float(df_filtered[col].max())
                filters[col] = st.slider(f"{col}", min_val, max_val, (min_val, max_val))

        # FILTER DATAFRAME
        filtered_df = df_filtered.copy()
        for col, (a, b) in filters.items():
            filtered_df = filtered_df[(filtered_df[col] >= a) & (filtered_df[col] <= b)]

        st.write(f"### {len(filtered_df)} tracks match your filters")
        st.dataframe(filtered_df[["track_name", "artists", "popularity"]], use_container_width=True)

        # SELECT TRACKS TO ADD
        st.subheader("➕ Add track to your playlist")

        track_to_add = st.selectbox(
            "Choose a track",
            filtered_df["track_name"] + " — " + filtered_df["artists"]
        )

        if st.button("Add to my playlist"):
            row = filtered_df.iloc[
                (filtered_df["track_name"] + " — " + filtered_df["artists"] == track_to_add).idxmax()
            ]
            st.session_state.my_playlist = pd.concat(
                [st.session_state.my_playlist, pd.DataFrame([row])],
                ignore_index=True
            )
            st.success("Added!")

        # SHOW PLAYLIST
        st.subheader("🎧 My Playlist")
        if len(st.session_state.my_playlist) == 0:
            st.info("Your playlist is empty. Add some tracks!")
        else:
            st.dataframe(
                st.session_state.my_playlist[["track_name", "artists", "popularity"]],
                use_container_width=True
            )


if __name__ == "__main__":
    main()
