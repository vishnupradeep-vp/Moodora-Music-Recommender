"""
app.py
------
Music Mood Finder for mood-based music discovery.
Allows users to enter any mood or feeling and instantly discover
curated English and Malayalam songs side-by-side with a music-inspired interface
and direct Spotify links.
"""

import html
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote, urlencode
import pandas as pd
import requests
import streamlit as st


from src.preprocess import load_songs
from src.recommend import MoodRecommender, MOOD_PROFILES

DATA_DIR = Path(__file__).resolve().parent / "data"
LIKED_SONGS_PATH = DATA_DIR / "liked_songs.json"


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def find_original_artwork(title, artist, album):
    """Find matching album art in Apple's public music catalog."""
    artist_names = [name.strip().casefold() for name in str(artist).split(",")]
    primary_artist = artist_names[0] if artist_names else str(artist)
    normalized_title = str(title).casefold().strip()
    normalized_album = str(album).casefold().strip()
    best_result = None
    best_score = 0.0

    search_terms = dict.fromkeys((f"{title} {primary_artist}", str(title)))
    for term in search_terms:
        params = urlencode({
            "term": term,
            "entity": "song",
            "limit": 25,
            "country": "IN",
        })
        try:
            response = requests.get(
                f"https://itunes.apple.com/search?{params}",
                headers={"User-Agent": "MusicMoodFinder/1.0"},
                timeout=(2, 3),
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except (requests.RequestException, ValueError) as exc:
            logging.warning("Could not look up album artwork for %r: %s", title, exc)
            continue

        for result in results:
            track_name = str(result.get("trackName", "")).casefold().strip()
            track_name = track_name.split(" (from", 1)[0].strip()
            result_artists = [
                name.strip().casefold()
                for name in str(result.get("artistName", "")).replace("&", ",").split(",")
            ]
            collection_name = str(result.get("collectionName", "")).casefold().strip()
            title_score = SequenceMatcher(None, normalized_title, track_name).ratio()
            artist_score = max((
                SequenceMatcher(None, requested, candidate).ratio()
                for requested in artist_names
                for candidate in result_artists
            ), default=0.0)
            album_score = SequenceMatcher(None, normalized_album, collection_name).ratio()
            album_matches = bool(normalized_album and normalized_album in collection_name)
            score = title_score * 0.65 + artist_score * 0.25 + album_score * 0.10

            minimum_title_score = 0.80 if album_matches else 0.86
            if title_score >= minimum_title_score and artist_score >= 0.75 and score > best_score:
                best_result = result
                best_score = score
        if best_result:
            break

    artwork_url = best_result.get("artworkUrl100") if best_result else None
    if artwork_url:
        return artwork_url.replace("100x100bb", "600x600bb")

    logging.warning("No matching original album artwork found for %r by %r", title, artist)
    return None


def add_original_artwork(recommendations):
    """Resolve album covers concurrently for the songs shown in a view."""
    rows = list(recommendations.iterrows())
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(
                find_original_artwork,
                row["title"],
                row["artist"],
                row.get("album_movie", ""),
            ): index
            for index, row in rows
        }
        for future, index in futures.items():
            recommendations.at[index, "original_artwork_url"] = future.result()
    return recommendations


def artwork_placeholder(title):
    initial = html.escape(str(title)[:1].upper() or "♪")
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 160">'
        '<rect width="160" height="160" rx="12" fill="#101A30"/>'
        '<circle cx="80" cy="80" r="54" fill="#8265F5"/>'
        f'<text x="80" y="98" text-anchor="middle" font-family="Arial" '
        f'font-size="52" font-weight="700" fill="#F4F6FF">{initial}</text>'
        "</svg>"
    )
    return f"data:image/svg+xml,{quote(svg)}"


def load_liked_song_ids():
    if not LIKED_SONGS_PATH.exists():
        return []

    try:
        liked_ids = json.loads(LIKED_SONGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not load your saved library ({exc}).")
        st.stop()

    if not isinstance(liked_ids, list) or any(
        not isinstance(song_id, int) or isinstance(song_id, bool)
        for song_id in liked_ids
    ):
        st.error(f"Your saved library has an invalid format: {LIKED_SONGS_PATH}")
        st.stop()

    return list(dict.fromkeys(liked_ids))


def save_liked_song_ids(liked_ids):
    try:
        LIKED_SONGS_PATH.write_text(
            json.dumps(liked_ids, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        st.error(f"Could not save your library ({exc}).")
        return False
    return True


# Page Configuration
st.set_page_config(
    page_title="Music Mood Finder — Discover Music for Your Mood",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Music Mood Finder styling
if "light_mode" not in st.session_state:
    st.session_state["light_mode"] = False

theme_tokens = (
    """
    --app-bg: #080d1e;
    --surface: #101a30;
    --surface-hover: #172440;
    --surface-solid: #0d162b;
    --text: #f4f6ff;
    --muted: #a0acc5;
    --subtle: #7d8ba8;
    --border: rgba(144, 165, 215, 0.14);
    --accent: #8265f5;
    --blue: #3478ed;
    --neon: #16c79a;
    --spotify-green: #1ed760;
    --success: #18b783;
    --warning: #f2a900;
    --danger: #ef5261;
    --shadow: rgba(1, 5, 18, 0.42);
    """
    if not st.session_state["light_mode"]
    else """
    --app-bg: #f2f5fb;
    --surface: #ffffff;
    --surface-hover: #f7f8fd;
    --surface-solid: #ffffff;
    --text: #17213a;
    --muted: #53627e;
    --subtle: #71809a;
    --border: rgba(54, 72, 119, 0.14);
    --accent: #6748e8;
    --blue: #2868d7;
    --neon: #078968;
    --spotify-green: #16883d;
    --success: #078968;
    --warning: #b87900;
    --danger: #d83e50;
    --shadow: rgba(30, 45, 82, 0.12);
    """
)

APP_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Circular+Std:wght@400;700;900&family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.stApp {{
    --app-bg: #080d1e;
    {theme_tokens}
    color: var(--text);
    background:
        radial-gradient(ellipse at 12% 0%, color-mix(in srgb, var(--accent) 17%, transparent), transparent 42%),
        radial-gradient(ellipse at 92% 8%, color-mix(in srgb, var(--blue) 13%, transparent), transparent 38%),
        var(--app-bg);
    background-attachment: fixed;
    min-height: 100vh;
    transition: background-color .35s ease, color .25s ease;
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: -40%;
    z-index: 0;
    pointer-events: none;
    background: conic-gradient(from 120deg at 50% 50%,
        transparent, color-mix(in srgb, var(--accent) 7%, transparent),
        transparent, color-mix(in srgb, var(--blue) 6%, transparent), transparent);
    filter: blur(90px);
    opacity: .35;
    animation: aurora-drift 34s ease-in-out infinite alternate;
}}

@keyframes aurora-drift {{
    from {{ transform: translate3d(-2%, -1%, 0) rotate(-8deg); }}
    to {{ transform: translate3d(2%, 1%, 0) rotate(8deg); }}
}}

.stAppViewContainer {{ position: relative; z-index: 1; background: transparent; }}
.block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
h1, h2, h3, p, label, [data-testid="stCaptionContainer"] {{ color: var(--text); }}

.brand-title {{
    font-size: 2.2rem; font-weight: 850; color: var(--text);
    letter-spacing: -1px; display: flex; align-items: center; gap: 12px; margin-bottom: 2px;
}}
.brand-green {{ color: var(--spotify-green); }}
.sub-header {{ color: var(--muted); font-size: 1rem; margin-bottom: 1.5rem; }}
.section-hint {{ margin-top: 10px; margin-bottom: 6px; color: var(--muted); font-size: .9rem; }}
.section-heading {{ margin: 0; color: var(--text); }}
.studio-hero {{
    position: relative; overflow: hidden; display: flex; justify-content: space-between;
    align-items: center; gap: 24px; min-height: 220px; margin: 8px 0 26px;
    padding: clamp(24px, 4vw, 44px); border: 1px solid var(--border); border-radius: 24px;
    background:
      radial-gradient(ellipse at 88% 18%, color-mix(in srgb, var(--blue) 31%, transparent), transparent 36%),
      radial-gradient(ellipse at 16% 120%, color-mix(in srgb, var(--accent) 26%, transparent), transparent 48%),
      linear-gradient(120deg, var(--surface-solid), color-mix(in srgb, var(--surface-solid) 70%, var(--app-bg)));
    box-shadow: 0 20px 60px var(--shadow);
}}
.studio-hero-copy {{ position: relative; z-index: 1; max-width: 700px; }}
.studio-eyebrow {{
    color: var(--accent); font-size: .72rem; font-weight: 800; letter-spacing: .18em;
    text-transform: uppercase; margin-bottom: 12px;
}}
.studio-title {{
    color: var(--text); font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1.04;
    letter-spacing: -.055em; font-weight: 850; margin: 0;
}}
.studio-title span {{
    background: linear-gradient(100deg, var(--accent), var(--blue));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.studio-copy {{ color: var(--muted); font-size: 1rem; line-height: 1.6; max-width: 560px; margin: 14px 0 18px; }}
.studio-pills {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.studio-pill {{
    display: inline-flex; align-items: center; gap: 7px; border: 1px solid var(--border);
    border-radius: 999px; padding: 7px 11px; background: color-mix(in srgb, var(--surface) 72%, transparent);
    color: var(--muted); font-size: .78rem; font-weight: 650;
}}
.studio-pill-live {{ border-color: color-mix(in srgb, var(--spotify-green) 32%, var(--border)); }}
.studio-pill-live strong {{ color: var(--spotify-green); font-weight: 750; }}
.spotify-dot {{
    display: inline-block; width: 7px; height: 7px; flex: 0 0 7px; border-radius: 50%;
    background: var(--spotify-green); box-shadow: 0 0 10px color-mix(in srgb, var(--spotify-green) 60%, transparent);
}}
.hero-art {{
    position: relative; display: flex; align-items: center; justify-content: center;
    flex: 0 0 190px; width: 190px; height: 190px; border-radius: 50%;
    background: radial-gradient(circle, color-mix(in srgb, var(--accent) 20%, transparent), transparent 69%);
}}
.hero-disc {{
    display: flex; align-items: center; justify-content: center; gap: 5px; width: 142px; height: 142px;
    border: 1px solid color-mix(in srgb, var(--accent) 38%, var(--border)); border-radius: 50%;
    background: radial-gradient(circle at 35% 25%, color-mix(in srgb, var(--blue) 30%, var(--surface-solid)), var(--surface-solid) 70%);
    box-shadow: 0 0 60px color-mix(in srgb, var(--accent) 17%, transparent), inset 0 0 28px color-mix(in srgb, var(--blue) 12%, transparent);
}}
.hero-disc i {{ display: block; width: 5px; height: var(--bar-height); border-radius: 999px; background: linear-gradient(var(--blue), var(--accent)); animation: waveform 1.05s ease-in-out infinite alternate; }}
.hero-disc i:nth-child(2n) {{ animation-delay: -.5s; }}
.hero-disc i:nth-child(3n) {{ animation-delay: -.8s; }}
@keyframes waveform {{ from {{ transform: scaleY(.5); opacity: .7; }} to {{ transform: scaleY(1); opacity: 1; }} }}
.discovery-heading {{ margin: 4px 0 14px; }}
.discovery-heading h2 {{ color: var(--text); font-size: 1.4rem; margin: 0; letter-spacing: -.03em; }}
.discovery-heading p {{ color: var(--muted); font-size: .9rem; margin: 4px 0 0; }}
.song-match-track {{ height: 4px; overflow: hidden; background: color-mix(in srgb, var(--text) 12%, transparent); border-radius: 999px; margin-top: 10px; }}
.song-match-fill {{ height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--accent), var(--blue)); }}
.results-label {{ color: var(--muted); font-size: .82rem; margin-top: 5px; }}

.spotify-card {{
    background: linear-gradient(145deg, color-mix(in srgb, var(--surface) 94%, var(--blue)), var(--surface) 62%);
    border-radius: 18px; padding: 15px; margin-bottom: 12px;
    border: 1px solid var(--border); transition: transform .22s ease, background .22s ease, box-shadow .22s ease;
    box-shadow: 0 12px 28px var(--shadow);
}}
.spotify-card:hover {{
    background: var(--surface-hover); border-color: color-mix(in srgb, var(--accent) 44%, var(--border));
    transform: translateY(-4px); box-shadow: 0 18px 36px var(--shadow);
}}
.recommendation-heading {{
    display: flex; align-items: center; justify-content: space-between; gap: 10px;
    margin: 4px 0 14px; padding: 0 2px;
}}
.recommendation-heading h3 {{ color: var(--text); font-size: 1.02rem; font-weight: 750; margin: 0; }}
.recommendation-heading p {{ color: var(--muted); font-size: .75rem; margin: 3px 0 0; }}
.recommendation-count {{
    flex: 0 0 auto; border: 1px solid var(--border); background: var(--surface);
    border-radius: 999px; color: var(--muted); font-size: .72rem; font-weight: 700; padding: 5px 9px;
}}
.song-card-layout {{ display: flex; gap: 16px; align-items: flex-start; }}
.song-artwork {{
    width: 108px; height: 108px; flex: 0 0 108px; border-radius: 12px;
    object-fit: cover; box-shadow: 0 8px 20px var(--shadow);
}}
.song-copy {{ flex: 1; min-width: 0; }}
.song-title-row {{ display: flex; justify-content: space-between; gap: 8px; align-items: flex-start; }}
.song-title {{ font-size: 1rem; font-weight: 750; color: var(--text); line-height: 1.35; }}
.song-native-title {{ color: var(--neon); }}
.song-artist {{ color: var(--muted); font-size: .88rem; margin-top: 4px; }}
.song-meta {{ color: var(--subtle); font-size: .78rem; margin-top: 3px; }}
.song-description {{ color: var(--muted); font-size: .8rem; line-height: 1.5; margin: 9px 0 7px; }}
.song-tags {{ display: flex; flex-wrap: wrap; gap: 5px; }}
.badge-match, .badge-era {{
    background: color-mix(in srgb, var(--neon) 13%, transparent); color: var(--neon);
    border: 1px solid color-mix(in srgb, var(--neon) 30%, transparent);
    padding: 3px 8px; border-radius: 999px; font-size: .73rem; font-weight: 700; display: inline-block;
}}
.badge-lang {{
    background: color-mix(in srgb, var(--blue) 14%, transparent); color: var(--blue);
    padding: 3px 9px; border-radius: 999px; font-size: .75rem; font-weight: 650; display: inline-block; margin-left: 6px;
}}
.badge-tag {{
    background: color-mix(in srgb, var(--text) 7%, transparent); color: var(--subtle);
    font-size: .72rem; padding: 3px 8px; border-radius: 999px; display: inline-block;
}}
.mood-banner {{
    background: linear-gradient(115deg, color-mix(in srgb, var(--accent) 15%, var(--surface)), var(--surface) 65%);
    border: 1px solid var(--border); border-radius: 18px;
    padding: 18px 20px; margin: 20px 0; display: flex; justify-content: space-between;
    align-items: center; flex-wrap: wrap; gap: 16px;
}}
.mood-banner-title {{ font-size: 1.12rem; font-weight: 800; color: var(--text); }}
.mood-banner-value {{ color: var(--neon); }}
.mood-banner-profile {{ display: grid; grid-template-columns: repeat(4, minmax(76px, 1fr)); gap: 8px; margin-top: 12px; }}
.mood-stat {{ min-width: 76px; }}
.mood-stat-label {{ color: var(--muted); font-size: .68rem; font-weight: 650; text-transform: uppercase; letter-spacing: .07em; }}
.mood-stat-value {{ color: var(--text); font-size: .82rem; font-weight: 750; margin-top: 2px; }}
.mood-stat-track {{ height: 3px; margin-top: 5px; background: color-mix(in srgb, var(--text) 12%, transparent); border-radius: 999px; overflow: hidden; }}
.mood-stat-track span {{ display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--accent), var(--blue)); }}
.mood-bpm {{ display: flex; height: 100%; align-items: center; color: var(--text); font-size: .82rem; font-weight: 750; }}
.mood-status {{ white-space: nowrap; }}

/* Language view segmented control */
.stRadio [data-testid="stRadioGroup"] {{
    display: flex; gap: 6px; padding: 5px; background: var(--surface);
    border: 1px solid var(--border); border-radius: 12px;
}}
.stRadio [data-testid="stRadioOption"] {{
    flex: 1 1 0; min-height: 44px; justify-content: center; padding: 9px 12px;
    border: 1px solid transparent; border-radius: 10px; color: var(--muted);
    background: transparent; cursor: pointer; transition: background .18s ease, color .18s ease;
}}
.stRadio [data-testid="stRadioOption"] > div > div > div:first-child {{ display: none; }}
.stRadio [data-testid="stRadioOption"] p {{ color: inherit; font-size: .88rem; font-weight: 650; text-align: center; white-space: nowrap; }}
.stRadio [data-testid="stRadioOption"]:hover {{ color: var(--text); background: var(--surface-hover); }}
.stRadio [data-testid="stRadioOption"][data-selected="true"] {{
    color: #ffffff; background: linear-gradient(110deg, var(--accent), var(--blue));
    border-color: var(--accent); box-shadow: 0 4px 14px color-mix(in srgb, var(--accent) 18%, transparent);
}}
.stRadio [data-testid="stRadioOption"][data-selected="true"] p {{ color: #ffffff; }}

.stButton > button, button[data-testid^="stBaseButton-"], [data-testid="stLinkButton"] a {{
    border-radius: 999px; font-weight: 700; letter-spacing: .2px; transition: all .2s ease; min-height: 44px;
}}
.stButton > button[kind="primary"],
div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(110deg, var(--accent), var(--blue)) !important;
    color: #ffffff !important; border: 0 !important; box-shadow: 0 6px 18px color-mix(in srgb, var(--accent) 24%, transparent);
}}
 .stButton > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {{ filter: brightness(1.1); transform: translateY(-2px); }}
.stButton > button[kind="secondary"],
div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"],
button[data-testid="stBaseButton-secondary"],
[data-testid="stLinkButton"] a {{
    background: var(--surface) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; box-shadow: none;
}}
.stButton > button[kind="secondary"]:hover, [data-testid="stLinkButton"] a:hover {{
    background: var(--surface-hover) !important; border-color: var(--accent) !important; transform: translateY(-1px);
}}
div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:hover {{
    background: var(--surface-hover) !important; border-color: var(--accent) !important; transform: translateY(-1px);
}}
button[data-testid="stBaseButton-secondary"]:hover {{
    background: var(--surface-hover) !important; border-color: var(--accent) !important; transform: translateY(-1px);
}}
div[data-baseweb="tab-list"] {{
    background: var(--surface); border-radius: 12px; padding: 5px 8px 0;
    border: 1px solid var(--border); gap: 8px;
}}
button[data-baseweb="tab"] {{ font-weight: 700; font-size: .94rem; color: var(--muted); padding: 10px 16px; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: var(--neon) !important; border-bottom-color: var(--neon) !important; }}
div[data-baseweb="input"], div[data-baseweb="select"] > div {{
    background: var(--surface) !important; border-color: var(--border) !important; border-radius: 12px !important;
}}
div[data-baseweb="input"] input, div[data-baseweb="select"] * {{ color: var(--text); }}
div[data-baseweb="input"]:focus-within {{ border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }}
[data-testid="stTextInput"] input {{
    background: var(--surface) !important; color: var(--text) !important; border-color: var(--border) !important;
}}
[data-testid="stTextInput"] input::placeholder {{ color: var(--muted) !important; opacity: 1; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stLinkButton"] a {{
    background: var(--spotify-green) !important; color: #07150b !important;
    border-color: var(--spotify-green) !important; font-weight: 750;
}}
[data-testid="stLinkButton"] a:hover {{ filter: brightness(1.08); border-color: var(--accent) !important; }}
[data-testid="stDataFrame"], [data-testid="stAlert"] {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px; overflow: hidden;
}}
[data-testid="stHorizontalBlock"] {{ gap: 1rem; }}

@media (max-width: 760px) {{
    .block-container {{ padding: 1rem .85rem 2rem; }}
    .brand-title {{ font-size: 1.65rem; }}
    .brand-title svg {{ width: 34px; height: 34px; }}
    .sub-header {{ font-size: .9rem; }}
    .studio-hero {{ min-height: 0; padding: 24px; border-radius: 18px; }}
    .hero-art {{ display: none; }}
    .studio-copy {{ font-size: .92rem; }}
    .studio-title {{ font-size: clamp(2rem, 8vw, 2.6rem); }}
    .mood-banner {{ padding: 15px; }}
    .mood-banner-profile {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .mood-status {{ margin-left: auto; }}
    [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {{ min-width: min(100%, 300px); }}
    .song-artwork {{ width: 82px; height: 82px; flex-basis: 82px; }}
    .spotify-card {{ padding: 13px; }}
    .stRadio [data-testid="stRadioGroup"] {{ flex-wrap: wrap; }}
    .stRadio [data-testid="stRadioOption"] {{ min-width: 135px; }}
}}
@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ animation-duration: .01ms !important; transition-duration: .01ms !important; }}
}}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


# Data & Model Initializers
@st.cache_data
def get_data():
    songs = load_songs(str(DATA_DIR / "songs.csv"))
    return songs


songs_df = get_data()
mood_model = MoodRecommender(songs_df)
if "liked_song_ids" not in st.session_state:
    st.session_state["liked_song_ids"] = load_liked_song_ids()

# Session State for Mood
if "current_mood" not in st.session_state:
    st.session_state["current_mood"] = "💖 Romantic & Love"


# Navigation Top Bar
header_col1, header_col2 = st.columns([6.4, 3.6])
with header_col1:
    st.markdown(
        """
        <div class="brand-title">
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none" aria-hidden="true">
                <defs>
                    <linearGradient id="music-mood-gradient" x1="4" y1="4" x2="36" y2="36" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#8265F5"/>
                    <stop offset="0.62" stop-color="#3478ED"/>
                    <stop offset="1" stop-color="#1ED760"/>
                    </linearGradient>
                </defs>
                <circle cx="20" cy="20" r="19" fill="url(#music-mood-gradient)"/>
                <path d="M13 11V25.5M13 11L28 8V22.5" stroke="#10202a" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>
                <ellipse cx="9.5" cy="27" rx="4" ry="2.5" transform="rotate(-20 9.5 27)" fill="#10202a"/>
                <ellipse cx="24.5" cy="24" rx="4" ry="2.5" transform="rotate(-20 24.5 24)" fill="#10202a"/>
            </svg>
            <span>Music <span class="brand-green">Mood Finder</span></span>
        </div>
        <div class="sub-header">Personalized music discovery for your exact vibe & mood — Curated in <b>English</b> & <b>Malayalam (മലയാളം)</b></div>
        """,
        unsafe_allow_html=True,
    )

with header_col2:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    c_btn1, c_btn2 = st.columns([1.1, 1])
    with c_btn1:
        if st.button("🎲 Surprise Me", type="primary", use_container_width=True, help="Pick a random mood and refresh recommendations"):
            st.session_state["current_mood"] = random.choice(list(MOOD_PROFILES.keys()))
            st.rerun()
    with c_btn2:
        if st.button("↺ Reset Mood", type="secondary", use_container_width=True, help="Return to the default romantic mood"):
            st.session_state["current_mood"] = "💖 Romantic & Love"
            st.rerun()
    st.toggle("Theme", key="light_mode", help="Switch between light and dark appearance")

# Main Tabs
tab_discovery, tab_catalogue, tab_library = st.tabs([
    "✦ Discover",
    "◎ Explore all songs",
    f"♡ Your collection ({len(st.session_state['liked_song_ids'])})",
])


# ==============================================================================
# TAB 1: MOOD DISCOVERY
# ==============================================================================
with tab_discovery:
    st.markdown(
        f"""
        <section class="studio-hero">
            <div class="studio-hero-copy">
                <div class="studio-eyebrow">YOUR PERSONAL LISTENING STUDIO</div>
                <h1 class="studio-title">Find the sound<br>of <span>your moment.</span></h1>
                <p class="studio-copy">
                    A soundtrack that meets you where you are. Tell us how today feels,
                    and explore a hand-curated mix across English and Malayalam.
                </p>
                <div class="studio-pills">
                    <span class="studio-pill studio-pill-live"><span class="spotify-dot"></span><strong>{len(songs_df)} tracks</strong> to explore</span>
                    <span class="studio-pill">◉ English · മലയാളം</span>
                    <span class="studio-pill">♡ {len(st.session_state['liked_song_ids'])} saved</span>
                </div>
            </div>
            <div class="hero-art" aria-hidden="true">
                <div class="hero-disc">
                    <i style="--bar-height:24px"></i><i style="--bar-height:43px"></i>
                    <i style="--bar-height:58px"></i><i style="--bar-height:35px"></i>
                    <i style="--bar-height:68px"></i><i style="--bar-height:46px"></i>
                    <i style="--bar-height:28px"></i><i style="--bar-height:52px"></i>
                    <i style="--bar-height:34px"></i>
                </div>
            </div>
        </section>
        <div class="discovery-heading">
            <h2>What does your moment sound like?</h2>
            <p>Choose a feeling or describe it in your own words.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. MOOD INPUT BAR
    col_input, col_search_btn = st.columns([4, 1])
    with col_input:
        user_mood_query = st.text_input(
            "Enter your mood or feeling:",
            placeholder="Type any feeling: e.g. 'romantic dinner with rain', 'gym workout hype', 'sad heartbreak', 'chill coffee vibe'...",
            label_visibility="collapsed",
            key="mood_text_input",
        )
    with col_search_btn:
        search_clicked = st.button("Find my sound  ↗", type="primary", use_container_width=True)

    if search_clicked and user_mood_query.strip():
        detected = MoodRecommender.detect_mood_from_text(user_mood_query)
        st.session_state["current_mood"] = detected

    # 2. QUICK MOOD CHIPS
    st.markdown('<div class="section-hint">QUICK MOOD PICKS</div>', unsafe_allow_html=True)

    mood_options = [
        "💖 Romantic & Love",
        "😊 Happy / Upbeat",
        "😢 Sad / Melancholic",
        "⚡ Energetic / Workout",
        "🎉 Party & Dance",
        "😌 Chill / Relaxed",
        "☕ Coffee & Acoustic",
        "💔 Heartbreak & Emotional",
        "🧘 Focus / Study",
    ]

    for row_start in range(0, len(mood_options), 3):
        chip_cols = st.columns(3)
        for column, i in enumerate(range(row_start, min(row_start + 3, len(mood_options)))):
            mood_label = mood_options[i]
            with chip_cols[column]:
                is_active = st.session_state["current_mood"] == mood_label
                btn_style = "primary" if is_active else "secondary"
                clean_name = mood_label.split(" ", 1)[0] + " " + mood_label.split(" ")[1]
                if st.button(clean_name, key=f"chip_{i}", type=btn_style, use_container_width=True):
                    st.session_state["current_mood"] = mood_label
                    st.rerun()

    active_mood = st.session_state["current_mood"]
    target_profile = MOOD_PROFILES.get(active_mood, MOOD_PROFILES["😊 Happy / Upbeat"])

    valence_pct = target_profile["valence"] * 100
    energy_pct = target_profile["energy"] * 100
    danceability_pct = target_profile["danceability"] * 100
    tempo_pct = min(100, target_profile["tempo"] / 2)

    st.markdown(
        f"""
        <div class="mood-banner">
            <div>
                <div class="mood-banner-title">Current Mood: <span class="mood-banner-value">{active_mood}</span></div>
                <div class="mood-banner-profile">
                    <div class="mood-stat">
                        <div class="mood-stat-label">Valence</div>
                        <div class="mood-stat-value">{valence_pct:.0f}%</div>
                        <div class="mood-stat-track"><span style="width:{valence_pct:.0f}%"></span></div>
                    </div>
                    <div class="mood-stat">
                        <div class="mood-stat-label">Energy</div>
                        <div class="mood-stat-value">{energy_pct:.0f}%</div>
                        <div class="mood-stat-track"><span style="width:{energy_pct:.0f}%"></span></div>
                    </div>
                    <div class="mood-stat">
                        <div class="mood-stat-label">Danceability</div>
                        <div class="mood-stat-value">{danceability_pct:.0f}%</div>
                        <div class="mood-stat-track"><span style="width:{danceability_pct:.0f}%"></span></div>
                    </div>
                    <div class="mood-stat">
                        <div class="mood-stat-label">Tempo</div>
                        <div class="mood-bpm">~{target_profile['tempo']} BPM</div>
                        <div class="mood-stat-track"><span style="width:{tempo_pct:.0f}%"></span></div>
                    </div>
                </div>
            </div>
            <span class="badge-match mood-status" style="font-size: 0.8rem; padding: 6px 11px;">✦ Vibe matched</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. LANGUAGE DISPLAY CONTROLS
    filter_col1, filter_col2 = st.columns([7, 3])
    with filter_col1:
        lang_view = st.radio(
            "Choose which music to explore:",
            ["🌐 English + Malayalam", "English Only", "🌴 Malayalam Only · മലയാളം"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with filter_col2:
        top_k = st.slider("Songs per language (up to 50):", min_value=3, max_value=50, value=6, step=1)

    # Get Recommendations
    try:
        eng_recs, mal_recs = mood_model.recommend_dual(active_mood, top_n=top_k)
    except (AttributeError, Exception):
        eng_recs = mood_model.recommend(active_mood, top_n=top_k, language="English")
        mal_recs = mood_model.recommend(active_mood, top_n=top_k, language="Malayalam")

    if "English + Malayalam" in lang_view:
        eng_recs = add_original_artwork(eng_recs.copy())
        mal_recs = add_original_artwork(mal_recs.copy())
    elif "English Only" in lang_view:
        eng_recs = add_original_artwork(eng_recs.copy())
    else:
        mal_recs = add_original_artwork(mal_recs.copy())

    # Helper function to render a mood-matched song card
    def render_song_card(row, col, is_eng=True, key_prefix="discovery"):
        with col:
            song_id = int(row["song_id"])
            is_liked = song_id in st.session_state["liked_song_ids"]
            title_display = html.escape(str(row["title"]))
            native_title = row.get("title_native")
            native_str = (
                f" <span class='song-native-title'>{html.escape(str(native_title))}</span>"
                if not is_eng and pd.notna(native_title) and native_title != row["title"]
                else ""
            )

            artwork_url = row.get("original_artwork_url")
            cover_img = html.escape(
                str(artwork_url if isinstance(artwork_url, str) and artwork_url else artwork_placeholder(row["title"])),
                quote=True,
            )
            match_pct = row.get("mood_match_pct", 95)
            album_movie = html.escape(str(row.get("album_movie", "Single")))
            artist = html.escape(str(row["artist"]))
            genre = html.escape(str(row.get("genre", "music")).lower())
            era = row.get("era")
            era_label = (
                "Trending 2026 chart"
                if era == "Trending 2026"
                else str(era)
                if isinstance(era, str) and era.strip()
                else ""
            )
            era_badge = (
                f'<span class="badge-era">{html.escape(era_label)}</span>'
                if era_label
                else ""
            )
            album_detail = (
                ""
                if era == "Trending 2026"
                else f"💿 {album_movie} · "
                if album_movie
                else ""
            )
            genre_article = "an" if genre.startswith(("a", "e", "i", "o", "u")) else "a"
            energy = float(row.get("energy", 0))
            acousticness = float(row.get("acousticness", 0))
            sound_style = "mellow acoustic" if acousticness >= 0.5 else "bright rhythmic"
            description = (
                f"{genre_article.capitalize()} {genre} track from {album_movie}, picked for your current vibe. "
                f"It has a {'high-energy' if energy >= 0.65 else 'laid-back'} feel "
                f"with {sound_style} textures."
            )
            spotify_url = html.escape(
                str(row.get("spotify_url", f"https://open.spotify.com/search/{row['title']}+{row['artist']}".replace(" ", "+"))),
                quote=True,
            )
            duration = html.escape(str(row.get("duration", "3:30")))
            tempo = float(row.get("tempo", 0))
            match_width = min(100, max(0, float(match_pct)))

            card_html = (
                f'<div class="spotify-card">'
                f'<div class="song-card-layout">'
                f'<img src="{cover_img}" alt="Album artwork for {title_display}" '
                f'class="song-artwork" />'
                f'<div class="song-copy">'
                f'<div class="song-title-row">'
                f'<div class="song-title">{title_display}{native_str}</div>'
                f'<span class="badge-match" style="white-space:nowrap;">{match_pct}% Match</span>'
                f'</div>'
                f'<div class="song-artist">{artist}</div>'
                f'<div class="song-meta">{album_detail}⏱️ {duration} · ⚡ {tempo:.0f} BPM</div>'
                f'<p class="song-description">{description}</p>'
                f'<div class="song-tags">'
                f'<span class="badge-tag">Valence: {float(row["valence"])*100:.0f}%</span>'
                f'<span class="badge-tag">Energy: {energy*100:.0f}%</span>'
                f'<span class="badge-tag">Acoustic: {acousticness*100:.0f}%</span>'
                f'{era_badge}'
                f'</div>'
                f'<div class="song-match-track" aria-label="{match_width:.0f}% mood match">'
                f'<div class="song-match-fill" style="width:{match_width:.0f}%"></div>'
                f'</div></div></div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

            action_col1, action_col2 = st.columns([1, 1])
            with action_col1:
                like_label = "♥ Saved" if is_liked else "♡ Like"
                if st.button(
                    like_label,
                    key=f"{key_prefix}_like_{song_id}",
                    type="primary" if is_liked else "secondary",
                    use_container_width=True,
                ):
                    liked_ids = st.session_state["liked_song_ids"].copy()
                    if is_liked:
                        liked_ids.remove(song_id)
                    else:
                        liked_ids.append(song_id)
                    if save_liked_song_ids(liked_ids):
                        st.session_state["liked_song_ids"] = liked_ids
                        st.rerun()
            with action_col2:
                st.link_button("▶ Play on Spotify", spotify_url, use_container_width=True)

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Render according to selected view
    if "English + Malayalam" in lang_view:
        col_eng, col_mal = st.columns(2)
        with col_eng:
            st.markdown(
                f"""
                <div class="recommendation-heading">
                    <div>
                        <h3>English mood mix</h3>
                        <p>Picked for {html.escape(active_mood)}</p>
                    </div>
                    <span class="recommendation-count">{len(eng_recs)} tracks · EN</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for _, row in eng_recs.iterrows():
                render_song_card(row, col_eng, is_eng=True)

        with col_mal:
            st.markdown(
                f"""
                <div class="recommendation-heading">
                    <div>
                        <h3>Malayalam mood mix · മലയാളം</h3>
                        <p>Picked for {html.escape(active_mood)}</p>
                    </div>
                    <span class="recommendation-count">{len(mal_recs)} tracks · ML</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for _, row in mal_recs.iterrows():
                render_song_card(row, col_mal, is_eng=False)

    elif "English Only" in lang_view:
        st.markdown(
            f'<div class="recommendation-heading"><div><h3>English mood mix</h3>'
            f'<p>Picked for {html.escape(active_mood)}</p></div>'
            f'<span class="recommendation-count">{len(eng_recs)} tracks · EN</span></div>',
            unsafe_allow_html=True,
        )
        grid_cols = st.columns(2)
        for idx, (_, row) in enumerate(eng_recs.iterrows()):
            render_song_card(row, grid_cols[idx % 2], is_eng=True)

    else:  # Malayalam Only
        st.markdown(
            f'<div class="recommendation-heading"><div><h3>Malayalam mood mix · മലയാളം</h3>'
            f'<p>Picked for {html.escape(active_mood)}</p></div>'
            f'<span class="recommendation-count">{len(mal_recs)} tracks · ML</span></div>',
            unsafe_allow_html=True,
        )
        grid_cols = st.columns(2)
        for idx, (_, row) in enumerate(mal_recs.iterrows()):
            render_song_card(row, grid_cols[idx % 2], is_eng=False)


# ==============================================================================
# TAB 2: COMPLETE SONG CATALOGUE & SEARCH
# ==============================================================================
with tab_catalogue:
    st.markdown("### 📚 Comprehensive Music Catalogue (English & Malayalam)")
    col_c1, col_c2, col_c3 = st.columns([3, 2, 2])
    with col_c1:
        cat_search = st.text_input("🔍 Search title, artist, or movie:", placeholder="e.g. Malare, Ed Sheeran, Aavesham...")
    with col_c2:
        cat_lang = st.selectbox("Language Filter:", ["All", "English", "Malayalam"])
    with col_c3:
        cat_genre = st.selectbox("Mood / Category:", ["All"] + sorted(songs_df["mood"].unique().tolist()))

    filtered_cat = songs_df.copy()
    if cat_search.strip():
        q = cat_search.lower().strip()
        filtered_cat = filtered_cat[
            filtered_cat["title"].str.lower().str.contains(q, na=False) |
            filtered_cat["artist"].str.lower().str.contains(q, na=False) |
            filtered_cat.get("album_movie", "").str.lower().str.contains(q, na=False)
        ]
    if cat_lang != "All":
        filtered_cat = filtered_cat[filtered_cat["language"] == cat_lang]
    if cat_genre != "All":
        filtered_cat = filtered_cat[filtered_cat["mood"] == cat_genre]

    st.markdown(f"Showing **{len(filtered_cat)}** curated tracks:")
    display_cols = ["song_id", "title", "title_native", "artist", "album_movie", "language", "era", "mood", "tempo", "valence", "energy", "duration"]
    st.dataframe(filtered_cat[[c for c in display_cols if c in filtered_cat.columns]], use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 3: SAVED MUSIC LIBRARY
# ==============================================================================
with tab_library:
    liked_ids = st.session_state["liked_song_ids"]
    st.markdown("### 💚 Your Liked Songs")
    st.caption("Your saved songs stay in your library when you restart the app.")

    if not liked_ids:
        st.info("Your library is empty. Tap ♡ Like on any recommendation to save it here.")
    else:
        library_songs = (
            songs_df.set_index("song_id")
            .reindex(liked_ids)
            .dropna(subset=["title"])
            .reset_index()
        )
        library_songs = add_original_artwork(library_songs)
        st.markdown(f"**{len(library_songs)}** saved tracks")
        library_cols = st.columns(2)
        for idx, (_, row) in enumerate(library_songs.iterrows()):
            render_song_card(
                row,
                library_cols[idx % 2],
                is_eng=row["language"] == "English",
                key_prefix="library",
            )
