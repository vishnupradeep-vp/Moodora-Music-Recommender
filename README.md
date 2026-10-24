# 🎵 Music Mood Finder

A mood-first Streamlit app for discovering English and Malayalam songs, saving favorites, and searching a curated catalogue.

## Features

- **🎧 Mood Discovery**: Find up to 50 English and 50 Malayalam tracks from a mood prompt or quick mood filters.
- **🎨 Mood-first listening studio**: A cinematic sound-wave hero, deep-navy dashboard palette with violet, blue, teal, and subtle Spotify-green accents, visual mood-profile meters, counted recommendation mixes, editorial album cards, responsive layouts, and a Theme switch for light or dark mode.
- **💚 Liked Songs Library**: Save recommendations and keep them in `data/liked_songs.json` across app restarts.
- **📚 Searchable Catalogue**: Browse and filter 159 tracks by title, artist, language, era, and mood.
- **🎵 Song Details**: Matching original album artwork when available, a short audio-profile description, feature badges, and Spotify links.
- **🗓️ Music Eras**: Includes 50 additional tracks from the 1990s, 2019, and the 2026 Global Spotify daily chart.

The 2026 chart picks reflect the Global Spotify daily chart snapshot dated 15 October 2026; they are charting tracks, not necessarily tracks released in 2026.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run recommendation engine demo
python -m src.recommend

# Run unit tests
python -m unittest discover -s tests

# Run the interactive Music Mood Finder app
streamlit run app.py
```

## Project Architecture

```
music-mood-finder/
├── data/
│   ├── generate_data.py   # synthetic data generation script
│   ├── songs.csv          # 159-song catalogue with audio features and era tags
│   ├── ratings.csv        # synthetic user listening history
│   └── liked_songs.json   # persistent liked song IDs (starts empty)
├── src/
│   ├── preprocess.py      # feature scaling & data loading
│   ├── recommend.py       # ContentBasedRecommender, CollaborativeRecommender, HybridRecommender
│   └── evaluate.py        # Precision@K and Recall@K evaluation metrics
├── tests/
│   ├── test_preprocess.py # unit tests for preprocessing
│   ├── test_recommend.py  # unit tests for recommenders
│   └── test_evaluate.py   # unit tests for evaluation metrics
├── app.py                 # Music Mood Finder Streamlit UI
├── requirements.txt
└── README.md
```

## Tech Stack
`python` · `pandas` · `numpy` · `scikit-learn` · `scipy` · `streamlit`

## License
MIT
