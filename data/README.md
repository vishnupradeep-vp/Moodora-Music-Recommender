# 🎵 Music Mood Finder — Music Recommendation System

A mini machine learning project that recommends songs using three
approaches: **content-based filtering**, **collaborative filtering**,
and a **hybrid** blend of both — wrapped in an interactive Streamlit app.

## Demo

```bash
streamlit run app.py
```

Then pick a mode in the sidebar:
- **Content-Based** — pick a song, get similar songs based on audio features (danceability, energy, tempo, valence, acousticness, loudness)
- **Collaborative** — pick a user, get songs predicted from the taste of similar users
- **Hybrid** — combine both signals with a weighted score

## How it works

| Method | Technique | Good for |
|---|---|---|
| Content-Based | Cosine similarity over normalized audio features | New songs / cold start |
| Collaborative | Matrix factorization (truncated SVD) on the user–song rating matrix | Discovering non-obvious recommendations |
| Hybrid | Weighted blend of both scores | Best of both worlds |

## Project structure

```
tunefinder/
├── data/
│   ├── generate_data.py   # creates synthetic songs.csv + ratings.csv
│   ├── songs.csv          # song catalogue with audio features
│   └── ratings.csv        # synthetic user listening history
├── src/
│   ├── preprocess.py      # loading & feature scaling
│   └── recommend.py       # ContentBasedRecommender, CollaborativeRecommender, HybridRecommender
├── app.py                 # Streamlit UI
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd tunefinder
pip install -r requirements.txt

# (optional) regenerate the dataset
cd data && python generate_data.py && cd ..

streamlit run app.py
```

## Using real data instead of synthetic data

This repo ships with a **synthetic dataset** (`data/generate_data.py`) so
the whole project runs out of the box with zero downloads or API keys.

To swap in real data:
1. Download the [Spotify Dataset from Kaggle](https://www.kaggle.com/datasets)
2. Save it as `data/songs.csv` with columns matching `FEATURE_COLUMNS` in `src/preprocess.py`
   (`danceability, energy, valence, acousticness, tempo, loudness`)
3. Replace `data/ratings.csv` with a real `user_id, song_id, rating` interaction log

## Example output

```
=== Content-Based Demo ===
Because you liked: Electric Waves (R&B)

             title    artist       genre  similarity
0   Hollow Journey  Artist G     Hip-Hop       0.997
1    Golden Lights  Artist Z         Pop       0.996
...

=== Collaborative Filtering Demo ===
Recommended for user 0:

              title    artist    genre  predicted_rating
0     Fading Static  Artist L  Country              4.22
...
```

Run it yourself:
```bash
python -m src.recommend
```

## Possible extensions
- Swap synthetic data for real Spotify/Last.fm data
- Add Spotify API (`spotipy`) integration for album art + audio previews
- Evaluate with Precision@K / Recall@K on held-out ratings
- Deploy the Streamlit app to Streamlit Community Cloud
- Add a `Dockerfile` for reproducible deployment

## Tech stack
`pandas` · `numpy` · `scikit-learn` · `scipy` · `streamlit`

## License
MIT
