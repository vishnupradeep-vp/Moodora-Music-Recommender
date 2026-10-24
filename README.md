# Music Mood Finder

### Mood-based music discovery for English and Malayalam songs

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-1DB954?style=for-the-badge&logo=streamlit&logoColor=white)](https://tunefinder-nsegltapp83frwee9neh5yu.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#license)

> **Live Application:** https://tunefinder-nsegltapp83frwee9neh5yu.streamlit.app/

---

## Overview

**Music Mood Finder** is a mood-first music discovery application built with Python and Streamlit. It helps users discover English and Malayalam songs based on their mood, music preferences, and catalogue filters.

The application combines a curated music catalogue with content-based, collaborative, and hybrid recommendation approaches in an interactive web interface.

The project currently contains **159 tracks** with metadata, mood information, audio features, language and era tags.

---

## Features

### Mood-Based Discovery

Users can describe their mood or select predefined mood filters to receive relevant recommendations.

Supported mood categories include:

- Happy
- Calm
- Romantic
- Sad
- Energetic
- Party
- Relaxed
- Nostalgic

The application can return up to **50 English and 50 Malayalam recommendations** for a mood query.

### Music Catalogue

A searchable catalogue allows users to filter tracks by:

- Song title
- Artist
- Language
- Era
- Mood

### Recommendation Engine

The project implements three recommendation approaches:

- **Content-Based Recommender**
- **Collaborative Recommender**
- **Hybrid Recommender**

### Liked Songs

Users can save recommended songs to a persistent favourites library.

Liked song IDs are stored in:

```text
data/liked_songs.json
```

### Song Details

Recommendations can include:

- Album artwork
- Artist and album information
- Audio-profile description
- Feature badges
- Spotify links

### User Interface

The Streamlit application includes:

- Modern music-focused dashboard
- Responsive layouts
- Dark and light themes
- Mood-profile visualizations
- Editorial-style album cards
- Cinematic music-oriented interface

---

## Recommendation Architecture

The recommendation pipeline combines song features and user-rating information.

```text
                         User Mood
                            |
                            v
                  +-------------------+
                  | Feature Processing |
                  +---------+---------+
                            |
             +--------------+--------------+
             |                             |
             v                             v
   +--------------------+       +--------------------+
   | Content-Based      |       | Collaborative      |
   | Recommender        |       | Recommender        |
   +---------+----------+       +---------+----------+
             |                            |
             +-------------+--------------+
                           |
                           v
                 +-------------------+
                 | Hybrid            |
                 | Recommender       |
                 +---------+---------+
                           |
                           v
                  Recommended Songs
```

---

## Recommendation Methods

### Content-Based Recommendation

The content-based model recommends songs according to similarities between their features.

Relevant features include:

- Energy
- Valence
- Danceability
- Acousticness
- Instrumentalness
- Tempo
- Mood
- Language
- Era

### Collaborative Recommendation

The collaborative model uses user listening and rating information to identify potential preferences.

The project includes synthetic rating data for experimentation and evaluation.

### Hybrid Recommendation

The hybrid model combines recommendation signals from multiple approaches to produce a more comprehensive recommendation result.

---

## Evaluation

The project includes evaluation utilities for measuring recommendation quality.

### Precision@K

Measures the proportion of recommended items that are relevant.

### Recall@K

Measures the proportion of relevant items that were successfully recommended.

Evaluation functionality is implemented in:

```text
src/evaluate.py
```

---

## Dataset

The application contains a curated catalogue of **159 tracks**.

The dataset includes information such as:

```text
Song
Artist
Album
Language
Mood
Era
Energy
Valence
Danceability
Acousticness
Instrumentalness
Tempo
```

### Music Eras

The catalogue includes additional selections from:

- 1990s
- 2019
- Global Spotify daily chart snapshot dated 15 October 2026

> The 2026 chart selections represent tracks appearing on the specified chart snapshot and are not necessarily tracks released in 2026.

---

## Project Structure

```text
music-mood-finder/
│
├── data/
│   ├── generate_data.py
│   ├── songs.csv
│   ├── ratings.csv
│   └── liked_songs.json
│
├── src/
│   ├── preprocess.py
│   ├── recommend.py
│   └── evaluate.py
│
├── tests/
│   ├── test_preprocess.py
│   ├── test_recommend.py
│   └── test_evaluate.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Project Components

### `app.py`

Main Streamlit application containing the user interface and application workflow.

### `src/preprocess.py`

Handles:

- Data loading
- Feature preprocessing
- Feature scaling
- Recommendation input preparation

### `src/recommend.py`

Contains:

```text
ContentBasedRecommender
CollaborativeRecommender
HybridRecommender
```

### `src/evaluate.py`

Provides recommendation evaluation metrics including:

```text
Precision@K
Recall@K
```

### `data/songs.csv`

Main music catalogue containing 159 tracks.

### `data/ratings.csv`

Synthetic user listening and rating data used for collaborative recommendation experiments.

### `data/liked_songs.json`

Persistent storage for liked song IDs.

### `data/generate_data.py`

Utility for generating and preparing project data.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning |
| SciPy | Scientific computing |
| Streamlit | Web application |
| CSV | Dataset storage |
| JSON | Persistent favourites storage |

---

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd music-mood-finder
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

### Start Streamlit

```bash
streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

### Run the Recommendation Demo

```bash
python -m src.recommend
```

### Run Unit Tests

```bash
python -m unittest discover -s tests
```

---

## Testing

Unit tests are provided for the core project components.

```bash
python -m unittest discover -s tests
```

Test modules include:

- `test_preprocess.py`
- `test_recommend.py`
- `test_evaluate.py`

---

## Deployment

The application is deployed using Streamlit.

### Live Demo

**https://tunefinder-nsegltapp83frwee9neh5yu.streamlit.app/**

[**Open Music Mood Finder →**](https://tunefinder-nsegltapp83frwee9neh5yu.streamlit.app/)

---

## Screenshots

Add application screenshots to showcase the interface.

Recommended structure:

```text
screenshots/
├── home.png
├── mood-discovery.png
├── recommendations.png
├── liked-songs.png
└── catalogue.png
```

Example:

```markdown
## Screenshots

### Home

![Home](screenshots/home.png)

### Mood Discovery

![Mood Discovery](screenshots/mood-discovery.png)

### Recommendations

![Recommendations](screenshots/recommendations.png)

### Liked Songs

![Liked Songs](screenshots/liked-songs.png)
```

---

## Key Project Statistics

| Metric | Value |
|---|---:|
| Tracks | 159 |
| Languages | English, Malayalam |
| Recommendation Models | 3 |
| Recommendation Metrics | Precision@K, Recall@K |
| Persistent Favourites | Supported |
| Search & Filtering | Supported |
| Themes | Light / Dark |
| Deployment | Streamlit |

---

## Future Enhancements

Potential improvements include:

- AI-powered natural-language mood understanding
- Voice-based mood input
- Real-time Spotify API integration
- YouTube integration
- User authentication
- Personalized user profiles
- Cloud database integration
- Listening analytics
- Lyrics-based mood analysis
- Additional Indian languages
- Deep-learning recommendation models

---

## Use Cases

Music Mood Finder can be used as:

- A personal music discovery platform
- A recommendation-system demonstration
- A machine-learning project
- A data-science portfolio project
- A Streamlit application project
- An academic project
- A recommendation algorithm experimentation platform

---

## Privacy and Data

The application stores liked song IDs in a local JSON file.

The rating dataset included in the project is synthetic and is intended for recommendation-system experimentation.

---

## License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more information.

---

## Author

**Music Mood Finder**

Built with:

**Python · Machine Learning · Data Science · Streamlit**

---

<p align="center">

### Music Mood Finder

**Discover music that matches your mood.**

[**🎧 Launch Live Application**](https://tunefinder-nsegltapp83frwee9neh5yu.streamlit.app/)

</p>
