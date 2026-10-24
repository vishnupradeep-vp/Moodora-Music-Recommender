"""
recommend.py
------------
Two recommendation strategies:

1. ContentBasedRecommender
   Recommends songs whose AUDIO FEATURES (danceability, energy, tempo, etc.)
   are most similar to a given song, using cosine similarity.
   Works even for brand-new songs with no listening history ("cold start" OK).

2. CollaborativeRecommender
   Recommends songs based on patterns in what OTHER USERS with similar taste
   have rated highly, using matrix factorization (truncated SVD).
   Needs historical rating data, but can surface non-obvious recommendations
   content-based filtering would miss.

3. HybridRecommender
   Simple weighted blend of both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import FEATURE_COLUMNS, load_ratings, load_songs, scale_features


class ContentBasedRecommender:
    def __init__(self, songs_df: pd.DataFrame):
        self.songs = songs_df.reset_index(drop=True)
        scaled = scale_features(self.songs)
        self.feature_matrix = scaled[FEATURE_COLUMNS].values
        self.similarity = cosine_similarity(self.feature_matrix)
        self._id_to_idx = {sid: i for i, sid in enumerate(self.songs["song_id"])}

    def recommend(self, song_id: int, top_n: int = 10) -> pd.DataFrame:
        if song_id not in self._id_to_idx:
            raise ValueError(f"song_id {song_id} not found")

        idx = self._id_to_idx[song_id]
        scores = list(enumerate(self.similarity[idx]))
        # Filter out the query song itself to prevent self-recommendation
        scores = [item for item in scores if item[0] != idx]
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_n]

        result = self.songs.iloc[[i for i, _ in top]].copy()
        result["similarity"] = [round(s, 3) for _, s in top]
        return result.reset_index(drop=True)

    def recommend_by_title(self, title: str, top_n: int = 10) -> pd.DataFrame:
        match = self.songs[self.songs["title"].str.lower() == title.lower()]
        if match.empty:
            raise ValueError(f"No song titled '{title}' found")
        return self.recommend(match.iloc[0]["song_id"], top_n)


class CollaborativeRecommender:
    """Matrix-factorization based collaborative filtering using truncated SVD.
    (A lightweight alternative to the `surprise` library -- pure scipy/numpy,
    so it installs cleanly everywhere.)"""

    def __init__(self, ratings_df: pd.DataFrame, songs_df: pd.DataFrame, n_factors: int = 20):
        self.songs = songs_df.reset_index(drop=True)
        self.song_ids = sorted(self.songs["song_id"].unique())
        self._song_to_idx = {s: i for i, s in enumerate(self.song_ids)}

        # Keep ratings only for songs present in song catalog
        valid_ratings = ratings_df[ratings_df["song_id"].isin(self._song_to_idx)].copy()
        self.ratings = valid_ratings
        self.user_ids = sorted(valid_ratings["user_id"].unique())
        self._user_to_idx = {u: i for i, u in enumerate(self.user_ids)}

        if not self.user_ids or not self.song_ids:
            self.user_means = np.array([])
            self.predicted = np.zeros((len(self.user_ids), len(self.song_ids)))
            return

        matrix = csr_matrix(
            (
                valid_ratings["rating"],
                (
                    valid_ratings["user_id"].map(self._user_to_idx),
                    valid_ratings["song_id"].map(self._song_to_idx),
                ),
            ),
            shape=(len(self.user_ids), len(self.song_ids)),
        )

        num_ratings_per_user = (matrix != 0).sum(axis=1).A1
        self.user_means = np.array(matrix.sum(axis=1)).flatten() / np.maximum(
            num_ratings_per_user, 1
        )
        dense = matrix.toarray().astype(float)
        mask = dense != 0
        normalized = dense - self.user_means[:, None] * mask

        min_dim = min(normalized.shape)
        k = min(n_factors, min_dim - 1)
        if k >= 1 and min_dim > 1:
            try:
                U, sigma, Vt = svds(normalized, k=k)
                sigma = np.diag(sigma)
                self.predicted = U @ sigma @ Vt + self.user_means[:, None]
            except Exception:
                self.predicted = dense
        else:
            self.predicted = self.user_means[:, None] * np.ones_like(dense)

    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        if user_id not in self._user_to_idx:
            raise ValueError(f"user_id {user_id} not found")

        u_idx = self._user_to_idx[user_id]
        pred_scores = self.predicted[u_idx]

        already_rated = set(
            self.ratings.loc[self.ratings["user_id"] == user_id, "song_id"]
        )

        candidates = [
            (self.song_ids[i], score)
            for i, score in enumerate(pred_scores)
            if self.song_ids[i] not in already_rated
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_n]

        top_ids = [sid for sid, _ in top]
        result = self.songs[self.songs["song_id"].isin(top_ids)].copy()
        score_map = dict(top)
        result["predicted_rating"] = result["song_id"].map(score_map).round(2)
        return result.sort_values("predicted_rating", ascending=False).reset_index(drop=True)


class HybridRecommender:
    """Blends content-based similarity with collaborative predicted rating."""

    def __init__(self, content_model: ContentBasedRecommender,
                 collab_model: CollaborativeRecommender, alpha: float = 0.5):
        self.content_model = content_model
        self.collab_model = collab_model
        self.alpha = alpha  # weight given to content-based score

    def recommend(self, user_id: int, seed_song_id: int, top_n: int = 10) -> pd.DataFrame:
        content = self.content_model.recommend(seed_song_id, top_n=50)
        collab = self.collab_model.recommend(user_id, top_n=50)

        merged = pd.merge(
            content[["song_id", "similarity"]],
            collab[["song_id", "predicted_rating"]],
            on="song_id",
            how="outer",
        )

        if merged.empty:
            return pd.DataFrame(columns=["song_id", "title", "artist", "genre", "hybrid_score"])

        # Populate song metadata for all merged candidates
        song_metadata = self.content_model.songs
        merged = merged.merge(song_metadata, on="song_id", how="left")

        # Fill similarity: if missing, calculate cosine similarity against seed_song_id
        if seed_song_id in self.content_model._id_to_idx:
            seed_idx = self.content_model._id_to_idx[seed_song_id]
            def get_sim(row):
                if pd.notna(row["similarity"]):
                    return row["similarity"]
                sid = row["song_id"]
                if sid in self.content_model._id_to_idx:
                    c_idx = self.content_model._id_to_idx[sid]
                    return round(float(self.content_model.similarity[seed_idx, c_idx]), 3)
                return 0.0
            merged["similarity"] = merged.apply(get_sim, axis=1)
        else:
            merged["similarity"] = merged["similarity"].fillna(0.0)

        # Fill missing predicted_rating
        mean_pr = merged["predicted_rating"].mean()
        merged["predicted_rating"] = merged["predicted_rating"].fillna(3.0 if pd.isna(mean_pr) else mean_pr)

        # Normalize predicted_rating to 0-1 range
        pr = merged["predicted_rating"]
        pr_min, pr_max = pr.min(), pr.max()
        if pr_max > pr_min:
            merged["predicted_rating_norm"] = (pr - pr_min) / (pr_max - pr_min)
        else:
            merged["predicted_rating_norm"] = 0.5

        merged["hybrid_score"] = (
            self.alpha * merged["similarity"]
            + (1 - self.alpha) * merged["predicted_rating_norm"]
        ).round(3)

        merged = merged.dropna(subset=["title"])
        return merged.sort_values("hybrid_score", ascending=False).head(top_n).reset_index(drop=True)



MOOD_PROFILES = {
    "😊 Happy / Upbeat": {
        "valence": 0.88, "energy": 0.82, "danceability": 0.80, "acousticness": 0.15, "tempo": 120, "loudness": -5
    },
    "💖 Romantic & Love": {
        "valence": 0.75, "energy": 0.55, "danceability": 0.60, "acousticness": 0.65, "tempo": 95, "loudness": -7
    },
    "😢 Sad / Melancholic": {
        "valence": 0.20, "energy": 0.30, "danceability": 0.35, "acousticness": 0.85, "tempo": 80, "loudness": -12
    },
    "💔 Heartbreak & Emotional": {
        "valence": 0.18, "energy": 0.35, "danceability": 0.30, "acousticness": 0.82, "tempo": 78, "loudness": -11
    },
    "⚡ Energetic / Workout": {
        "valence": 0.75, "energy": 0.92, "danceability": 0.80, "acousticness": 0.05, "tempo": 132, "loudness": -3
    },
    "🎉 Party & Dance": {
        "valence": 0.88, "energy": 0.90, "danceability": 0.88, "acousticness": 0.08, "tempo": 126, "loudness": -4
    },
    "😌 Chill / Relaxed": {
        "valence": 0.55, "energy": 0.40, "danceability": 0.55, "acousticness": 0.70, "tempo": 90, "loudness": -9
    },
    "☕ Coffee & Acoustic": {
        "valence": 0.65, "energy": 0.42, "danceability": 0.50, "acousticness": 0.80, "tempo": 88, "loudness": -9
    },
    "🧘 Focus / Study": {
        "valence": 0.40, "energy": 0.22, "danceability": 0.25, "acousticness": 0.92, "tempo": 75, "loudness": -14
    },
    "🤪 Funny / Playful": {
        "valence": 0.82, "energy": 0.72, "danceability": 0.84, "acousticness": 0.15, "tempo": 115, "loudness": -6
    },
    "🍕 Hungry / Munchies & Chill": {
        "valence": 0.65, "energy": 0.50, "danceability": 0.65, "acousticness": 0.40, "tempo": 95, "loudness": -8
    },
}

MOOD_KEYWORDS = {
    "💖 Romantic & Love": ["romantic", "love", "romance", "crush", "date", "wedding", "proposal", "pranayam", "premam", "heart", "darling", "honey", "soulmate"],
    "😊 Happy / Upbeat": ["happy", "cheerful", "smile", "joy", "fun", "good vibe", "uplifting", "positive", "bright", "sunny", "sanmsham", "chiriyode"],
    "😢 Sad / Melancholic": ["sad", "crying", "tears", "depressed", "sorrow", "alone", "lonely", "dark", "pain", "hurt", "sankadam", "vishadam"],
    "💔 Heartbreak & Emotional": ["heartbreak", "heartbroken", "breakup", "broken heart", "ex", "betrayed", "viraham", "pirinju", "crying", "miss you"],
    "⚡ Energetic / Workout": ["workout", "gym", "energetic", "motivation", "pump", "training", "running", "power", "beast", "heavy", "lift", "hype"],
    "🎉 Party & Dance": ["party", "dance", "club", "celebration", "dj", "bass", "festival", "aavesham", "thallumaala", "banger", "night out"],
    "😌 Chill / Relaxed": ["chill", "relax", "relaxed", "calm", "peace", "evening", "slow", "lazy", "cozy", "breeze", "unwind", "mellow"],
    "☕ Coffee & Acoustic": ["acoustic", "coffee", "guitar", "rain", "rainy", "monsoon", "mazha", "unplugged", "serene", "tea"],
    "🧘 Focus / Study": ["focus", "study", "work", "reading", "ambient", "peaceful", "meditation", "concentration", "soulful", "instrumental"],
}


class MoodRecommender:
    """Recommends songs matching a person's current emotional state/mood."""

    def __init__(self, songs_df: pd.DataFrame):
        self.songs = songs_df.reset_index(drop=True)
        self.scaled_songs = scale_features(self.songs)

    @staticmethod
    def detect_mood_from_text(text: str) -> str:
        """Parses user natural language query and maps to the best mood profile."""
        if not text or not text.strip():
            return "😊 Happy / Upbeat"

        clean = text.lower().strip()
        # Direct key matches
        for mood_key in MOOD_PROFILES:
            clean_key = mood_key.split(" ", 1)[-1].lower()
            if clean_key in clean or any(part.strip().lower() in clean for part in clean_key.split("/")):
                return mood_key

        # Keyword matching score
        scores = {}
        for mood_key, keywords in MOOD_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in clean)
            if score > 0:
                scores[mood_key] = score

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        # Default fallback
        return "😊 Happy / Upbeat"

    def recommend(self, mood: str, top_n: int = 10, language: str | None = None) -> pd.DataFrame:
        if mood not in MOOD_PROFILES:
            raise ValueError(f"Unknown mood '{mood}'. Available: {list(MOOD_PROFILES.keys())}")

        target_features = MOOD_PROFILES[mood]
        
        target_vec = np.array([
            target_features["danceability"],
            target_features["energy"],
            target_features["valence"],
            target_features["acousticness"],
            target_features["tempo"] / 200.0,
            (target_features["loudness"] + 20.0) / 20.0,
        ])

        song_vecs = self.scaled_songs[FEATURE_COLUMNS].values
        sims = cosine_similarity([target_vec], song_vecs)[0]

        result = self.songs.copy()
        result["mood_match_score"] = [round(float(s), 3) for s in sims]
        result["mood_match_pct"] = [int(min(99, max(50, s * 100))) for s in sims]
        result["mood"] = mood

        if language and "language" in result.columns:
            result = result[result["language"].str.lower() == language.lower()]
        
        return result.sort_values("mood_match_score", ascending=False).head(top_n).reset_index(drop=True)

    def recommend_by_query(self, query: str, top_n: int = 10, language: str | None = None) -> pd.DataFrame:
        """Accepts free-form mood descriptions or keywords and recommends matching songs."""
        matched_mood = self.detect_mood_from_text(query)
        return self.recommend(matched_mood, top_n=top_n, language=language)

    def recommend_dual(self, mood: str, top_n: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns top matching songs in both English and Malayalam."""
        if mood not in MOOD_PROFILES:
            mood = self.detect_mood_from_text(mood)
        english_songs = self.recommend(mood, top_n=top_n, language="English")
        malayalam_songs = self.recommend(mood, top_n=top_n, language="Malayalam")
        return english_songs, malayalam_songs



if __name__ == "__main__":
    songs = load_songs("data/songs.csv")
    ratings = load_ratings("data/ratings.csv")

    print("=== Content-Based Demo ===")
    cb = ContentBasedRecommender(songs)
    sample_song = songs.iloc[0]
    print(f"Because you liked: {sample_song['title']} ({sample_song['genre']})\n")
    print(cb.recommend(sample_song["song_id"], top_n=5)[["title", "artist", "genre", "similarity"]])

    print("\n=== Collaborative Filtering Demo ===")
    cf = CollaborativeRecommender(ratings, songs)
    sample_user = ratings["user_id"].iloc[0]
    print(f"Recommended for user {sample_user}:\n")
    print(cf.recommend(sample_user, top_n=5)[["title", "artist", "genre", "predicted_rating"]])

    print("\n=== Hybrid Demo ===")
    hybrid = HybridRecommender(cb, cf)
    print(hybrid.recommend(sample_user, sample_song["song_id"], top_n=5)[
        ["title", "artist", "genre", "hybrid_score"]
    ])

    print("\n=== Mood Demo (Happy / Sad / Funny / Hungry) ===")
    mood_rec = MoodRecommender(songs)
    for m in ["😊 Happy / Upbeat", "😢 Sad / Melancholic", "🤪 Funny / Playful", "🍕 Hungry / Munchies & Chill"]:
        print(f"\n--- Mood: {m} ---")
        print(mood_rec.recommend(m, top_n=3)[["title", "artist", "genre", "mood_match_score"]])

