import unittest
import pandas as pd
import numpy as np

from src.preprocess import FEATURE_COLUMNS, load_songs
from src.recommend import (
    ContentBasedRecommender,
    CollaborativeRecommender,
    HybridRecommender,
    MoodRecommender,
)


def _get_dummy_data():
    songs = pd.DataFrame([
        {
            "song_id": 1, "title": "Song A", "artist": "Artist 1", "genre": "Pop",
            "danceability": 0.8, "energy": 0.7, "valence": 0.6,
            "acousticness": 0.1, "tempo": 120.0, "loudness": -5.0,
        },
        {
            "song_id": 2, "title": "Song B", "artist": "Artist 2", "genre": "Pop",
            "danceability": 0.82, "energy": 0.71, "valence": 0.61,
            "acousticness": 0.11, "tempo": 121.0, "loudness": -5.1,
        },
        {
            "song_id": 3, "title": "Song C", "artist": "Artist 3", "genre": "Rock",
            "danceability": 0.3, "energy": 0.9, "valence": 0.4,
            "acousticness": 0.05, "tempo": 140.0, "loudness": -3.0,
        },
        {
            "song_id": 4, "title": "Song D", "artist": "Artist 4", "genre": "Classical",
            "danceability": 0.1, "energy": 0.2, "valence": 0.2,
            "acousticness": 0.9, "tempo": 80.0, "loudness": -15.0,
        },
    ])

    ratings = pd.DataFrame([
        {"user_id": 101, "song_id": 1, "rating": 5.0},
        {"user_id": 101, "song_id": 2, "rating": 4.0},
        {"user_id": 102, "song_id": 1, "rating": 4.5},
        {"user_id": 102, "song_id": 3, "rating": 5.0},
    ])
    return songs, ratings


class TestContentBasedRecommender(unittest.TestCase):
    def setUp(self):
        self.songs, _ = _get_dummy_data()
        self.recommender = ContentBasedRecommender(self.songs)

    def test_seed_song_is_never_recommended(self):
        # Song 1 and Song 2 are nearly identical in features
        rec = self.recommender.recommend(song_id=1, top_n=2)
        recommended_ids = rec["song_id"].tolist()
        self.assertNotIn(1, recommended_ids)
        self.assertIn(2, recommended_ids)

    def test_recommend_by_title(self):
        rec = self.recommender.recommend_by_title("Song A", top_n=2)
        self.assertEqual(len(rec), 2)
        self.assertNotIn("Song A", rec["title"].tolist())

    def test_invalid_song_id_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.recommender.recommend(9999)


class TestCollaborativeRecommender(unittest.TestCase):
    def setUp(self):
        self.songs, self.ratings = _get_dummy_data()
        self.recommender = CollaborativeRecommender(self.ratings, self.songs)

    def test_recommends_unrated_songs(self):
        # User 101 rated songs 1 and 2
        rec = self.recommender.recommend(user_id=101, top_n=2)
        rec_ids = rec["song_id"].tolist()
        self.assertNotIn(1, rec_ids)
        self.assertNotIn(2, rec_ids)

    def test_small_matrix_does_not_crash_svd(self):
        # Single user, single rating matrix should not crash svds
        small_ratings = pd.DataFrame([{"user_id": 1, "song_id": 1, "rating": 4.0}])
        small_collab = CollaborativeRecommender(small_ratings, self.songs)
        self.assertIsNotNone(small_collab.predicted)

    def test_unknown_user_raises_error(self):
        with self.assertRaises(ValueError):
            self.recommender.recommend(99999)


class TestHybridRecommender(unittest.TestCase):
    def setUp(self):
        self.songs, self.ratings = _get_dummy_data()
        self.cb = ContentBasedRecommender(self.songs)
        self.cf = CollaborativeRecommender(self.ratings, self.songs)
        self.hybrid = HybridRecommender(self.cb, self.cf, alpha=0.5)

    def test_hybrid_recommendation_contains_metadata_and_scores(self):
        rec = self.hybrid.recommend(user_id=101, seed_song_id=1, top_n=3)
        self.assertFalse(rec.empty)
        self.assertIn("hybrid_score", rec.columns)
        self.assertIn("title", rec.columns)
        self.assertFalse(rec["title"].isna().any())


class TestMoodRecommender(unittest.TestCase):
    def setUp(self):
        self.songs, _ = _get_dummy_data()
        from src.recommend import MoodRecommender
        self.mood_rec = MoodRecommender(self.songs)

    def test_happy_mood(self):
        rec = self.mood_rec.recommend("😊 Happy / Upbeat", top_n=2)
        self.assertEqual(len(rec), 2)
        self.assertIn("mood_match_score", rec.columns)

    def test_unknown_mood_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.mood_rec.recommend("InvalidMood")

    def test_recommends_50_tracks_per_language_from_catalogue(self):
        recommender = MoodRecommender(load_songs("data/songs.csv"))

        english, malayalam = recommender.recommend_dual(
            "😊 Happy / Upbeat",
            top_n=50,
        )

        self.assertEqual(len(english), 50)
        self.assertEqual(len(malayalam), 50)
        self.assertTrue(english["language"].eq("English").all())
        self.assertTrue(malayalam["language"].eq("Malayalam").all())


if __name__ == "__main__":
    unittest.main()
