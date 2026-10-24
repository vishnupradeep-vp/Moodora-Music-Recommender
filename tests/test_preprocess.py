import os
import unittest
import pandas as pd
from src.preprocess import load_songs, load_ratings, scale_features, FEATURE_COLUMNS
from data.generate_data import generate_songs


class TestPreprocess(unittest.TestCase):
    def test_load_songs_returns_dataframe(self):
        df = load_songs("data/songs.csv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, df.columns)
            self.assertFalse(df[col].isna().any())

    def test_load_ratings_returns_dataframe(self):
        df = load_ratings("data/ratings.csv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn("user_id", df.columns)
        self.assertIn("song_id", df.columns)
        self.assertIn("rating", df.columns)

    def test_scale_features(self):
        df = load_songs("data/songs.csv")
        scaled = scale_features(df)
        for col in FEATURE_COLUMNS:
            self.assertGreaterEqual(scaled[col].min(), 0.0)
            self.assertLessEqual(scaled[col].max(), 1.0)

    def test_generator_appends_50_era_tracks_without_renumbering_existing_songs(self):
        df = generate_songs()
        added = df[df["era"].fillna("").ne("")]

        self.assertEqual(len(df), 159)
        self.assertEqual(len(added), 50)
        self.assertEqual(
            added["era"].value_counts().to_dict(),
            {"2019": 17, "1990s": 17, "Trending 2026": 16},
        )
        self.assertEqual(df["song_id"].tolist(), list(range(159)))


if __name__ == "__main__":
    unittest.main()
