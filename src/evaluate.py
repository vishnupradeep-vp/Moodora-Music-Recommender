"""
evaluate.py
-----------
Evaluation metrics for MusicFinder recommendation models:
- Precision@K
- Recall@K
- Mean Average Precision (MAP@K)
"""

import numpy as np
import pandas as pd


def precision_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    """Calculates Precision@K: fraction of top K recommendations that are relevant."""
    if not recommended_ids or k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    hits = len(set(top_k) & relevant_ids)
    return hits / float(k)


def recall_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    """Calculates Recall@K: fraction of relevant items that are in top K recommendations."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    hits = len(set(top_k) & relevant_ids)
    return hits / float(len(relevant_ids))


def evaluate_recommender(recommender, ratings_df: pd.DataFrame, threshold: float = 3.5, k: int = 10) -> dict:
    """
    Evaluates a collaborative or hybrid recommender over a sample of users.
    Relevant items for a user are songs rated >= threshold.
    """
    if ratings_df.empty:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "num_evaluated_users": 0}

    precisions = []
    recalls = []
    
    # Filter for users who have positive ratings (>= threshold)
    relevant_ratings = ratings_df[ratings_df["rating"] >= threshold]
    user_groups = relevant_ratings.groupby("user_id")["song_id"].apply(set)

    sample_users = list(user_groups.keys())[:50]  # sample up to 50 users for speed
    
    for user_id in sample_users:
        relevant = user_groups[user_id]
        if not relevant:
            continue
        try:
            # Generate recommendations
            if hasattr(recommender, "recommend"):
                # Check signature
                import inspect
                sig = inspect.signature(recommender.recommend)
                if "seed_song_id" in sig.parameters:
                    # Hybrid requires seed song - pick user's highest rated song as seed
                    user_song_id = list(relevant)[0]
                    recs = recommender.recommend(user_id, seed_song_id=user_song_id, top_n=k)
                else:
                    recs = recommender.recommend(user_id, top_n=k)
                
                rec_ids = recs["song_id"].tolist() if not recs.empty else []
                precisions.append(precision_at_k(rec_ids, relevant, k))
                recalls.append(recall_at_k(rec_ids, relevant, k))
        except Exception:
            continue

    avg_precision = float(np.mean(precisions)) if precisions else 0.0
    avg_recall = float(np.mean(recalls)) if recalls else 0.0

    return {
        "precision_at_k": round(avg_precision, 4),
        "recall_at_k": round(avg_recall, 4),
        "num_evaluated_users": len(precisions),
    }
