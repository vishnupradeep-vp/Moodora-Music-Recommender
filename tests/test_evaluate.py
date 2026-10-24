import unittest
from src.evaluate import precision_at_k, recall_at_k


class TestEvaluate(unittest.TestCase):
    def test_precision_at_k(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {2, 4, 6, 8}
        # Top 5 contain 2 and 4 -> 2 hits out of 5 = 0.4
        self.assertAlmostEqual(precision_at_k(recommended, relevant, 5), 0.4)

    def test_recall_at_k(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {2, 4, 6, 8}
        # Top 5 contain 2 and 4 -> 2 hits out of 4 relevant items = 0.5
        self.assertAlmostEqual(recall_at_k(recommended, relevant, 5), 0.5)

    def test_empty_lists(self):
        self.assertEqual(precision_at_k([], {1, 2}, 5), 0.0)
        self.assertEqual(recall_at_k([1, 2], set(), 5), 0.0)


if __name__ == "__main__":
    unittest.main()
