#!/usr/bin/env python3
"""Unit tests for the pure-math core of compute_wnba.py."""

import unittest
from compute_wnba import normalize_name, build_prior, estimate_posterior


class TestNameNormalizer(unittest.TestCase):
    def test_override(self):
        self.assertEqual(normalize_name("A'ja Wilson"), "Aja Wilson")

    def test_accented_chars_stripped(self):
        result = normalize_name("Sabrina Ionescu")
        self.assertEqual(result, "Sabrina Ionescu")

    def test_accented_name(self):
        # é → e after NFD strip
        self.assertEqual(normalize_name("Joséphine"), "Josephine")

    def test_whitespace_stripped(self):
        self.assertEqual(normalize_name("  Maya Moore  "), "Maya Moore")

    def test_pass_through(self):
        self.assertEqual(normalize_name("Breanna Stewart"), "Breanna Stewart")


class TestPriorBuilder(unittest.TestCase):
    def test_full_career(self):
        # 20 games, 10 hits at threshold 20
        games = [22] * 10 + [15] * 10
        prior = build_prior(games, 20)
        # alpha + beta should be close to 20 (career counts + 2 for Laplace)
        self.assertAlmostEqual(prior["alpha"] + prior["beta"], 22, delta=1)
        self.assertEqual(prior["confidence"], "full")

    def test_blended_career(self):
        # 8 games → blended
        games = [22] * 4 + [15] * 4
        prior = build_prior(games, 20)
        total = prior["alpha"] + prior["beta"]
        self.assertGreater(total, 8)
        self.assertLess(total, 12)
        self.assertEqual(prior["confidence"], "partial")

    def test_no_career(self):
        prior = build_prior([], 20)
        self.assertEqual(prior["confidence"], "limited")
        # Should be league-average prior (non-zero)
        self.assertGreater(prior["alpha"], 0)
        self.assertGreater(prior["beta"], 0)

    def test_hit_count_never_exceeds_game_count(self):
        for threshold in [10, 15, 18, 20, 25, 30]:
            games = [25] * 20  # all hits
            prior = build_prior(games, threshold)
            # alpha - 1 (Laplace) should not exceed game count
            self.assertLessEqual(prior["alpha"] - 1, len(games))


class TestBetaBinomialEstimator(unittest.TestCase):
    def test_no_games_returns_prior_mean(self):
        result = estimate_posterior([], 20, 3.0, 7.0)
        self.assertAlmostEqual(result["posterior_p"], 3.0 / 10.0)
        self.assertEqual(result["n_current"], 0)

    def test_all_hits_pulls_toward_1(self):
        result = estimate_posterior([25, 22, 21, 23, 20], 20, 2.0, 8.0)
        self.assertGreater(result["posterior_p"], 2.0 / 10.0)

    def test_all_misses_pulls_toward_0(self):
        result = estimate_posterior([5, 8, 10, 7, 9], 20, 8.0, 2.0)
        self.assertLess(result["posterior_p"], 8.0 / 10.0)

    def test_recency_matters(self):
        # Same 5 games, different order — recent hits (list end = most recent) yield higher p
        prior_a, prior_b = 3.0, 7.0
        hits_recent = [5, 8, 20, 21, 22]   # last 3 are hits (most recent)
        hits_old = [20, 21, 22, 5, 8]       # first 3 are hits (oldest)
        r_recent = estimate_posterior(hits_recent, 20, prior_a, prior_b)
        r_old = estimate_posterior(hits_old, 20, prior_a, prior_b)
        self.assertGreater(r_recent["posterior_p"], r_old["posterior_p"])

    def test_n_current_is_raw_count(self):
        games = [25, 22, 18, 30, 12]
        result = estimate_posterior(games, 20, 2.0, 8.0)
        self.assertEqual(result["n_current"], 5)


if __name__ == "__main__":
    unittest.main()
