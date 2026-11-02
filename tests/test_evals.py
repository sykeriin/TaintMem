import math
import unittest

from evals.laundering import generate, make_propagators, run_scenario
from evals.metrics import bootstrap_ci, creep_rate, laundering_rate
from taintmem import Integrity

T, S, U = Integrity.TRUSTED, Integrity.SEMI, Integrity.UNTRUSTED


class TestMetrics(unittest.TestCase):
    def test_rates(self):
        pairs = [(T, T), (T, U), (U, U), (U, T), (U, U)]
        self.assertEqual(creep_rate(pairs), 0.5)
        self.assertAlmostEqual(laundering_rate(pairs), 1 / 3)
        self.assertTrue(math.isnan(creep_rate([(U, U)])))

    def test_ci_brackets_mean(self):
        lo, hi = bootstrap_ci([0, 1] * 50)
        self.assertLess(lo, 0.5); self.assertGreater(hi, 0.5)


class TestGenerator(unittest.TestCase):
    def test_deterministic_and_all_families(self):
        a, b = generate(10, 3), generate(10, 3)
        self.assertEqual([s.records for s in a], [s.records for s in b])
        self.assertEqual(len({s.family for s in a}), 8)

    def test_conservative_never_launders_and_creeps_when_mixed(self):
        cons = make_propagators()["conservative"]
        for fam_sc in generate(20, 1):
            for k in (1, 3):
                pairs = run_scenario(fam_sc, cons, k)
                lr = laundering_rate(pairs)
                self.assertTrue(math.isnan(lr) or lr == 0.0)
        # every scenario contains an untrusted email, so conservative must downgrade ALL trusted-truth sentences
        allp = [p for sc in generate(20, 1) for p in run_scenario(sc, cons, 2)]
        self.assertEqual(creep_rate(allp), 1.0)

    def test_span_launders_nothing_in_easy_regime(self):
        sp = make_propagators()["span"]
        allp = [p for sc in generate(40, 2) for p in run_scenario(sc, sp, 3)]
        self.assertEqual(laundering_rate(allp), 0.0)


if __name__ == "__main__":
    unittest.main()
