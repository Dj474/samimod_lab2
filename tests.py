"""Юнит-тесты прогона и модели аэропорта:  python -m unittest tests -v"""

import unittest

from airport import AirportModel, HIGH, NORMAL, ST_READY, Plane
from engine import Run, Runner
from simulation import load_config, model_factory


class RecorderModel:
    """Минимальная модель: процессы SimPy пишут момент своего срабатывания."""

    def __init__(self, plan):
        self.plan = plan
        self.log = []

    def start(self, run):
        self.run = run
        for label, delay in self.plan:
            run.env.process(self.worker(label, delay))

    def worker(self, label, delay):
        yield self.run.env.timeout(delay)
        self.log.append((self.run.time, label))

    def finish(self, run):
        pass

    def responses(self):
        return {"n": len(self.log)}


def short_cfg(**over):
    return dict(load_config("config.json"), **dict({"horizon": 500.0}, **over))


class RunTests(unittest.TestCase):
    def test_events_processed_in_time_order_fifo_on_ties(self):
        model = RecorderModel([("c", 3.0), ("a", 1.0), ("b1", 2.0), ("b2", 2.0)])
        Run(model, 10.0, seed=1).simulate()
        self.assertEqual([label for _, label in model.log], ["a", "b1", "b2", "c"])
        self.assertEqual([t for t, _ in model.log], [1.0, 2.0, 2.0, 3.0])

    def test_no_drain_cuts_run_at_horizon(self):
        model = RecorderModel([("in", 4.0), ("out", 6.0)])
        run = Run(model, 5.0, seed=1, drain=False).simulate()
        self.assertEqual([label for _, label in model.log], ["in"])
        self.assertEqual(run.time, 5.0)


class AirportTests(unittest.TestCase):
    def test_same_seed_gives_same_responses(self):
        cfg = short_cfg()
        a = Runner(model_factory, cfg, 1).make_run(3).simulate().responses
        b = Runner(model_factory, cfg, 1).make_run(3).simulate().responses
        self.assertEqual(a, b)

    def test_mass_balance_and_empty_store_after_drain(self):
        for seed in range(1, 6):
            run = Runner(model_factory, short_cfg(), 1).make_run(seed).simulate()
            _, _, err = run.model.balance()
            self.assertLess(err, 1e-6)
            self.assertEqual(len(run.model.warehouse), 0)
            self.assertTrue(all(p.state == ST_READY for p in run.model.planes))

    def test_balance_without_drain_counts_cargo_in_planes(self):
        run = Runner(model_factory, short_cfg(drain=False), 1).make_run(2).simulate()
        self.assertEqual(run.time, 500.0)
        self.assertLess(run.model.balance()[2], 1e-6)

    def test_high_plane_only_when_no_ready_normal(self):
        violations = []
        original = Plane.assign

        def checked_assign(plane):
            if plane.kind == HIGH:
                ready_normal = [p for p in plane.model.planes
                                if p.kind == NORMAL and p.state == ST_READY]
                if ready_normal:
                    violations.append(plane.model.run.time)
            original(plane)

        Plane.assign = checked_assign
        try:
            for seed in range(1, 6):
                Runner(model_factory, short_cfg(), 1).make_run(seed).simulate()
        finally:
            Plane.assign = original
        self.assertEqual(violations, [])

    def test_departure_only_after_full_load_before_horizon(self):
        early_partial = []
        original = Plane.depart

        def checked_depart(plane):
            run = plane.model.run
            if run.time <= run.run_time and plane.loaded < plane.capacity:
                early_partial.append((run.time, plane.loaded))
            original(plane)

        Plane.depart = checked_depart
        try:
            Runner(model_factory, short_cfg(), 1).make_run(4).simulate()
        finally:
            Plane.depart = original
        self.assertEqual(early_partial, [])

    def test_overload_store_grows(self):
        cfg = short_cfg(arrival_rate=4.0, horizon=3000.0, drain=False)
        model = AirportModel(cfg, record=True)
        Run(model, cfg["horizon"], seed=1, drain=False).simulate()
        q = [h[1] for h in model.history]
        third = len(q) // 3
        self.assertGreater(sum(q[-third:]) / third, 3 * sum(q[:third]) / third)


if __name__ == "__main__":
    unittest.main()
