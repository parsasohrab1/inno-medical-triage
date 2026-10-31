import unittest
from inno_medical_triage import core


def pt(**kw):
    d = dict(age=40, hr=80, rr=16, sbp=120, spo2=98, gcs=15, pain=3, complaint="sprain", needs_lifesaving=False, expected_resources=1)
    d.update(kw); return d


class TestTriage(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(core.suggest_level(pt(needs_lifesaving=True))[0], 1)
        self.assertEqual(core.suggest_level(pt(spo2=80))[0], 1)
        self.assertEqual(core.suggest_level(pt(complaint="chest_pain"))[0], 2)
        self.assertEqual(core.suggest_level(pt(expected_resources=3))[0], 3)
        self.assertEqual(core.suggest_level(pt(expected_resources=1))[0], 4)
        self.assertEqual(core.suggest_level(pt(expected_resources=0))[0], 5)

    def test_child_vitals_use_child_limits(self):
        self.assertEqual(core.suggest_level(pt(age=4, hr=140))[0], 4)
        self.assertEqual(core.suggest_level(pt(age=40, hr=140))[0], 2)

    def test_every_suggestion_has_a_reason(self):
        for o, _ in core.simulate_patients(1, 200):
            self.assertTrue(core.suggest_level(o)[1])

    def test_waiting_alerts_order(self):
        q = [{"id": "a", "level": 4, "arrived_min": 0}, {"id": "b", "level": 2, "arrived_min": 0}, {"id": "c", "level": 5, "arrived_min": 90}]
        al = core.waiting_alerts(q, 100)
        self.assertEqual([x["id"] for x in al], ["b", "a"])

    def test_forecast_horizon(self):
        _, y = core.arrivals(0)
        self.assertEqual(len(core.forecast_arrivals(y[:30], 8)), 6)
