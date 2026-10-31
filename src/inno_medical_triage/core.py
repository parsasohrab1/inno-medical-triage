"""Decision-support ESI-style triage level suggestion, ED arrival forecast and waiting alerts.

Lab prototype on SYNTHETIC data only (TRL 4). A suggestion for the triage nurse; never an autonomous decision.
Wait-time limits below are illustrative placeholders to be set by the clinical governance team.
"""
import numpy as np

HIGH_RISK = {"chest_pain", "stroke_signs", "sepsis_suspected", "overdose", "severe_bleeding", "suicidal"}
COMPLAINTS = sorted(HIGH_RISK) + ["sprain", "cough", "abdominal_pain", "rash", "headache", "laceration", "fever", "back_pain"]
COMPLAINT_P = np.array([0.25 / len(HIGH_RISK)] * len(HIGH_RISK) + [0.75 / (len(COMPLAINTS) - len(HIGH_RISK))] * (len(COMPLAINTS) - len(HIGH_RISK)))
MAX_WAIT_MIN = {1: 0, 2: 10, 3: 30, 4: 60, 5: 120}

CRITERIA = [
    ("within_one_level", ">=", 0.97, "suggested level within one of reference level"),
    ("undertriage_rate", "<=", 0.05, "ESI 1-2 patients suggested level 3-5 (undertriage)"),
    ("exact_agreement", ">=", 0.75, "exact agreement with reference level"),
    ("arrival_forecast_gain", ">=", 0.10, "hourly arrival forecast MAE gain over same-hour-yesterday"),
]


def suggest_level(p):
    """p: dict age,hr,rr,sbp,spo2,gcs,pain,complaint,needs_lifesaving,expected_resources -> (level, reasons)."""
    reasons = []
    if p["needs_lifesaving"] or p["gcs"] <= 8 or p["spo2"] < 85 or p["sbp"] < 70:
        return 1, ["immediate life-saving intervention indicated"]
    child = p["age"] < 8
    danger = (p["hr"] > (160 if child else 130)) or (p["rr"] > (40 if child else 30)) or p["spo2"] < 92 or p["sbp"] < 90
    if p["complaint"] in HIGH_RISK:
        reasons.append("high-risk complaint: %s" % p["complaint"])
    if p["gcs"] <= 13:
        reasons.append("altered level of consciousness")
    if p["pain"] >= 8:
        reasons.append("severe pain")
    if danger:
        reasons.append("vital signs in danger zone")
    if reasons:
        return 2, reasons
    r = p["expected_resources"]
    lvl = 3 if r >= 2 else 4 if r == 1 else 5
    if lvl == 3 and ((p["hr"] > 100) or (p["rr"] > 20) or p["spo2"] < 94):
        reasons.append("consider up-triage: borderline vital signs")
    return lvl, reasons or ["expected resources: %d" % r]


def waiting_alerts(queue, now_min):
    """queue: list of dict(id, level, arrived_min). Returns patients waiting beyond limit, worst first."""
    out = []
    for q in queue:
        wait = now_min - q["arrived_min"]
        if wait > MAX_WAIT_MIN[q["level"]]:
            out.append({"id": q["id"], "level": q["level"], "waited_min": wait, "over_by": wait - MAX_WAIT_MIN[q["level"]]})
    return sorted(out, key=lambda x: (x["level"], -x["over_by"]))


def _true_patient(rng):
    sev = rng.random()
    comp = rng.choice(COMPLAINTS, p=COMPLAINT_P)
    age = int(rng.integers(1, 90))
    hr = rng.normal(85 + 70 * sev ** 3, 10); rr = rng.normal(16 + 18 * sev ** 3, 2); sbp = rng.normal(125 - 55 * sev ** 3, 10)
    spo2 = min(100, rng.normal(98 - 14 * sev ** 3, 1.5)); gcs = int(np.clip(round(15 - 8 * sev ** 4 * rng.random() * 2), 3, 15))
    pain = int(np.clip(rng.normal(3 + 5 * sev, 1.5), 0, 10))
    res = int(np.clip(round(rng.normal(0.4 + 3 * sev, 0.7)), 0, 3))
    return dict(age=age, hr=hr, rr=rr, sbp=sbp, spo2=spo2, gcs=gcs, pain=pain, complaint=str(comp),
                needs_lifesaving=bool(sev > 0.985), expected_resources=res)


def simulate_patients(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = _true_patient(rng)
        ref, _ = suggest_level(p)
        obs = dict(p)  # what the nurse enters: measurement noise + resource misestimate
        obs["hr"] += rng.normal(0, 3); obs["rr"] += rng.normal(0, 1); obs["sbp"] += rng.normal(0, 4); obs["spo2"] = min(100, obs["spo2"] + rng.normal(0, 0.8))
        obs["pain"] = int(np.clip(obs["pain"] + rng.integers(-1, 2), 0, 10))
        if rng.random() < 0.15:
            obs["expected_resources"] = int(np.clip(obs["expected_resources"] + rng.choice([-1, 1]), 0, 3))
        out.append((obs, ref))
    return out


def arrivals(seed=0, days=60):
    rng = np.random.default_rng(seed + 3)
    prof = 3 + 5 * np.exp(-((np.arange(24) - 14) ** 2) / 40) + 3 * np.exp(-((np.arange(24) - 20) ** 2) / 20)
    dow = rng.uniform(0.9, 1.15, 7)
    lam = np.array([prof * dow[d % 7] for d in range(days)])
    return lam, rng.poisson(lam)


def forecast_arrivals(hist, hour, horizon=6):
    """hist: (days,24) counts up to current day (partial last day up to 'hour'). Forecast next `horizon` hours."""
    full = hist[:-1] if hist.shape[0] > 1 else hist
    prof = full[-14:].mean(0)
    today = hist[-1][:hour]
    ratio = np.clip((today[-6:].sum() + 1) / (prof[max(0, hour - 6):hour].sum() + 1), 0.6, 1.6) if hour > 0 else 1.0
    return np.array([prof[(hour + h) % 24] for h in range(horizon)]) * (0.5 + 0.5 * ratio)


def evaluate(seed=0):
    data = simulate_patients(seed)
    ref = np.array([r for _, r in data]); sug = np.array([suggest_level(o)[0] for o, _ in data])
    lam, y = arrivals(seed)
    em, eb = [], []
    for d in range(20, y.shape[0]):
        for hr in (6, 12, 18):
            f = forecast_arrivals(y[: d + 1], hr); a = y[d, hr:hr + 6] if hr + 6 <= 24 else np.concatenate([y[d, hr:], y[d, :hr + 6 - 24]])
            if hr + 6 > 24:
                continue
            em.append(np.abs(f - a).mean()); eb.append(np.abs(y[d - 1, hr:hr + 6] - a).mean())
    hi = ref <= 2
    return {"within_one_level": float((np.abs(sug - ref) <= 1).mean()), "exact_agreement": float((sug == ref).mean()),
            "undertriage_rate": float((sug[hi] >= 3).mean()), "arrival_forecast_gain": float(1 - np.mean(em) / np.mean(eb)), "n_patients": len(ref)}


def generate(seed=0):
    data = simulate_patients(seed, 800)
    rows = []
    for i, (o, ref) in enumerate(data):
        r = {"id": "T%04d" % i}; r.update({k: (round(float(v), 1) if isinstance(v, (float, np.floating)) else v) for k, v in o.items()})
        r["reference_level"] = ref; r["suggested_level"] = suggest_level(o)[0]; rows.append(r)
    lam, y = arrivals(seed)
    return {"patients": rows, "hourly_arrivals": [dict(day=d, hour=h, arrivals=int(y[d, h])) for d in range(y.shape[0]) for h in range(24)]}


def demo(seed=0):
    p = dict(age=58, hr=118, rr=24, sbp=100, spo2=95, gcs=15, pain=6, complaint="chest_pain", needs_lifesaving=False, expected_resources=3)
    return "suggested level: %s\nwaiting alerts: %s" % (suggest_level(p), waiting_alerts([{"id": "A", "level": 2, "arrived_min": 0}, {"id": "B", "level": 4, "arrived_min": 50}], 25))
