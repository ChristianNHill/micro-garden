"""Personality knobs and label presets (ARCHITECTURE.md §2.4, reviewed with Chris 2026-09-16).

Every knob is a 0-1 scale; unlisted knobs sit at 0.5. A duck is a label preset plus jitter, so two ducks
with the same label still differ.
"""
import numpy as np

KNOBS = [
    "aggressiveness", "stink_affinity", "timidity", "curiosity", "sociability", "kindness", "appetite",
    "energy", "sleepiness", "heat_tolerance", "water_love", "boredom_rate", "chattiness", "carelessness",
    "smarts", "playfulness", "music_affinity", "hoarding", "vanity",
]
JITTER = 0.1

LABELS = {
    # the Chao Doctor list
    "Gentle": dict(aggressiveness=0.05, kindness=0.9, sociability=0.7),
    "Naughty": dict(aggressiveness=0.6, kindness=0.2, curiosity=0.7, playfulness=0.8),
    "Energetic": dict(energy=0.9, sleepiness=0.2, playfulness=0.7),
    "Quiet": dict(energy=0.3, chattiness=0.1, sociability=0.3),
    "Big eater": dict(appetite=0.95, aggressiveness=0.5),
    "Chatty": dict(chattiness=0.95, sociability=0.8),
    "Easily bored": dict(boredom_rate=0.9, curiosity=0.6),
    "Curious": dict(curiosity=0.95, timidity=0.2),
    "Carefree": dict(timidity=0.1, water_love=0.8, stink_affinity=0.5),
    "Careless": dict(carelessness=0.9, timidity=0.1),
    "Smart": dict(smarts=0.95),
    "Cry baby": dict(timidity=0.9, aggressiveness=0.05, curiosity=0.2),  # low curiosity: sorrow fades slowly
    "Lonely": dict(sociability=0.95),
    "Naive": dict(smarts=0.2, timidity=0.1, curiosity=0.7),
    "No personality": dict(),
    # ours
    # seeks company and gets hungry fast, so food defense comes up often (Gate 5 retest, 2026-09-16)
    "Bully": dict(aggressiveness=0.95, kindness=0.1, appetite=0.9, sociability=0.8),
    "Zoomer": dict(energy=0.95, boredom_rate=0.8, playfulness=0.8),
    "Napper": dict(sleepiness=0.95, energy=0.3, heat_tolerance=0.3),
    "Show-off": dict(vanity=0.9, playfulness=0.8, sociability=0.8),
    "Scaredy": dict(timidity=0.95, sociability=0.4, stink_affinity=0.0),
    "Loner": dict(sociability=0.05, curiosity=0.6, timidity=0.3),
}


def preset(label: str, rng: np.random.Generator | None = None) -> dict[str, float]:
    """Knob values for a label, jittered when rng is given."""
    unknown = set(LABELS[label]) - set(KNOBS)
    assert not unknown, f"{label}: unknown knobs {unknown}"
    k = {name: LABELS[label].get(name, 0.5) for name in KNOBS}
    if rng is not None:
        k = {name: float(np.clip(v + rng.uniform(-JITTER, JITTER), 0, 1)) for name, v in k.items()}
    return k


def stack(ducks: list[dict[str, float]]) -> dict[str, np.ndarray]:
    """Per-knob arrays, one value per duck."""
    return {name: np.array([d[name] for d in ducks]) for name in KNOBS}


if __name__ == "__main__":
    for label in LABELS:
        preset(label)
    rng = np.random.default_rng(0)
    a, b = preset("Bully", rng), preset("Bully", rng)
    assert a != b and all(0 <= v <= 1 for v in a.values())
    assert preset("No personality") == {k: 0.5 for k in KNOBS}
    print(f"{len(LABELS)} labels over {len(KNOBS)} knobs ok")
