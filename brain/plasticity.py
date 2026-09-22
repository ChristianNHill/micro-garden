"""Mushroom body learning: dopamine-gated depression at Kenyon cell -> MBON synapses.

These 21,438 synapses are the only mutable ones in the brain, and they are per duck: everything else is
the shared connectome. The rule is the standard one (Aso and Rubin 2016): a Kenyon cell that fired
recently loses its grip on an MBON while that MBON's dopamine is high, so an odor paired with sugar
stops driving the MBONs that say "leave it alone", and the duck approaches.

Which MBON is under which dopamine neuron comes from the connectome rather than a compartment table:
PAM and PPL1 each gate the MBONs they actually synapse onto. PAM carries reward (sugar, petting) and
PPL1 punishment (startle). Neither has a wired path from the senses (sugar to PAM is zero edges in
FlyWire v783), so both are driven as events (explicit code, not the wiring), the way the encoder drives a sense.

Forgetting runs on two clocks. Most of a lesson fades over RECOVER_S. A share of it, CONSOLIDATE, is
laid down in a slow copy of the weights that takes SLOW_RECOVER_S to come back, and the fast weights
recover toward that copy rather than to baseline, so a duck petted today still knows the hand tomorrow.
"""
import numpy as np
import torch

from brain.lif import DT_MS, SYN_GAIN

# The mushroom body's own synapse, stronger than the brain's uniform SYN_GAIN. Kenyon cells are 77% of an
# MBON's excitatory input by weight, but once sparsened they supplied only 22% of its firing and learning
# could not move the MBONs. At x10 they supply 79%, which matches the wiring.
MB_GAIN = 10.0
# Extra spike threshold for Kenyon cells, on top of the shared one. Without it 28% of them answer any odor
# and two odors share 79% of their cells, so nothing can be learned about one smell in particular. At +1.0
# about 5% respond, as measured in the fly (Turner et al. 2008), and the overlap falls to 0.44. At +2.0 they
# go silent.
KC_THRESHOLD = 1.0
KC_TRACE_MS = 1000.0  # how long a Kenyon cell stays eligible after firing; the pairing window
DEPRESS_PER_S = 3.0  # fraction of a synapse's remaining weight lost per second of full coincidence
# A dopamine neuron's burst to a shock or to sugar lasts about a second, but a garden event (a clap, a
# bite, a hand on the head) is one 20 ms body step, too short to teach anything. So each event leaves
# dopamine that fades over DOPAMINE_S.
DOPAMINE_S = 1.0
RECOVER_S = 300.0  # weights drift back over about five minutes: the forgetting curve
# What is left once the fast part has gone. A design choice, not a measurement: a fifth of a lesson
# kept is enough to see tomorrow and small enough that a duck still visibly gets over things today.
CONSOLIDATE = 0.2
SLOW_RECOVER_S = 3 * 24 * 3600.0  # three days, the same horizon as catch-up (brain/save.py MAX_GAP_S)


class Plasticity:
    """Per-duck KC->MBON weights. Supply the current these synapses deliver; the shared matrix must
    have had its KC->MBON block removed by `strip` or the current is counted twice."""

    def __init__(self, W, sets: dict, batch: int, device: str, smarts=0.5):
        """smarts is the personality knob, one per duck or a scalar: a quick learner's synapses depress
        1.5 times as fast as an ordinary duck's (0.5) and a slow one's half as fast."""
        block = W[sets["MBON"]][:, sets["kenyon_cells"]].tocoo()
        self.post = torch.from_numpy(sets["MBON"][block.row].astype(np.int64)).to(device)
        self.pre = torch.from_numpy(sets["kenyon_cells"][block.col].astype(np.int64)).to(device)
        self.pre_local = torch.from_numpy(block.col.astype(np.int64)).to(device)  # into trace, not the brain
        self.base = torch.from_numpy((block.data * SYN_GAIN * MB_GAIN).astype(np.float32)).to(device)
        self.w = self.base.repeat(batch, 1)  # (batch, edges), the only weights that ever change
        self.slow = self.base.repeat(batch, 1)  # where w recovers to: the part of a lesson that lasts
        self.trace = torch.zeros((batch, len(sets["kenyon_cells"])), device=device)
        self.dopamine = torch.zeros((2, batch, 1), device=device)  # reward, punishment: what is left of each burst
        self.kc = torch.from_numpy(sets["kenyon_cells"].astype(np.int64)).to(device)
        # an edge is gated by a dopamine group only if the group synapses onto its MBON: the compartment
        # map, taken from the wiring
        gate = {}
        for name in ("PAM", "PPL1"):
            onto = np.asarray(abs(W[sets["MBON"]][:, sets[name]]).sum(1)).ravel()
            gate[name] = torch.from_numpy((onto[block.row] > 0).astype(np.float32)).to(device)
        self.reward_gate, self.punish_gate = gate["PAM"], gate["PPL1"]
        self.device = device
        self.n = W.shape[0]
        self.rate = torch.as_tensor(np.broadcast_to(0.5 + np.asarray(smarts, np.float32), batch).copy(),
                                    device=device).reshape(-1, 1)

    def current(self, spikes: torch.Tensor) -> torch.Tensor:
        """(batch, n) synaptic current these synapses deliver for this tick's Kenyon cell spikes."""
        fired = spikes[:, self.pre].float()
        out = torch.zeros((spikes.shape[0], self.n), device=self.device)
        out.scatter_add_(1, self.post.expand(spikes.shape[0], -1), self.w * fired)
        return out

    def step(self, spikes: torch.Tensor, reward: np.ndarray, punish: np.ndarray) -> None:
        """Advance one tick. reward and punish are dopamine levels in [0, 1], one per duck."""
        decay = np.exp(-DT_MS / KC_TRACE_MS)
        self.trace.mul_(decay).add_(spikes[:, self.kc].float()).clamp_(max=1.0)
        as_col = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=self.device).reshape(-1, 1)
        self.dopamine = torch.maximum(self.dopamine * float(np.exp(-DT_MS / 1000 / DOPAMINE_S)),
                                      torch.stack([as_col(reward), as_col(punish)]))
        dope = (self.dopamine[0] * self.reward_gate + self.dopamine[1] * self.punish_gate).clamp_(0, 1)
        eligible = self.trace.gather(1, self.pre_local.expand(self.trace.shape[0], -1))
        lost = self.w * eligible * dope * self.rate * (DEPRESS_PER_S * DT_MS / 1000)
        self.w -= lost
        self.slow -= CONSOLIDATE * lost
        self.rest(DT_MS / 1000)

    def rest(self, s: float) -> None:
        """Forget for s seconds with nothing happening: one tick of it, or a night away."""
        self.w.clamp_(min=0)
        self.slow.clamp_(min=0)
        before = self.slow.clone()
        self.slow += (self.base - self.slow) * (1 - np.exp(-s / SLOW_RECOVER_S))
        self.w += (self.slow - self.w) * (1 - np.exp(-s / RECOVER_S)) + (self.slow - before)
        torch.minimum(self.w, self.slow, out=self.w)

    def fondness(self) -> np.ndarray:
        """What this duck has learned about the smell in its nose right now, -1 to 1, per duck.

        For the Kenyon cells just firing, the share of reward-gated (PAM) weight lost is how much the duck
        likes this smell, and the share of punishment-gated (PPL1) weight lost is how much it dreads it.
        This is an explicit readout, like music's: read from the synapses because MBON rates also move with
        smell strength, and nothing downstream of the MBONs in this wiring turns a duck.
        """
        nose = self.trace.gather(1, self.pre_local.expand(self.trace.shape[0], -1))
        gone = (1 - self.w / self.base).clamp_(min=0) * nose
        share = lambda gate: (gone * gate).sum(1) / ((nose * gate).sum(1) + 1e-6)
        return (share(self.reward_gate) - share(self.punish_gate)).cpu().numpy()

    def strength(self) -> np.ndarray:
        """Mean weight as a fraction of baseline, per duck. 1.0 is untouched, lower is learned."""
        return (self.w.sum(1) / self.base.sum()).cpu().numpy()


def sparsen(brain, sets: dict) -> None:
    """Raise the Kenyon cells' threshold so an odor lights a sparse few of them, as in the fly."""
    brain.thresh_offset[torch.from_numpy(sets["kenyon_cells"]).to(brain.dev)] = KC_THRESHOLD


def strip(W, sets: dict):
    """The shared matrix without its KC->MBON block, which Plasticity now owns."""
    W = W.tolil(copy=True)
    W[np.ix_(sets["MBON"], sets["kenyon_cells"])] = 0
    return W.tocsr()


def demo() -> None:
    """Coincidence depresses, either alone does not, and the weights come back (Gate 7)."""
    from scipy import sparse

    n, kc, mbon = 60, np.arange(0, 50), np.arange(50, 58)
    # 50 Kenyon cells onto MBON 50, and PAM 59 innervating that MBON so it has a compartment to gate
    rows = np.append(np.repeat(50, 50), 50)
    cols = np.append(kc, 59)
    W = sparse.csr_matrix((np.append(np.full(50, 10.0), 5.0), (rows, cols)), shape=(n, n))
    sets = {"kenyon_cells": kc, "MBON": mbon, "PAM": np.array([59]), "PPL1": np.array([58])}
    spikes = torch.zeros((1, n), dtype=torch.bool)
    spikes[0, kc[:10]] = True

    p = Plasticity(W, sets, 1, "cpu")
    for _ in range(100):  # 1 s of odor with no reward
        p.step(spikes, np.array([0.0]), np.array([0.0]))
    assert abs(p.strength()[0] - 1.0) < 1e-3, f"odor alone must not teach: {p.strength()[0]}"

    p = Plasticity(W, sets, 1, "cpu")
    for _ in range(100):  # 1 s of reward with no odor
        p.step(torch.zeros_like(spikes), np.array([1.0]), np.array([0.0]))
    assert abs(p.strength()[0] - 1.0) < 1e-3, f"reward alone must not teach: {p.strength()[0]}"

    p = Plasticity(W, sets, 1, "cpu")
    for _ in range(100):  # 1 s of the two together
        p.step(spikes, np.array([1.0]), np.array([0.0]))
    paired = p.strength()[0]
    assert paired < 0.95, f"pairing must depress: {paired}"

    # smarts is a scale: three ducks given the same lesson, slow, ordinary and quick
    p3 = Plasticity(W, sets, 3, "cpu", smarts=np.array([0.05, 0.5, 0.95]))
    for _ in range(100):
        p3.step(spikes.expand(3, -1), np.ones(3), np.zeros(3))
    slow, mid, quick = p3.strength()
    assert slow > mid > quick and abs(mid - paired) < 1e-6, f"smarts must grade: {slow} {mid} {quick}"

    quiet = torch.zeros_like(spikes)
    for _ in range(int(RECOVER_S * 1000 / DT_MS)):  # one recovery time constant of nothing
        p.step(quiet, np.array([0.0]), np.array([0.0]))
    # One time constant gives back 1 - 1/e of the fast part: 0.51 of the way. Ask for 0.8 of that, since
    # dopamine and the Kenyon cell trace outlast the pairing and deepen the lesson a little after `paired`.
    due = 0.8 * (1 - CONSOLIDATE) * (1 - np.exp(-1))
    assert p.strength()[0] > paired + due * (1 - paired), f"must forget: {paired} -> {p.strength()[0]}"
    after = p.strength()[0]
    p.rest(8 * 3600.0)  # a night away: the fast part is gone, the consolidated part is not
    kept = 1 - p.strength()[0]
    want = CONSOLIDATE * (1 - paired)
    assert 0.8 * want < kept < want, f"a night must leave most of the consolidated fifth: {kept} of {want}"
    p.rest(10 * SLOW_RECOVER_S)
    assert p.strength()[0] > 0.999, f"and a month must not: {p.strength()[0]}"
    print(f"ok  odor alone 1.000, reward alone 1.000, paired {paired:.3f} "
          f"(slow {slow:.3f}, quick {quick:.3f}), after {RECOVER_S:.0f} s {after:.3f}, "
          f"after a night {1 - kept:.3f}")


if __name__ == "__main__":
    demo()
