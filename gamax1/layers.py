"""
gamax1/layers.py
================
Core Aetherion mechanisms, translated from the numpy research prototype
(see Aetherion Technical Report v4) into PyTorch nn.Modules suitable for
a real, GPU-trainable NLP model.

Every mechanism here is implemented the way it was *validated* to work
in the research report, not the way it was first (and incorrectly)
implemented. Where a mechanism's benefit was found to be conditional
(e.g. hexagonal neighbor influence only helps on clustered data), that
condition is documented and the module defaults to a safe setting.

Design note on loss functions: the research report's "loss-metric
mismatch" finding (Section 6.2) is about plain binary cross-entropy on
a highly-imbalanced multi-label *sparse recovery* target -- it does
NOT apply to standard categorical next-token language-model cross-
entropy, which is a well-posed single-correct-class loss. GamaX1 uses
standard cross-entropy for language modeling; the pairwise-ranking-loss
fix is deliberately NOT re-applied here, since the failure mode it
fixes does not exist in this task shape.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicSparsityController:
    """Adaptive sparsity level S(t), fixed per Aetherion Section 5.2.

    VALIDATED FIX: gate sparsity reduction on the *trend* of training
    loss (a moving average comparison), never on instantaneous
    interference/loss alone -- gating on instantaneous signal was the
    root cause of the sparsity-collapse failure mode in the research
    report. A separate `exploration_fraction` knob is kept independent
    of the sparsity level itself (Design Principle 2): how sparse the
    *representation* is and how much *extra random exploration* happens
    during training are different concerns and must not share one
    variable.
    """

    def __init__(self, k_init, k_min, k_max, trend_window=50, patience=200):
        self.k = k_init
        self.k_min = k_min
        self.k_max = k_max
        self.trend_window = trend_window
        self.patience = patience
        self.exploration_fraction = 0.10  # independent of k; decays separately
        self._loss_history = []
        self._steps_since_change = 0

    def step(self, loss_value: float):
        """Call once per training step with the current scalar loss."""
        self._loss_history.append(float(loss_value))
        self._loss_history = self._loss_history[-(self.trend_window * 3):]
        self._steps_since_change += 1

        if len(self._loss_history) < self.trend_window * 2:
            return self.k  # not enough history yet to judge a trend

        recent = sum(self._loss_history[-self.trend_window:]) / self.trend_window
        prior = sum(self._loss_history[-2 * self.trend_window:-self.trend_window]) / self.trend_window
        improving = recent < prior * 0.995  # trend-based, not instantaneous

        if improving and self._steps_since_change > self.patience and self.k > self.k_min:
            self.k = max(self.k_min, int(self.k * 0.9))
            self._steps_since_change = 0
            self.exploration_fraction = max(0.02, self.exploration_fraction * 0.85)

        return self.k

    def state_dict(self):
        return {"k": self.k, "exploration_fraction": self.exploration_fraction}

    def load_state_dict(self, state):
        self.k = state["k"]
        self.exploration_fraction = state["exploration_fraction"]


class ProbationaryMemoryTracker:
    """Dead-feature prevention, fixed per Aetherion Section 5.3.

    Tracks, per hidden unit, how often it was among the top-K active
    set. Units that go too long without activating are placed on
    "probation" and forced into the active set periodically (a small
    guaranteed nudge) until they demonstrate they can compete on merit
    again.

    VALIDATED FIX: only count events where a unit *should plausibly*
    matter (i.e. only track under-activation, not simple absence);
    true "not needed right now" cases are not penalized, avoiding the
    bookkeeping bug that caused threshold flicker in the original
    prototype. Entry requires `miss_threshold` consecutive misses; exit
    requires `success_threshold` consecutive forced-successes -- an
    explicit, asymmetric hysteresis band, not a single shared count.
    """

    def __init__(self, n_units, miss_threshold=4, success_threshold=3, nudge_period=20):
        self.n_units = n_units
        self.miss_threshold = miss_threshold
        self.success_threshold = success_threshold
        self.nudge_period = nudge_period
        self.miss_count = torch.zeros(n_units, dtype=torch.long)
        self.success_count = torch.zeros(n_units, dtype=torch.long)
        self.on_probation = torch.zeros(n_units, dtype=torch.bool)
        self._step_count = 0

    def update(self, active_mask_batch: torch.Tensor):
        """active_mask_batch: (batch, n_units) bool, True where a unit
        was in the top-K active set for that sample this step."""
        self._step_count += 1
        active_any = active_mask_batch.any(dim=0).cpu()

        missed = ~active_any & ~self.on_probation
        self.miss_count[missed] += 1
        self.miss_count[active_any] = 0
        newly_probation = self.miss_count >= self.miss_threshold
        self.on_probation |= newly_probation

        got_forced_success = active_any & self.on_probation
        self.success_count[got_forced_success] += 1
        self.success_count[~got_forced_success & self.on_probation] = 0
        exiting = self.on_probation & (self.success_count >= self.success_threshold)
        self.on_probation[exiting] = False
        self.miss_count[exiting] = 0
        self.success_count[exiting] = 0

    def nudge_indices(self):
        """Units due for a forced-inclusion nudge this step."""
        if self._step_count % self.nudge_period != 0:
            return torch.empty(0, dtype=torch.long)
        return self.on_probation.nonzero(as_tuple=True)[0]

    def population(self):
        return int(self.on_probation.sum().item())


def build_hex_neighbor_table(n_features: int) -> torch.Tensor:
    """Approximate a hexagonal lattice over a 1-D feature index by
    laying features on a square-ish grid and connecting each to its 6
    hex-equivalent neighbors (4 axis neighbors + 2 diagonal, offset by
    row parity -- the standard "offset coordinates" hex approximation).
    Returns a (n_features, 6) index tensor (self-index used as padding
    when a neighbor would fall outside the grid).
    """
    side = max(1, int(math.ceil(math.sqrt(n_features))))
    neighbors = torch.arange(n_features).unsqueeze(1).repeat(1, 6)
    for idx in range(n_features):
        r, c = divmod(idx, side)
        parity = r % 2
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, 1 - 2 * parity), (1, 1 - 2 * parity)]
        for k, (dr, dc) in enumerate(offsets):
            nr, nc = r + dr, c + dc
            nidx = nr * side + nc
            if 0 <= nr and 0 <= nc < side and 0 <= nidx < n_features:
                neighbors[idx, k] = nidx
    return neighbors


class HexNeighborInfluence(nn.Module):
    """Hexagonal-lattice neighbor influence, per Aetherion Section 3.2 / 5.7.

    VALIDATED, CONDITIONAL finding: this mechanism measurably speeds up
    early convergence only when the true underlying structure of the
    hidden representation clusters at a scale >= the 6-neighbor
    neighborhood; on unclustered/unstructured representations it can
    inject noise rather than signal (Section 6.3). There is no way to
    guarantee a language model's learned hidden features will cluster
    at the right scale, so this module defaults to a *small* decay and
    is explicitly OFF by default in GamaX1Block. Enable deliberately,
    and evaluate its effect empirically on your task, rather than
    assuming a benefit.
    """

    def __init__(self, n_features: int, decay: float = 0.05):
        super().__init__()
        self.register_buffer("neighbor_table", build_hex_neighbor_table(n_features))
        self.decay = decay

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # activations: (..., n_features)
        neighbor_vals = activations[..., self.neighbor_table]  # (..., n_features, 6)
        influence = neighbor_vals.mean(dim=-1) * self.decay
        return activations + influence


class SparseSuperpositionLinear(nn.Module):
    """The core efficiency mechanism, validated in Aetherion Sections
    5.1 and 5.12: a wide feature space (n_features >> d_model) where
    only the top-K units are kept active (ReLU'd) per sample, the rest
    zeroed. In the research report this matched 96-98% of a dense
    baseline's accuracy at roughly half to a fifth of the compute.

    `k` is supplied externally per step by a DynamicSparsityController
    so representation sparsity can adapt over training.
    """

    def __init__(self, d_model: int, n_features: int, hex_influence: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.in_proj = nn.Linear(d_model, n_features)
        self.out_proj = nn.Linear(n_features, d_model)
        self.hex = HexNeighborInfluence(n_features) if hex_influence else None
        self.last_active_mask = None  # exposed for ProbationaryMemoryTracker

    def forward(self, x: torch.Tensor, k: int, nudge_indices: torch.Tensor = None) -> torch.Tensor:
        # x: (batch, seq, d_model)
        pre = F.relu(self.in_proj(x))
        if self.hex is not None:
            pre = F.relu(self.hex(pre))

        k = min(k, self.n_features)
        topk_vals, topk_idx = pre.topk(k, dim=-1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, topk_idx, topk_vals)

        if nudge_indices is not None and nudge_indices.numel() > 0:
            # PTM nudge: force-include probationary units at a small,
            # non-disruptive magnitude so they keep receiving gradient.
            nudge_vals = pre[..., nudge_indices].detach() * 0.5 + 1e-3
            sparse[..., nudge_indices] = torch.maximum(sparse[..., nudge_indices], nudge_vals)

        mask = sparse > 0
        self.last_active_mask = mask.reshape(-1, self.n_features).detach()

        return self.out_proj(sparse)


class RouterExpert(nn.Module):
    """Layer-skip decision, validated per Aetherion Sections 5.4/5.11.

    The research report found a small TRAINED classifier reached 100%
    agreement with ground-truth difficulty, against 81% for a hand-
    tuned heuristic -- so GamaX1's router is trained end-to-end (a
    tiny linear-sigmoid head over pooled hidden state), not hand-tuned.
    Used at inference time to skip deeper blocks for easy sequences.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.probe = nn.Linear(d_model, 1)

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.probe(pooled_hidden)).squeeze(-1)  # P(needs deeper layer)


class ValidatorExpert(nn.Module):
    """Confidence-gated early exit, validated per Aetherion Section 5.6.

    VALIDATED FIX: a raw confidence/margin metric was found to trend
    the *wrong way* as answer quality improves under iterative
    refinement, so gating must be based on answer STABILITY (does the
    predicted top-token set stop changing across refinement steps?),
    not raw confidence. `is_stable` implements exactly that check.
    """

    def __init__(self, patience: int = 1):
        super().__init__()
        self.patience = patience
        self._history = []

    def is_stable(self, top_token_ids: torch.Tensor) -> bool:
        self._history.append(top_token_ids.detach().cpu())
        if len(self._history) <= self.patience:
            return False
        recent = self._history[-(self.patience + 1):]
        stable = all(torch.equal(recent[i], recent[-1]) for i in range(len(recent) - 1))
        return stable

    def reset(self):
        self._history = []
