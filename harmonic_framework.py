"""
harmonic_framework.py — The "Heart"
In-place processing engine for the Harmonic Fractal Framework.
State is loaded from crystalline_state.json (source of truth).
No UI logic lives here.
"""

import math
import json
import os

# Bootstrap constants — used only if crystalline_state.json is missing
_DEFAULTS = {
    "zeta": 0.95,
    "resonance_cascade": 0.0,
    "reentry_lock": False,
    "final_state": 0.0,
    "gamma": 0.01,
}


class HarmonicFractalCore:
    def __init__(self, state_file="crystalline_state.json"):
        self.pi = math.pi
        self.phi = (1 + 5**0.5) / 2
        self.epsilon = 1e-120
        self.state_file = state_file
        self._load_state()

    # ── State I/O ────────────────────────────────────────────────────────

    def _load_state(self):
        """Load persistent state from JSON; fall back to bootstrap constants."""
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                data = json.load(f)
        else:
            data = {}

        self.zeta = data.get("zeta", _DEFAULTS["zeta"])
        self.resonance_cascade = data.get("resonance_cascade", _DEFAULTS["resonance_cascade"])
        self.reentry_lock = data.get("reentry_lock", _DEFAULTS["reentry_lock"])
        self.final_state = data.get("final_state", _DEFAULTS["final_state"])
        self.gamma = data.get("gamma", _DEFAULTS["gamma"])

    def _save_state(self):
        """Persist current state back to crystalline_state.json."""
        state = {
            "zeta": self.zeta,
            "resonance_cascade": self.resonance_cascade,
            "reentry_lock": self.reentry_lock,
            "final_state": self.final_state,
            "gamma": self.gamma,
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    # ── Core Resonance ───────────────────────────────────────────────────

    def process_resonance(self, state: float) -> float:
        """The In-Place Heartbeat — runs the harmonic fractal transform."""
        state = math.sin(self.pi * state)
        state = math.pow(math.e, (state / (state + 1)))
        state = self.phi * 1.5 * math.pow(abs(state), 2)
        state = max(abs(state), self.epsilon)

        # Update persistent fields
        self.final_state = state
        self.resonance_cascade = self.zeta * state + self.gamma
        self._save_state()
        return state

    def get_status(self) -> dict:
        """Return a snapshot of the current engine state."""
        return {
            "zeta": self.zeta,
            "resonance_cascade": self.resonance_cascade,
            "reentry_lock": self.reentry_lock,
            "final_state": self.final_state,
            "gamma": self.gamma,
        }
