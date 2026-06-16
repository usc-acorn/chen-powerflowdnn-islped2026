from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)

class VFModel:
    """Shared voltage-frequency model used across oracle and plot scripts."""

    v_ref: float # 1.1
    f_ref_hz: float # 500Hz for our design

    def delay(self, v: float) -> float:
        return (
            4922
            - 11041*v
            + 8656*v**2
            - 2306*v**3
        )

    def f_hz(self, v):
        v = min(max(v, 0.9), 1.3)
        d_ref = self.delay(self.v_ref)
        dv = self.delay(v)
        return self.f_ref_hz * (d_ref / dv)
    
if __name__ == "__main__":
    x = VFModel(v_ref=1.1, f_ref_hz=500e6)
    for i in [1.3, 1.25, 1.2, 1.15, 1.1, 1.05, 1.0, 0.95, 0.9]:
        print(f"v={i:.2f}V -> f={x.f_hz(i)/1e6:.2f}MHz") 
        print(f"v={i:.2f}V -> f={1/x.f_hz(i)*1e9:.2f}ns") 