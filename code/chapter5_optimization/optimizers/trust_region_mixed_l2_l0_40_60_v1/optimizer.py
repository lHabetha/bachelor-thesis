"""Trust-region optimizer with 40/60 mixed L2/L0 proximity selection."""
from __future__ import annotations

from dataclasses import dataclass

from optimizers._mixed_l2_l0_demo_common import (
    MixedTrustRegion,
)

OPTIMIZER_ID = "trust_region_mixed_l2_l0_40_60_v1"


@dataclass
class TrustRegionMixedL2L0(MixedTrustRegion):
    optimizer_id: str = OPTIMIZER_ID
