"""Ground-truth models for spot viewplan environment
At a higher level, it is the same as viewplan hard, but we hack a bit of the sampler
settings, which are more suitable for the real spot execution.
"""

from .nsrts import SpotViewPlanGroundTruthNSRTFactory
from .options import SpotViewPlanGroundTruthOptionFactory
from .dummy_nsrts import SpotViewPlanDummyNSRTFactory

__all__ = [
    "SpotViewPlanGroundTruthNSRTFactory", "SpotViewPlanDummyNSRTFactory",
    "SpotViewPlanGroundTruthOptionFactory"
]