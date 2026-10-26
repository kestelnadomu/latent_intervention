"""
Compatibility alias: the talent SFM moved to ``exp/sim_talent/talent_sfm.py``.

``exp/sim/config.yaml`` (a symlink to ``exp/sim_talent/config.yaml``) and frozen
benchmark protocols still point at ``exp.sim.talent_sfm``. Changing those bytes
would invalidate every trained semantic decoder (they store the config's SHA-256),
so this module makes the old dotted path resolve to the same module object.
"""

import sys

from exp.sim_talent import talent_sfm as _talent_sfm

sys.modules[__name__] = _talent_sfm
