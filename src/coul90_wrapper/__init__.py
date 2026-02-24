# Expose coul90_mod functions at package level
# The compiled module has nested structure: coul90_mod.coul90_mod.coul90
# We need to expose it as coul90_mod.coul90 for compatibility
from .coul90_mod import coul90_mod as _inner_mod

class coul90_mod:
    """Wrapper class to provide expected interface"""
    coul90 = staticmethod(_inner_mod.coul90)
    coulph = staticmethod(_inner_mod.coulph)
