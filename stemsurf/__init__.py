"""stemsurf — AI stem-separation + auto-mix engine.

Pipeline: separate -> analyze -> space -> clarify -> balance -> mixdown.
"""

__version__ = "0.1.0"

from .pipeline import MixPipeline  # noqa: F401
