"""Demo / standalone components that aren't part of the Scene Renderer pipeline.

- single_volume: the original single-volume STEP-port renderer. Useful as
  a minimal example and for the Graphix demo module. Production code uses
  slicer_wgpu.scene_renderer instead.
"""

from . import single_volume

__all__ = ["single_volume"]
