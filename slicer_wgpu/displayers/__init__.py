"""Displayers: MRML-node observers that maintain Fields on a SceneRenderer.

Add new node categories by subclassing Displayer; one Displayer per node
class. The SceneRendererBridge wires them all to a common SceneRenderer
and arbitrates structural-vs-uniform updates.
"""

from .base import Displayer
from .volume import VolumeRenderingDisplayer
from .fiducial import FiducialDisplayer

__all__ = ["Displayer", "VolumeRenderingDisplayer", "FiducialDisplayer"]
