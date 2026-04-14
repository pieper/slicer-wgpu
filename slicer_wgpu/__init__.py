"""slicer_wgpu -- WebGPU-backed rendering for 3D Slicer via pygfx.

Submodules:
    fields           -- Field ABC + ImageField + FiducialField
    scene_renderer   -- per-Field-compositing ray tracer (SceneRenderer)
    displayers       -- MRML-node observers that maintain Fields
    mrml_bridge      -- DualView layout, displayable managers + the
                        SceneRendererManager that wires Displayers to a
                        SceneRenderer (with pick-and-drag handling)
    demos.single_volume -- Stage-1 standalone single-volume renderer
                        (kept for the Graphix demo module)
"""

# 3D Slicer exposes PythonQt's bindings as a flat top-level "qt" module
# (qt.QObject, qt.QWidget, etc.), whereas upstream rendercanvas expects the
# standard PythonQt layout (PythonQt.QtCore.QObject, PythonQt.QtGui.QWidget).
# Install a sys.modules alias so rendercanvas's `from PythonQt import QtCore,
# QtGui` resolves back to Slicer's flat "qt" module. Because every Qt class
# is on that single namespace, sub-module lookups like QtCore.QObject and
# QtGui.QWidget both resolve correctly.
def _install_pythonqt_shim():
    import sys
    import types

    if "qt" not in sys.modules or "PythonQt" in sys.modules:
        return
    _qt = sys.modules["qt"]
    _pythonqt = types.ModuleType("PythonQt")
    _pythonqt.QtCore = _qt
    _pythonqt.QtGui = _qt
    _pythonqt.QtWidgets = _qt
    _pythonqt.Qt = _qt
    sys.modules["PythonQt"] = _pythonqt
    sys.modules["PythonQt.QtCore"] = _qt
    sys.modules["PythonQt.QtGui"] = _qt
    sys.modules["PythonQt.QtWidgets"] = _qt


_install_pythonqt_shim()
del _install_pythonqt_shim


from . import fields, scene_renderer, displayers, mrml_bridge  # noqa: E402
from . import demos  # noqa: E402

__all__ = ["fields", "scene_renderer", "displayers", "mrml_bridge", "demos"]
__version__ = "0.2.0"
