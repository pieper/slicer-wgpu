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


def _force_vulkan_only_wgpu_instance():
    """On Linux, create the wgpu instance with only the Vulkan backend enabled.

    wgpu enumerates EVERY backend when it first creates its instance / requests an adapter.
    Its OpenGL-ES backend opens an EGL *platform* display chosen from the environment
    (WAYLAND_DISPLAY -> wayland, else DISPLAY -> X11). On NVIDIA under XWayland (a headless /
    browser-streamed desktop) that EGL probe aborts the whole process -- eglGetPlatformDisplay
    returns BAD_ACCESS and wgpu-hal panics across the C FFI (unrecoverable; SlicerApp aborts).
    This bites EVERY requester -- pygfx's get_shared() just as much as our own helper.

    Clearing the windowing env around the request does NOT help (the GL backend is enumerated
    regardless of when the env is clear, and WGPU_BACKEND only changes adapter *selection*).
    The fix is to create the instance with ONLY the Vulkan backend, so the GL/ES backend is
    never created and its probe never runs. Vulkan WSI still serves offscreen and on-screen
    surfaces, and DISPLAY is left untouched so VTK/GLX rendering is unaffected.

    Linux-only: macOS uses Metal and Windows uses DX12/Vulkan, where this restriction would
    remove the only available backend. Override the backend list with
    SLICER_WGPU_INSTANCE_BACKENDS (comma-separated, e.g. "Vulkan,GL"). Idempotent, and a no-op
    once the wgpu instance already exists (then it is too late to choose backends).
    """
    import os
    import sys

    if not sys.platform.startswith("linux"):
        return
    try:
        import wgpu
        if getattr(wgpu, "_slicer_wgpu_instance_extras_set", False):
            return
        from wgpu.backends.wgpu_native import _helpers
        if _helpers._the_instance is not None:
            return   # too late -- instance already created with all backends
        backends = [b.strip() for b in
                    os.environ.get("SLICER_WGPU_INSTANCE_BACKENDS", "Vulkan").split(",")
                    if b.strip()]
        from wgpu.backends.wgpu_native.extras import set_instance_extras
        set_instance_extras(backends=backends)
        wgpu._slicer_wgpu_instance_extras_set = True
    except Exception as exc:
        print(f"slicer_wgpu: could not restrict wgpu instance to Vulkan: {exc}")


_force_vulkan_only_wgpu_instance()
del _force_vulkan_only_wgpu_instance


# Base submodules — these do NOT depend on rendercanvas, so the offscreen VTK-injection
# path (which only needs fields/displayers/scene_renderer) stays rendercanvas-free.
from . import fields, scene_renderer, displayers  # noqa: E402

# mrml_bridge (DualView / QRenderWidget) and the demos pull in rendercanvas.qt and the whole
# Qt-canvas / screen-present stack. Import them LAZILY so merely importing slicer_wgpu (or
# slicer_wgpu.fields) for injection never loads rendercanvas. Explicit access still works:
# `from slicer_wgpu import mrml_bridge` / `slicer_wgpu.mrml_bridge` trigger the import on demand.
_LAZY = {"mrml_bridge", "demos"}


def __getattr__(name):  # PEP 562 module-level lazy attributes
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["fields", "scene_renderer", "displayers", "mrml_bridge", "demos"]
__version__ = "0.2.0"
