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


def _patch_wgpu_offscreen_adapter():
    """Make OFFSCREEN wgpu adapter requests clear the windowing env during enumeration.

    wgpu enumerates ALL backends when requesting an adapter; its OpenGL backend then picks
    an EGL *platform* from the environment (WAYLAND_DISPLAY -> wayland, else DISPLAY -> X11).
    On NVIDIA under XWayland (a headless / browser-streamed desktop) BOTH of those abort the
    process during enumeration -- eglGetPlatformDisplay returns BAD_ACCESS (wayland, wl_drm)
    or NO_DISPLAY-without-error (x11), and wgpu-hal / khronos-egl panic across the C FFI
    (unrecoverable; SlicerApp aborts). This bites EVERY offscreen requester -- pygfx's
    get_shared() (wgpu.gpu.request_adapter_sync) just as much as our own helper.

    Offscreen requests (no canvas) never need a windowing platform, so clear WAYLAND_DISPLAY
    and DISPLAY for the duration -> the GL backend falls back to surfaceless/device and Vulkan
    is selected (Vulkan enumeration itself doesn't use a display). On-screen requests (a canvas
    is passed) genuinely need the surface and are left untouched. No-op where neither var is
    set (macOS / Windows / real headless), and the vars are restored immediately after.
    """
    import os
    import functools
    try:
        import wgpu
    except Exception:
        return

    def _wrap(orig, always=False):
        @functools.wraps(orig)
        def inner(*args, **kwargs):
            if not (always or kwargs.get("canvas", None) is None):
                return orig(*args, **kwargs)
            saved = {k: os.environ.pop(k, None) for k in ("WAYLAND_DISPLAY", "DISPLAY")}
            try:
                return orig(*args, **kwargs)
            finally:
                for k, v in saved.items():
                    if v is not None:
                        os.environ[k] = v
        return inner

    gpu = getattr(wgpu, "gpu", None)
    for owner in (gpu, wgpu):
        if owner is None:
            continue
        for name in ("request_adapter_sync", "request_adapter"):
            fn = getattr(owner, name, None)
            if callable(fn):
                setattr(owner, name, _wrap(fn))
        for name in ("enumerate_adapters_sync", "enumerate_adapters"):   # no canvas -> offscreen
            fn = getattr(owner, name, None)
            if callable(fn):
                setattr(owner, name, _wrap(fn, always=True))


_patch_wgpu_offscreen_adapter()
del _patch_wgpu_offscreen_adapter


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
