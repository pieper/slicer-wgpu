# slicer-wgpu

WebGPU-backed rendering for [3D Slicer](https://slicer.org) via [pygfx](https://github.com/pygfx/pygfx).

This package provides:

- **`slicer_wgpu.volume_renderer`** — a standalone pygfx `WorldObject` + `Material` + WGSL shader
  implementing direct volume rendering ported from [STEP](https://github.com/pieper/step) with
  fixes inspired by the earlier OpenCL-based
  [RenderCL](https://github.com/pieper/SlicerCL/blob/master/RenderCL/Render.cl.in) shader from
  [SlicerCL](https://github.com/pieper/SlicerCL): ray-inset, VTK-style gradient-opacity LUT,
  sRGB Phong shading, linear compositing, per-pixel headlight.
- **`slicer_wgpu.mrml_bridge`** — displayable managers that mirror a Slicer MRML scene into a pygfx
  scene (models, volume rendering, segmentations, camera, view), plus a `DualView` layout that
  installs a pygfx widget alongside Slicer's native VTK 3D view for side-by-side comparison.

## Install

```python
# From inside 3D Slicer's Python console:
import slicer
slicer.util.pip_install(
    "https://github.com/pieper/rendercanvas/archive/refs/heads/main.zip"
)
slicer.util.pip_install(
    "https://github.com/pieper/slicer-wgpu/archive/refs/heads/main.zip"
)
```

The `rendercanvas` install uses [pieper/rendercanvas](https://github.com/pieper/rendercanvas)
because the upstream package does not yet support PythonQt (Slicer's Qt binding). Once the
PythonQt support PR is merged upstream, the first step can be dropped.

## Use

```python
from slicer_wgpu import mrml_bridge
dv = mrml_bridge.install()              # side-by-side pygfx + VTK 3D panes
# ... load volumes / models / segmentations; both panes mirror the scene ...
mrml_bridge.uninstall()                  # restore original layout
```

## Status

Stage 1 (single volume) complete. See
[the SlicerWGPU Graphix module self-test](https://github.com/pieper/SlicerWGPU) for a scripted
end-to-end smoke test.
