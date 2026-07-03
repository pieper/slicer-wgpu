"""slicer_wgpu.headless -- Slicer-free offscreen volume renderer.

A thin driver that renders a numpy volume to a numpy RGBA frame with no
dependency on 3D Slicer, VTK, or Qt at call time. It reuses the pygfx
ray-march volume renderer from ``slicer_wgpu.demos.single_volume``
(``SlicerVolumeRenderer`` / ``SlicerVolumeMaterial`` + WGSL) and drives it
through a pygfx offscreen canvas, so it runs anywhere wgpu has an adapter
(Metal on macOS, Vulkan in a headless Linux/GPU container).

This is the "rendering" half of the SlicerLive remote-render seam: the web /
encode / transport half lives in the SlicerLive repo and calls::

    r = HeadlessVolumeRenderer(1024, 768)
    r.set_volume(array, spacing=(sx, sy, sz))
    r.set_transfer_function(color_points, opacity_points, scalar_range)
    r.frame_volume()                 # auto-fit the camera once
    rgba = r.render()                # -> (H, W, 4) uint8, on every camera move
    r.orbit(d_az_deg, d_el_deg); r.dolly(factor)   # apply browser input

Camera semantics mirror VTK's Azimuth/Elevation/Dolly so the swap in the
existing modal harness is behaviour-preserving.
"""

from __future__ import annotations

import math

import numpy as np
import pygfx

from .demos.single_volume import SlicerVolumeRenderer, SlicerVolumeMaterial


# ---------------------------------------------------------------------------
# Transfer-function helpers (numpy; no MRML / VTK)
# ---------------------------------------------------------------------------

def build_color_opacity_lut(color_points, opacity_points, scalar_range, n=256):
    """Build an (n, 4) float32 RGBA LUT by interpolating control points.

    color_points:   list of (scalar, r, g, b)   with r,g,b in [0, 1]
    opacity_points: list of (scalar, alpha)      with alpha in [0, 1]
    scalar_range:   (lo, hi) domain the LUT is sampled over (maps to [0, 1]).
    """
    lo, hi = float(scalar_range[0]), float(scalar_range[1])
    xs = np.linspace(lo, hi, n)
    lut = np.zeros((n, 4), dtype=np.float32)

    cp = sorted(color_points, key=lambda p: p[0])
    cx = np.array([p[0] for p in cp], dtype=np.float64)
    for ch in range(3):
        cy = np.array([p[1 + ch] for p in cp], dtype=np.float64)
        lut[:, ch] = np.interp(xs, cx, cy).astype(np.float32)

    op = sorted(opacity_points, key=lambda p: p[0])
    ox = np.array([p[0] for p in op], dtype=np.float64)
    oy = np.array([p[1] for p in op], dtype=np.float64)
    lut[:, 3] = np.interp(xs, ox, oy).astype(np.float32)
    return lut


def grayscale_ramp(scalar_range, max_opacity=0.9):
    """A simple black->white / transparent->opaque preset over the range."""
    lo, hi = float(scalar_range[0]), float(scalar_range[1])
    color = [(lo, 0.0, 0.0, 0.0), (hi, 1.0, 1.0, 1.0)]
    opacity = [(lo, 0.0), (hi, float(max_opacity))]
    return color, opacity


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class HeadlessVolumeRenderer:
    """Offscreen single-volume ray-march renderer returning numpy RGBA frames."""

    def __init__(self, width, height, *, fov=30.0, background=(0.0, 0.0, 0.0, 1.0),
                 power_preference="high-performance"):
        self.width = int(width)
        self.height = int(height)
        self.fov = float(fov)

        # Offscreen target -- rendercanvas.offscreen needs no Qt / no display.
        from rendercanvas.offscreen import RenderCanvas as OffscreenCanvas
        self._canvas = OffscreenCanvas(size=(self.width, self.height), pixel_ratio=1)
        self._renderer = pygfx.renderers.WgpuRenderer(self._canvas)

        self._material = SlicerVolumeMaterial(background=background)
        self._wobject = SlicerVolumeRenderer(self._material)
        self._scene = pygfx.Scene()
        self._scene.add(self._wobject)

        self._camera = pygfx.PerspectiveCamera(self.fov, self.width / self.height)

        # Camera orbit state (VTK-like: azimuth/elevation about a fixed center).
        self._center = np.zeros(3, dtype=np.float64)
        self._radius = 100.0            # bounding-sphere radius of the volume
        self._azimuth = 0.0             # degrees
        self._elevation = 0.0           # degrees
        self._distance = 400.0          # eye-to-center distance
        self._have_volume = False
        # Invalidation epoch: bumped by any structural change (volume, transfer function,
        # camera reset, framebuffer resize, sample params). A render loop watches it to know the
        # image is stale and to re-arm the adaptive budget / "settled" state. First-class so the
        # LOCAL (browser) render path can key off the same signal.
        self.epoch = 0
        # Adaptive geometry (v0): render at a dense reduced resolution during motion so every GPU
        # thread does real work (no warp divergence -> true ~1/f**2 speedup), then reconstruct to
        # full res (here: ship-small + client upsample). Cache one offscreen target per reduced
        # size so we don't reallocate on the hot path. last_render_size is the native size of the
        # most recent frame (< full size while moving) so the caller/client can scale it up.
        self._lod_targets = {}
        self.last_render_size = (self.width, self.height)
        self._prog = None                 # active progressive-refinement sequence (v1)

    def _invalidate(self):
        self.epoch += 1

    # -- volume ------------------------------------------------------------

    def set_volume(self, array, spacing=(1.0, 1.0, 1.0), *, scalar_range=None,
                   opacity_unit_distance=None, sample_step=None):
        """Upload a numpy volume as a 3D texture and set world bounds.

        array:   3D array indexed [K, J, I] (slices, rows, cols) -- the layout
                 numpy uses for medical volumes. Cast to float32.
        spacing: physical voxel size (sI, sJ, sK) along the I, J, K axes.
        scalar_range: (lo, hi) for the transfer-function domain / clim; if
                 None, uses the data min/max.
        opacity_unit_distance: physical distance (world units) over which a
                 voxel's transfer-function alpha accumulates to full. Defaults
                 to the min voxel spacing, but that makes opacity scale with
                 resolution -- pass a FIXED value when rendering several levels
                 of the same data (a pyramid) so they look consistent.
        sample_step: ray-march step (world units); defaults to min spacing.
        """
        arr = np.ascontiguousarray(array, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"volume must be 3D [K,J,I], got shape {arr.shape}")
        nk, nj, ni = arr.shape
        si, sj, sk = (float(spacing[0]), float(spacing[1]), float(spacing[2]))

        size_i, size_j, size_k = ni * si, nj * sj, nk * sk  # physical extents

        # World is axis-aligned and centered at the origin: voxel center (i,j,k)
        # -> world ( -size/2 + (idx+0.5)*spacing ) on each axis. Then
        #   tex_axis = world_axis / size_axis + 0.5   (a diagonal affine).
        # wgpu texture coord .x/.y/.z index the I/J/K axes respectively.
        p2t = np.eye(4, dtype=np.float32)
        p2t[0, 0] = 1.0 / size_i
        p2t[1, 1] = 1.0 / size_j
        p2t[2, 2] = 1.0 / size_k
        p2t[0, 3] = p2t[1, 3] = p2t[2, 3] = 0.5

        if scalar_range is None:
            scalar_range = (float(arr.min()), float(arr.max()))
        lo, hi = float(scalar_range[0]), float(scalar_range[1])
        if hi <= lo:
            hi = lo + 1.0

        self._material.volume_texture = pygfx.Texture(arr, dim=3)
        self._material.clim = (lo, hi)
        self._material.bounds_min = (-size_i / 2, -size_j / 2, -size_k / 2)
        self._material.bounds_max = (size_i / 2, size_j / 2, size_k / 2)
        self._material.patient_to_texture = p2t
        min_spacing = min(si, sj, sk)
        self._material.sample_step = min_spacing if sample_step is None else float(sample_step)
        self._material.opacity_unit_distance = (
            min_spacing if opacity_unit_distance is None else float(opacity_unit_distance))

        # The WGSL always references t_grad_lut (its use is guarded at runtime by
        # gradient_opacity_enabled), so the binding must exist to compile. Bind a
        # neutral all-ones LUT and leave gradient modulation disabled by default.
        if self._material.grad_lut_texture is None:
            ones = np.ones((256, 1), dtype=np.float32)
            self._material.grad_lut_texture = pygfx.Texture(ones, dim=1)
            self._material.gradient_range = (0.0, 1.0)
        self._material.gradient_opacity_enabled = 0.0

        # Camera framing state.
        self._center[:] = 0.0
        self._radius = 0.5 * math.sqrt(size_i ** 2 + size_j ** 2 + size_k ** 2)
        self._have_volume = True

        # A default grayscale TF so a bare set_volume() already renders.
        color, opacity = grayscale_ramp((lo, hi))
        self.set_transfer_function(color, opacity, (lo, hi))
        return self

    # -- transfer function -------------------------------------------------

    def set_transfer_function(self, color_points, opacity_points, scalar_range,
                              *, n=256):
        """Set the color/opacity transfer function from control points."""
        lo, hi = float(scalar_range[0]), float(scalar_range[1])
        if hi <= lo:
            hi = lo + 1.0
        lut = build_color_opacity_lut(color_points, opacity_points, (lo, hi), n=n)
        self._material.lut_texture = pygfx.Texture(lut, dim=1)
        self._material.clim = (lo, hi)
        self._invalidate()
        return self

    # -- camera ------------------------------------------------------------

    def set_sample_step(self, step):
        """Override the ray-march step (world units) live — e.g. a coarse step during camera
        motion for speed, then the fine (voxel) step when the view settles. Opacity is
        step-invariant (opacity/sample scales with step/opacity_unit_distance), so brightness
        stays constant; only detail/cost change."""
        self._material.sample_step = float(step)   # render-loop-internal; not an epoch invalidation

    def set_sample_budget(self, budget, frame_seed=None):
        """Adaptive selective casting: cast only `budget` (0..1) fraction of pixels this frame
        (a stochastic subset); the rest are holes for reconstruction/accumulation. frame_seed
        rotates the sampled set across frames."""
        self._material.sample_budget = float(max(0.02, min(1.0, budget)))
        if frame_seed is not None:
            self._material.frame_seed = float(frame_seed)   # render-loop-internal; not an epoch invalidation

    def frame_volume(self, margin=1.4):
        """Auto-fit: place the camera so the volume fills the view."""
        half_fov = math.radians(self.fov) * 0.5
        self._distance = (self._radius / max(math.sin(half_fov), 1e-3)) * margin
        self._invalidate()
        return self

    def set_camera(self, azimuth_deg=None, elevation_deg=None, distance=None):
        if azimuth_deg is not None:
            self._azimuth = float(azimuth_deg)
        if elevation_deg is not None:
            self._elevation = max(-89.0, min(89.0, float(elevation_deg)))
        if distance is not None:
            self._distance = max(1e-3, float(distance))
        self._invalidate()
        return self

    def orbit(self, d_azimuth_deg, d_elevation_deg):
        """VTK-style Azimuth/Elevation: rotate the eye about the volume center. (Camera motion is
        tracked via the input-pending path, not the epoch, so this does NOT bump epoch.)"""
        self._azimuth += float(d_azimuth_deg)
        self._elevation = max(-89.0, min(89.0, self._elevation + float(d_elevation_deg)))
        return self

    def dolly(self, factor):
        """VTK-style Dolly: factor > 1 moves the eye toward the center."""
        f = float(factor)
        if f > 1e-6:
            self._distance = max(1e-3, self._distance / f)
        return self

    def _place_camera(self):
        az = math.radians(self._azimuth)
        el = math.radians(self._elevation)
        # Orbit in world space; world_y is up.
        dir_ = np.array([
            math.cos(el) * math.sin(az),
            math.sin(el),
            math.cos(el) * math.cos(az),
        ], dtype=np.float64)
        eye = self._center + dir_ * self._distance
        near = max(self._distance - self._radius * 2.0, self._distance * 1e-3)
        far = self._distance + self._radius * 2.0
        self._camera.depth_range = (near, far)
        self._camera.local.position = tuple(eye)
        self._camera.look_at(tuple(self._center))

    # -- render ------------------------------------------------------------

    def render(self, scale=1.0):
        """Render the current camera view; return an (rh, rw, 4) uint8 RGBA array.

        scale (0,1] is the LINEAR resolution fraction, snapped to an integer downscale factor
        1/f (f in 1..4). scale=1.0 (default) renders at full size — unchanged behaviour. A
        reduced scale renders a *dense* smaller frame (no per-pixel skipping, so no GPU warp
        divergence): render cost drops ~1/f**2. The returned frame is the native reduced size
        (see last_render_size); the caller upsamples to full (or ships it small for the client
        to scale, which MJPEG allows). Use scale<1 while the view is moving, scale=1 when it
        settles for a crisp hero frame."""
        if not self._have_volume:
            raise RuntimeError("set_volume() must be called before render()")
        self._place_camera()
        rw, rh = self._scaled_size(scale)
        canvas, renderer = self._target_for(rw, rh)
        canvas.request_draw(lambda: renderer.render(self._scene, self._camera))
        # Offscreen draw() returns the rendered frame as an (rh, rw, 4) uint8 view.
        frame = np.asarray(canvas.draw())
        self.last_render_size = (rw, rh)
        return frame

    def _scaled_size(self, scale):
        """Full size for scale>=~1, else full/f (f=2..4) picked by rounding 1/scale. Integer
        factors keep the reduced-target cache tiny and the aspect ratio matched to full."""
        return _scaled_size(self.width, self.height, scale)

    def begin_progressive(self, factor):
        """Start converging to native full-res over factor**2 interleaved passes (v1). Call after
        the view settles; then call progressive_step() until it reports converged. Returns the
        total pass count so a caller can drive a convergence indicator."""
        f = max(2, min(4, int(round(factor))))
        self._prog = {"f": f, "order": _prog_order(f), "i": 0, "hist": None}
        return f * f

    def progressive_step(self):
        """Render+scatter the next interleaved pass; return (full_res_frame, converged). The frame
        sharpens each call and, once converged, equals a direct render() pixel-for-pixel."""
        if self._prog is None:
            self.begin_progressive(2)

        def set_dither(s, ox, oy):
            m = self._material
            m.dither_scale = s; m.dither_ox = ox; m.dither_oy = oy
        frame, converged = _progressive_step(
            self._place_camera, self._camera, self._scene, (self.width, self.height),
            self._lod_targets, (self._canvas, self._renderer), self._prog, set_dither)
        self.last_render_size = (self.width, self.height)
        return frame, converged

    def _target_for(self, w, h):
        """(canvas, renderer) for size (w,h). The full-size target is the primary canvas; reduced
        sizes get their own cached offscreen canvas+renderer (LRU, few live) rendering the SAME
        shared scene/camera — so no per-frame allocation while moving."""
        if (w, h) == (self.width, self.height):
            return self._canvas, self._renderer
        return _target_for(self._lod_targets, w, h)

    def resize(self, width, height, max_dim=4096):
        """Resize the offscreen framebuffer (and camera aspect). Camera pose is preserved.
        If either axis exceeds max_dim the size is scaled down PROPORTIONALLY (aspect kept)."""
        w, h = _clamp_size(width, height, max_dim)
        if w == self.width and h == self.height:
            return self
        self.width, self.height = w, h
        from rendercanvas.offscreen import RenderCanvas as OffscreenCanvas
        self._canvas = OffscreenCanvas(size=(self.width, self.height), pixel_ratio=1)
        self._renderer = pygfx.renderers.WgpuRenderer(self._canvas)
        self._camera = pygfx.PerspectiveCamera(self.fov, self.width / self.height)
        self._lod_targets.clear()               # reduced targets were sized off the old full size
        self.last_render_size = (self.width, self.height)
        self._invalidate()
        return self


def _clamp_size(width, height, max_dim):
    """Clamp (width,height) so neither axis exceeds max_dim, preserving aspect; floor 16."""
    w = max(16, int(width)); h = max(16, int(height))
    m = max(w, h)
    if m > max_dim:
        sc = max_dim / m
        w = max(16, int(round(w * sc))); h = max(16, int(round(h * sc)))
    return w, h


def _scaled_size(width, height, scale):
    """Adaptive-geometry reduced size: full for scale>=~1, else full/f with f = round(1/scale)
    clamped to 2..4. Integer factors keep the reduced-target cache small and match the aspect
    ratio of the full frame to within one pixel."""
    s = float(scale)
    f = int(round(1.0 / max(s, 1e-3)))
    if f <= 1:                        # scale ~>0.71 -> not worth reducing; render full
        return int(width), int(height)
    f = min(4, f)
    rw = max(16, (int(width) + f - 1) // f)
    rh = max(16, (int(height) + f - 1) // f)
    return rw, rh


def _target_for(cache, w, h, max_live=3):
    """Get/create a cached (offscreen canvas, renderer) for a reduced render size. Bounds the
    cache to max_live entries (drops the oldest) so long-lived servers don't accumulate GPU
    targets as windows resize through many sizes."""
    key = (w, h)
    t = cache.get(key)
    if t is None:
        from rendercanvas.offscreen import RenderCanvas as OffscreenCanvas
        c = OffscreenCanvas(size=(w, h), pixel_ratio=1)
        r = pygfx.renderers.WgpuRenderer(c)
        if len(cache) >= max_live:
            cache.pop(next(iter(cache)))
        cache[key] = t = (c, r)
    return t


# ---------------------------------------------------------------------------
# Progressive refinement (adaptive geometry v1): interleaved sub-lattice accumulation
# ---------------------------------------------------------------------------
# On settle, converge to native full-res over f**2 cheap reduced-res passes. Pass i renders at
# 1/f scale with the camera's view_offset jittered so its pixels cast the SAME rays as full-res
# pixels (f*k+ox, f*l+oy); those exact samples are scattered into a persistent full-res history
# buffer. After all f**2 offsets the buffer equals a direct full-res render PIXEL-FOR-PIXEL (same
# rays) — so "converged" is unambiguous and lossless, not an interpolated guess. Offsets are
# emitted in a dispersed (ordered-dither) order so the image sharpens uniformly, not as a sweep;
# order[0] is (0,0) so the first pass can nearest-upsample to a complete soft frame to start from.

_PROG_ORDER = {
    2: [(0, 0), (1, 1), (1, 0), (0, 1)],
    3: [(0, 0), (2, 2), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (0, 1), (1, 0)],
    4: [(0, 0), (2, 2), (2, 0), (0, 2), (1, 1), (3, 3), (3, 1), (1, 3),
        (1, 0), (3, 2), (3, 0), (1, 2), (0, 1), (2, 3), (2, 1), (0, 3)],
}


def _prog_order(f):
    return _PROG_ORDER.get(f, [(ox, oy) for oy in range(f) for ox in range(f)])


def _scatter(hist, reduced, ox, oy, f):
    """Write the reduced frame's exact samples into the (ox,oy) stride-f sub-lattice of hist."""
    sub = hist[oy::f, ox::f]
    h2 = min(sub.shape[0], reduced.shape[0])
    w2 = min(sub.shape[1], reduced.shape[1])
    sub[:h2, :w2] = reduced[:h2, :w2]


def _progressive_step(place_camera, camera, scene, full_wh, lod_targets, full_target, prog,
                      set_dither=None):
    """One interleaved refinement pass, shared by both renderers. Renders the next dispersed
    offset at 1/f via a jittered view_offset and scatters it into prog['hist']. Returns
    (hist, converged). full_target is (canvas, renderer) for the native size. set_dither(f,ox,oy)
    maps the shader's anti-wood-grain seed to full-frame pixel coords for this pass — without it,
    all f*f pixels of a block share one ray-start dither offset, which reads as fxf blockiness on
    detailed data (the seed must vary per FULL pixel, like a native render)."""
    W, H = full_wh
    f = prog["f"]
    order = prog["order"]
    if prog["i"] >= len(order):
        # Already converged. Return the finished buffer idempotently so extra calls (e.g. a
        # renderer shared across connections, or a caller that over-steps) can't IndexError.
        return prog["hist"], True
    ox, oy = order[prog["i"]]
    place_camera()
    rw, rh = _scaled_size(W, H, 1.0 / f)
    # view_offset: width=full so there is NO zoom, only a sub-pixel shift that re-aims the reduced
    # grid onto full-res pixels (f*k+ox, f*l+oy). Sign convention verified EMPIRICALLY (test_lattice):
    # offset v maps the reduced grid onto lattice (f-1)/2 + v on both axes, so lattice o needs
    # v = o - (f-1)/2. (The mirrored sign scrambles every fxf block -> visible box mosaic.)
    camera.set_view_offset(W, H, ox - (f - 1) / 2.0, oy - (f - 1) / 2.0, W, H)
    canvas, renderer = full_target if (rw, rh) == (W, H) else _target_for(lod_targets, rw, rh)
    if set_dither is not None:
        set_dither(float(f), float(ox), float(oy))
    try:
        canvas.request_draw(lambda: renderer.render(scene, camera))
        reduced = np.asarray(canvas.draw())
    finally:
        if set_dither is not None:
            set_dither(1.0, 0.0, 0.0)          # identity for any subsequent normal render
        camera.clear_view_offset()
    if prog["hist"] is None:
        # First pass (offset (0,0)): nearest-upsample to a complete soft frame to refine from.
        prog["hist"] = np.repeat(np.repeat(reduced, f, 0), f, 1)[:H, :W].copy()
    _scatter(prog["hist"], reduced, ox, oy, f)
    prog["i"] += 1
    return prog["hist"], prog["i"] >= len(prog["order"])


# ---------------------------------------------------------------------------
# Multi-volume: SceneRenderer + N ImageFields, composited per-sample
# ---------------------------------------------------------------------------

class HeadlessSceneRenderer:
    """Offscreen MULTI-volume renderer. Wraps slicer_wgpu.scene_renderer.SceneRenderer, which
    composites several ImageFields (each its own volume + transfer function) in one ray-march —
    the production SlicerWGPU path. Same offscreen canvas + orbit/dolly camera as
    HeadlessVolumeRenderer, read back to an (H, W, 4) uint8 array.

    Usage:
        r = HeadlessSceneRenderer(W, H)
        r.add_volume(arr1, spacing1, color_points=c1, opacity_points=o1, scalar_range=(lo,hi), ...)
        r.add_volume(arr2, spacing2, color_points=c2, opacity_points=o2, scalar_range=(lo,hi), ...)
        r.build()                       # each volume centered at world origin -> they overlap
        rgba = r.render()               # (H, W, 4) uint8
        r.orbit(d_az, d_el); r.dolly(f); r.set_sample_step(step)
    """

    def __init__(self, width, height, *, fov=30.0, background=(0.0, 0.0, 0.0, 1.0),
                 light_direction=(0.0, 0.0, 0.0)):   # zero-length -> per-pixel headlight (VTK default)
        from rendercanvas.offscreen import RenderCanvas as OffscreenCanvas
        self.width = int(width)
        self.height = int(height)
        self.fov = float(fov)
        self._canvas = OffscreenCanvas(size=(self.width, self.height), pixel_ratio=1)
        self._renderer = pygfx.renderers.WgpuRenderer(self._canvas)
        self._scene = pygfx.Scene()
        self._camera = pygfx.PerspectiveCamera(self.fov, self.width / self.height)
        self._background = background
        self._light = light_direction
        self._fields = []
        self._so = None
        self._center = np.zeros(3, dtype=np.float64)
        self._radius = 100.0
        self._azimuth = 0.0
        self._elevation = 0.0
        self._distance = 400.0
        self.epoch = 0            # invalidation epoch (see HeadlessVolumeRenderer.epoch)
        # Adaptive geometry (v0): see HeadlessVolumeRenderer — dense reduced-res targets, cached.
        self._lod_targets = {}
        self.last_render_size = (self.width, self.height)
        self._prog = None                 # active progressive-refinement sequence (v1)

    def _invalidate(self):
        self.epoch += 1

    def add_volume(self, array, spacing=(1.0, 1.0, 1.0), *, color_points, opacity_points,
                   scalar_range, opacity_unit_distance=None, sample_step=None,
                   k_ambient=0.1, k_diffuse=0.9, k_specular=0.2, shininess=10.0,
                   center_at_origin=True):
        """Add one volume + its transfer function as an ImageField. By default the volume is
        centered at the world origin (so multiple volumes overlap and composite, matching the
        SceneRendering test_MultiVolume demo)."""
        from slicer_wgpu.fields import ImageField
        arr = np.ascontiguousarray(array, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"volume must be 3D [K,J,I], got {arr.shape}")
        nk, nj, ni = arr.shape
        si, sj, sk = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        size_i, size_j, size_k = ni * si, nj * sj, nk * sk

        # world (centered at origin) -> texture[0,1]: tex_a = world_a/size_a + 0.5  (diagonal affine)
        p2t = np.eye(4, dtype=np.float32)
        p2t[0, 0] = 1.0 / size_i
        p2t[1, 1] = 1.0 / size_j
        p2t[2, 2] = 1.0 / size_k
        p2t[0, 3] = p2t[1, 3] = p2t[2, 3] = 0.5

        lo, hi = float(scalar_range[0]), float(scalar_range[1])
        if hi <= lo:
            hi = lo + 1.0
        lut = build_color_opacity_lut(color_points, opacity_points, (lo, hi))

        min_sp = min(si, sj, sk)
        oud = min_sp if opacity_unit_distance is None else float(opacity_unit_distance)
        step = min_sp if sample_step is None else float(sample_step)
        off = (0.0, 0.0, 0.0) if center_at_origin else (size_i / 2, size_j / 2, size_k / 2)
        # The generated WGSL always references t_grad_lut<slot> (guarded at runtime by
        # gradient_opacity_enabled), so bind a neutral all-ones grad LUT even when disabled
        # (also matches these CT presets, whose gradientOpacity is a constant 1).
        grad_lut = np.ones((256, 1), dtype=np.float32)
        field = ImageField(
            volume_array=arr, lut_array=lut, grad_lut_array=grad_lut, clim=(lo, hi),
            gradient_range=(0.0, 1.0),
            bounds_min=(-size_i / 2 + off[0], -size_j / 2 + off[1], -size_k / 2 + off[2]),
            bounds_max=(size_i / 2 + off[0], size_j / 2 + off[1], size_k / 2 + off[2]),
            patient_to_texture=p2t, sample_step_mm=step, opacity_unit_distance=oud,
            gradient_opacity_enabled=False,
            k_ambient=k_ambient, k_diffuse=k_diffuse, k_specular=k_specular, shininess=shininess)
        self._fields.append(field)
        return self

    def build(self, margin=1.4):
        from slicer_wgpu.scene_renderer import SceneRenderer
        self._so = SceneRenderer.build_for_fields(self._fields)
        self._so.material.background = self._background
        self._so.material.light_direction = self._light
        self._scene.add(self._so)
        self._so.recompute_scene_bounds()
        boxes = [f.aabb() for f in self._fields if f.aabb() is not None]
        lo = np.min(np.stack([b[0] for b in boxes]), axis=0)
        hi = np.max(np.stack([b[1] for b in boxes]), axis=0)
        self._center = (lo + hi) / 2.0
        self._radius = 0.5 * float(np.linalg.norm(hi - lo))
        self.frame_volume(margin)
        self._invalidate()
        return self

    def set_sample_step(self, step):
        """Live coarse/fine ray-march step for all fields (motion-adaptive LOD)."""
        if self._so is not None:
            self._so.material.sample_step = float(step)   # render-loop-internal; not an epoch invalidation

    # -- camera (same semantics as HeadlessVolumeRenderer) -----------------

    def frame_volume(self, margin=1.4):
        half_fov = math.radians(self.fov) * 0.5
        self._distance = (self._radius / max(math.sin(half_fov), 1e-3)) * margin
        self._invalidate()
        return self

    def set_camera(self, azimuth_deg=None, elevation_deg=None, distance=None):
        if azimuth_deg is not None:
            self._azimuth = float(azimuth_deg)
        if elevation_deg is not None:
            self._elevation = max(-89.0, min(89.0, float(elevation_deg)))
        if distance is not None:
            self._distance = max(1e-3, float(distance))
        self._invalidate()
        return self

    def orbit(self, d_azimuth_deg, d_elevation_deg):
        self._azimuth += float(d_azimuth_deg)
        self._elevation = max(-89.0, min(89.0, self._elevation + float(d_elevation_deg)))
        return self

    def dolly(self, factor):
        f = float(factor)
        if f > 1e-6:
            self._distance = max(1e-3, self._distance / f)
        return self

    def _place_camera(self):
        az = math.radians(self._azimuth); el = math.radians(self._elevation)
        dir_ = np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)],
                        dtype=np.float64)
        eye = self._center + dir_ * self._distance
        self._camera.depth_range = (max(self._distance - self._radius * 2.0, self._distance * 1e-3),
                                    self._distance + self._radius * 2.0)
        self._camera.local.position = tuple(eye)
        self._camera.look_at(tuple(self._center))

    def render(self, scale=1.0):
        """Render the multi-volume scene; return an (rh, rw, 4) uint8 RGBA array. See
        HeadlessVolumeRenderer.render — scale (0,1] renders a dense reduced frame (1/f, f=1..4)
        for ~1/f**2 cost during motion; scale=1.0 (default) is full-size/unchanged."""
        if self._so is None:
            raise RuntimeError("build() must be called before render()")
        self._place_camera()
        rw, rh = _scaled_size(self.width, self.height, scale)
        if (rw, rh) == (self.width, self.height):
            canvas, renderer = self._canvas, self._renderer
        else:
            canvas, renderer = _target_for(self._lod_targets, rw, rh)
        canvas.request_draw(lambda: renderer.render(self._scene, self._camera))
        self.last_render_size = (rw, rh)
        return np.asarray(canvas.draw())

    def begin_progressive(self, factor):
        """Start converging to native full-res over factor**2 interleaved passes (v1). See
        HeadlessVolumeRenderer.begin_progressive."""
        f = max(2, min(4, int(round(factor))))
        self._prog = {"f": f, "order": _prog_order(f), "i": 0, "hist": None}
        return f * f

    def progressive_step(self):
        """Render+scatter the next interleaved pass; return (full_res_frame, converged). Converged
        frame equals a direct render() pixel-for-pixel (same rays)."""
        if self._so is None:
            raise RuntimeError("build() must be called before progressive_step()")
        if self._prog is None:
            self.begin_progressive(2)
        frame, converged = _progressive_step(
            self._place_camera, self._camera, self._scene, (self.width, self.height),
            self._lod_targets, (self._canvas, self._renderer), self._prog,
            self._so.material.set_dither_mapping)
        self.last_render_size = (self.width, self.height)
        return frame, converged

    def resize(self, width, height, max_dim=4096):
        """Resize the offscreen framebuffer to (width, height); camera pose preserved. Scene
        (worldobject + fields) unchanged — only canvas/renderer/camera aspect are rebuilt.
        Scaled down proportionally if an axis exceeds max_dim."""
        w, h = _clamp_size(width, height, max_dim)
        if w == self.width and h == self.height:
            return self
        self.width, self.height = w, h
        from rendercanvas.offscreen import RenderCanvas as OffscreenCanvas
        self._canvas = OffscreenCanvas(size=(self.width, self.height), pixel_ratio=1)
        self._renderer = pygfx.renderers.WgpuRenderer(self._canvas)
        self._camera = pygfx.PerspectiveCamera(self.fov, self.width / self.height)
        self._lod_targets.clear()               # reduced targets were sized off the old full size
        self.last_render_size = (self.width, self.height)
        self._invalidate()
        return self
