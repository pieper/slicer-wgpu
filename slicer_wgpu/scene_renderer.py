"""Scene Renderer: a per-field-compositing ray tracer.

Owns a list of Fields and generates a WGSL shader that, at each ray
sample point, calls each Field's `sample_field_<kind><slot>(wp, ray_dir,
ray_origin) -> FieldSample`, combines the per-field contributions, and
front-to-back composites the result.

WGSL is generated when the renderer is built (`build_for_fields`). Per-
frame uniform updates don't require regeneration; only structural changes
(adding/removing a Field, or a Field changing kind) do. The host owns
the lifecycle and decides when to rebuild.

Picking is dispatched in Python: `pick_at(ndc_x, ndc_y)` walks the
field list, asks each picking-capable Field for a hit, and returns the
nearest one. Drag updates flow back into the same Field's `drag_update`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pygfx
import wgpu

from pygfx.renderers.wgpu import (
    Binding,
    register_wgpu_render_function,
    GfxSampler,
    GfxTextureView,
)
from pygfx.renderers.wgpu.shader.base import BaseShader

from .fields import Field


VERTEX_AND_FRAGMENT = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
FRAGMENT_ONLY = wgpu.ShaderStage.FRAGMENT


# ----------------------------------------------------------------------
# Shader template -- everything outside the per-field fragments
# ----------------------------------------------------------------------

_SHADER_TEMPLATE = """
{$ include 'pygfx.std.wgsl' $}

struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    let x = select(-1.0, 3.0, in.vertex_index == 1u);
    let y = select(-1.0, 3.0, in.vertex_index == 2u);
    var varyings: Varyings;
    varyings.position = vec4<f32>(x, y, 0.0, 1.0);
    return varyings;
}

// ---- Field sample protocol ----
struct FieldSample {
    color_srgb: vec3<f32>,
    opacity: f32,
    hit: bool,
};

// ==== Shadow sampling (stub or real, depending on shader variant) ====
__SAMPLE_SHADOW_FN__
// ==== End shadow sampling ====

// ==== Per-field WGSL (generated) ====
__FIELD_FUNCTIONS__
// ==== End per-field WGSL ====

fn ray_aabb(ray_origin: vec3<f32>, ray_dir: vec3<f32>,
            bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let inv_dir = vec3<f32>(1.0) / ray_dir;
    let tb = (bmin - ray_origin) * inv_dir;
    let tt = (bmax - ray_origin) * inv_dir;
    let tmn = min(tt, tb);
    let tmx = max(tt, tb);
    let t_near = max(max(tmn.x, tmn.y), tmn.z);
    let t_far  = min(min(tmx.x, tmx.y), tmx.z);
    return vec2<f32>(t_near, t_far);
}

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    var out: FragmentOutput;

    let size = u_stdinfo.physical_size;
    let ndc_x = (varyings.position.x / size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (varyings.position.y / size.y) * 2.0;

    // WebGPU NDC z is [0, 1] with z=0 at the near plane. pygfx's
    // ndc_to_world_pos applies inv(projection)*inv(view) directly so
    // it respects whatever projection matrix the camera wrote. The
    // `sign(cam_transform_diag_product)` hack (copied from pygfx's
    // volume_ray.wgsl) only holds for axis-aligned cameras -- for
    // off-axis rotations the view-matrix diagonals are small and
    // mixed-sign, so the product flips sign spuriously and the ray
    // inverts, giving back-to-front compositing (spine appears in
    // front of ribs).
    let world_near = ndc_to_world_pos(vec4<f32>(ndc_x, ndc_y, 0.0, 1.0));
    let world_far  = ndc_to_world_pos(vec4<f32>(ndc_x, ndc_y, 1.0, 1.0));
    let ray_origin = world_near;
    let ray_dir    = normalize(world_far - world_near);

    // Combined scene AABB across all fields (uniform).
    let bmin = u_material.scene_bounds_min.xyz;
    let bmax = u_material.scene_bounds_max.xyz;
    let trange = ray_aabb(ray_origin, ray_dir, bmin, bmax);
    var t_near = trange.x;
    var t_far  = trange.y;

    if (t_far <= t_near || t_far <= 0.0) {
        let bg = srgb2physical(u_material.background.rgb);
        out.color = vec4<f32>(bg, 1.0);
        return out;
    }

    let step = max(u_material.sample_step, 1e-3);
    t_near = max(t_near + step, 0.0);
    t_far  = t_far - step;
    if (t_far <= t_near) {
        let bg = srgb2physical(u_material.background.rgb);
        out.color = vec4<f32>(bg, 1.0);
        return out;
    }

    // Per-pixel jitter to break wood-grain
    let seed = fract(sin(dot(vec3<f32>(varyings.position.xy, 0.0),
                              vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
    var t = t_near + seed * step;

    var integrated = vec4<f32>(0.0);
    var safety: i32 = 0;
    let max_steps: i32 = 4096;

    loop {
        if (t >= t_far)          { break; }
        if (safety >= max_steps) { break; }
        if (integrated.a >= 0.99){ break; }

        let wp = ray_origin + ray_dir * t;

        // Combine all fields at this sample point. Per-sample compositing
        // is STEP-style: each field contributes its own (color, opacity);
        // weighted by opacity, the contributions are summed into a single
        // (lit, op_total) for the front-to-back step.
        var lit_sum = vec3<f32>(0.0);
        var op_sum  = 0.0;

__FIELD_DISPATCH__

        if (op_sum > 0.0) {
            let lit = lit_sum / op_sum;
            // Alpha is the max of the per-field opacities, NOT the sum,
            // so two opaque fields at the same point don't double-count.
            let op = clamp(op_sum, 0.0, 1.0);
            let lit_lin = srgb2physical(lit);
            integrated.r = integrated.r + (1.0 - integrated.a) * op * lit_lin.r;
            integrated.g = integrated.g + (1.0 - integrated.a) * op * lit_lin.g;
            integrated.b = integrated.b + (1.0 - integrated.a) * op * lit_lin.b;
            integrated.a = integrated.a + (1.0 - integrated.a) * op;
        }

        t = t + step;
        safety = safety + 1;
    }

    let bg = srgb2physical(u_material.background.rgb);
    let final_lin = mix(bg, integrated.rgb, integrated.a);
    out.color = vec4<f32>(final_lin, 1.0);
    return out;
}
"""


def _build_field_dispatch_block(fields: list[Field], slot_indices: list[int]) -> str:
    """For each Field, emit a WGSL block that calls sample_field_<kind><slot>
    and accumulates into lit_sum / op_sum. Indented to match the template.
    """
    lines = []
    for field, slot in zip(fields, slot_indices):
        fn = f"sample_field_{field.field_kind}{slot}"
        lines.append(f"        {{")
        lines.append(f"            let s = {fn}(wp, ray_dir, ray_origin);")
        lines.append(f"            if (s.hit) {{")
        lines.append(f"                lit_sum = lit_sum + s.color_srgb * s.opacity;")
        lines.append(f"                op_sum = op_sum + s.opacity;")
        lines.append(f"            }}")
        lines.append(f"        }}")
    if not lines:
        # No fields: still need a no-op so the loop body compiles.
        lines.append("        // (no fields)")
    return "\n".join(lines)


def _build_field_functions_block(fields: list[Field], slot_indices: list[int]) -> str:
    chunks = []
    for field, slot in zip(fields, slot_indices):
        chunks.append(field.sampling_wgsl(slot))
        chunks.append(field.tf_wgsl(slot))
    return "\n".join(chunks)


# ----------------------------------------------------------------------
# Material + WorldObject
# ----------------------------------------------------------------------

class SceneMaterial(pygfx.Material):
    """Scene-renderer material. Concrete subclass per field configuration
    (since the uniform layout depends on which fields are present); see
    `make_material_class`. Construct via `SceneRenderer.build_for_fields`.
    """

    # Filled in by make_material_class:
    uniform_type = dict(pygfx.Material.uniform_type)
    _slicer_wgpu_field_kinds: tuple = ()      # ordered (kind, slot) pairs for shader codegen

    def __init__(self, **kwargs):
        super().__init__()
        # Our fragment shader outputs the fully-composited sRGB colour with
        # alpha=1. We want pygfx to blit it directly to the target without
        # trying to alpha-blend against any preceding geometry. "solid"
        # forces an opaque one-write pass; the default "auto" mode
        # silently drops the draw for geometryless WorldObjects.
        self.alpha_config = {"mode": "solid", "method": "opaque",
                             "premultiply_alpha": False}
        # Sensible defaults for the shared scene uniforms.
        self.background = (0.05, 0.05, 0.08, 1.0)
        self.scene_bounds_min = (-100.0, -100.0, -100.0)
        self.scene_bounds_max = (100.0, 100.0, 100.0)
        # Zero means "use per-pixel headlight" in the fragment shader.
        self.light_direction = (0.0, 0.0, 0.0, 0.0)
        self.light_intensity = 1.0
        # Fill light: unshadowed, off by default. xyz points FROM surface
        # TO the fill light source. When fill_light_intensity is 0 or the
        # direction is a zero vector, the fill contribution is skipped.
        self.fill_light_direction = (0.0, 0.0, 0.0, 0.0)
        self.fill_light_intensity = 0.0
        self.sample_step = 1.0
        for k, v in kwargs.items():
            setattr(self, k, v)

    # -------- Scene-wide uniforms --------

    def _set_vec4(self, name, v):
        arr = np.zeros(4, dtype=np.float32)
        a = np.asarray(v, dtype=np.float32).ravel()
        arr[:min(len(a), 4)] = a[:min(len(a), 4)]
        if len(a) == 3:
            arr[3] = 1.0
        self.uniform_buffer.data[name] = arr
        self.uniform_buffer.update_full()

    @property
    def background(self):
        return tuple(float(x) for x in self.uniform_buffer.data["background"])
    @background.setter
    def background(self, v): self._set_vec4("background", v)

    @property
    def scene_bounds_min(self):
        return tuple(float(x) for x in self.uniform_buffer.data["scene_bounds_min"][:3])
    @scene_bounds_min.setter
    def scene_bounds_min(self, v): self._set_vec4("scene_bounds_min", v)

    @property
    def scene_bounds_max(self):
        return tuple(float(x) for x in self.uniform_buffer.data["scene_bounds_max"][:3])
    @scene_bounds_max.setter
    def scene_bounds_max(self, v): self._set_vec4("scene_bounds_max", v)

    @property
    def sample_step(self):
        return float(self.uniform_buffer.data["sample_step"])
    @sample_step.setter
    def sample_step(self, v):
        self.uniform_buffer.data["sample_step"] = float(v)
        self.uniform_buffer.update_full()

    @property
    def light_direction(self):
        return tuple(float(x) for x in self.uniform_buffer.data["light_direction"][:3])
    @light_direction.setter
    def light_direction(self, v): self._set_vec4("light_direction", v)

    @property
    def light_intensity(self):
        return float(self.uniform_buffer.data["light_intensity"])
    @light_intensity.setter
    def light_intensity(self, v):
        self.uniform_buffer.data["light_intensity"] = float(v)
        self.uniform_buffer.update_full()

    @property
    def fill_light_direction(self):
        return tuple(float(x) for x in self.uniform_buffer.data["fill_light_direction"][:3])
    @fill_light_direction.setter
    def fill_light_direction(self, v): self._set_vec4("fill_light_direction", v)

    @property
    def fill_light_intensity(self):
        return float(self.uniform_buffer.data["fill_light_intensity"])
    @fill_light_intensity.setter
    def fill_light_intensity(self, v):
        self.uniform_buffer.data["fill_light_intensity"] = float(v)
        self.uniform_buffer.update_full()


def make_material_class(fields: list[Field], slot_indices: list[int]):
    """Dynamically build a SceneMaterial subclass with the right per-field
    uniform layout, and a SceneRenderer subclass paired with it. pygfx
    caches the compiled shader by (WorldObject class, Material class), so
    a fresh class per field configuration gives us the desired recompile.
    """
    uniform_type = dict(pygfx.Material.uniform_type)
    # Scene-wide uniforms. Do NOT add manual "__padXX" scalars to force
    # alignment here: pygfx auto-aligns by sorting fields by alignment
    # (big->small), AND strips any field whose name starts with "__"
    # from the generated WGSL struct -- so a Python-side "__pad" field
    # silently offsets every subsequent scalar relative to the shader's
    # view of the same buffer. pygfx-friendly order (vec4 first, then
    # scalars) is all that's required.
    uniform_type["background"]        = "4xf4"
    uniform_type["scene_bounds_min"]  = "4xf4"
    uniform_type["scene_bounds_max"]  = "4xf4"
    # light_direction points FROM the surface TO the light (same convention
    # as the shadow compute pass). When its length is <1e-6, the fragment
    # shader falls back to a per-pixel headlight, preserving the
    # unshaded-default look for callers that haven't opted into shadows.
    uniform_type["light_direction"]       = "4xf4"
    # Optional fill light (unshadowed, intensity-scaled). xyz is the
    # surface-to-light direction; intensity==0 disables the fill term.
    uniform_type["fill_light_direction"]  = "4xf4"
    uniform_type["light_intensity"]       = "f4"
    uniform_type["fill_light_intensity"]  = "f4"
    uniform_type["sample_step"]           = "f4"

    # Per-field uniforms
    for field, slot in zip(fields, slot_indices):
        for k, t in field.uniform_type(slot).items():
            if k in uniform_type:
                raise ValueError(f"Uniform name collision: {k!r}")
            uniform_type[k] = t

    suffix = "_" + "_".join(f"{f.field_kind}{s}" for f, s in zip(fields, slot_indices)) or "_empty"
    mat_cls = type(
        "SceneMaterial" + suffix,
        (SceneMaterial,),
        {
            "uniform_type": uniform_type,
            "_slicer_wgpu_field_kinds": tuple(
                (f.field_kind, s) for f, s in zip(fields, slot_indices)
            ),
        },
    )
    obj_cls = type(
        "SceneRenderer" + suffix,
        (SceneRenderer,),
        {},
    )
    return obj_cls, mat_cls


_SAMPLE_SHADOW_STUB = """
fn sample_shadow(wp: vec3<f32>) -> f32 {
    return 1.0;
}
"""

_SAMPLE_SHADOW_REAL = """
fn sample_shadow(wp: vec3<f32>) -> f32 {
    let bmin = u_material.scene_bounds_min.xyz;
    let bmax = u_material.scene_bounds_max.xyz;
    let extent = bmax - bmin;
    let safe_extent = vec3<f32>(
        max(extent.x, 1e-6),
        max(extent.y, 1e-6),
        max(extent.z, 1e-6),
    );
    let uvw = (wp - bmin) / safe_extent;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
        return 1.0;
    }
    return textureSampleLevel(t_shadow, s_shadow, uvw, 0.0).r;
}
"""


# pygfx's get_cached_shader_module (engine/pipeline.py) keys its cache on
# `shader.hash`, which -- when the class fullname has >=2 dots -- is
# derived from fullname(module.class) alone. Without disambiguation the
# cache would hit across module reloads and shader-template edits,
# silently re-using a stale compiled pipeline.
#
# We want cache HITS when the WGSL is unchanged (production: same field
# configuration re-built many times) and cache MISSES when the WGSL
# content changes (developer edits SHADER_TEMPLATE, a new field kind
# appears, shadows get enabled, etc.). A content hash of the generated
# WGSL achieves both: we fold an 8-hex-char MD5 of the WGSL into the
# shader class `__name__`, so identical WGSL yields identical classnames
# and pygfx's cache does the right thing automatically.
import hashlib as _hashlib


def _shader_name_for(wgsl: str) -> str:
    digest = _hashlib.md5(wgsl.encode("utf-8")).hexdigest()[:8]
    return f"_SceneShader_{digest}"


class SceneRenderer(pygfx.WorldObject):
    """Screen-space scene renderer. Owns a list of Fields; the shader is
    generated from that list when `build_for_fields` constructs the
    object. Geometry is None -- vertex shader emits a fullscreen triangle.
    """

    def __init__(self, material: SceneMaterial, **kwargs):
        super().__init__(geometry=None, **kwargs)
        self.material = material
        self._fields: list[Field] = []
        self._slot_indices: list[int] = []
        self._field_mtimes: list[int] = []
        self._shader_wgsl: str = ""

    @classmethod
    def build_for_fields(cls, fields: Iterable[Field],
                         shadow_volume=None) -> "SceneRenderer":
        """Construct a SceneRenderer + matching SceneMaterial subclass for
        the given list of fields. The shader is generated and registered.

        If ``shadow_volume`` is provided (a ``shadows.ShadowVolume`` instance),
        the fragment shader is compiled with ``t_shadow``/``s_shadow``
        bindings and a real ``sample_shadow(wp)``; otherwise
        ``sample_shadow`` returns 1.0 and no shadow texture is bound.
        """
        fields = list(fields)
        # Assign slot indices per field-kind (img0, img1, fid0, fid1, ...).
        slot_indices = []
        per_kind_count: dict = {}
        for f in fields:
            n = per_kind_count.get(f.field_kind, 0)
            slot_indices.append(n)
            per_kind_count[f.field_kind] = n + 1

        obj_cls, mat_cls = make_material_class(fields, slot_indices)

        shadowed = shadow_volume is not None
        sample_shadow_src = _SAMPLE_SHADOW_REAL if shadowed else _SAMPLE_SHADOW_STUB

        # Generate the WGSL once, store on the class so the shader's
        # get_code() can read it back.
        wgsl = (
            _SHADER_TEMPLATE
            .replace("__SAMPLE_SHADOW_FN__", sample_shadow_src)
            .replace("__FIELD_FUNCTIONS__", _build_field_functions_block(fields, slot_indices))
            .replace("__FIELD_DISPATCH__",  _build_field_dispatch_block(fields, slot_indices))
        )

        # Content-addressed shader class name: identical WGSL produces
        # identical class names (cache hit in production), and ANY
        # change to the generated WGSL produces a new class name
        # (cache miss, fresh compile). See _shader_name_for.
        _shader_cls_name = _shader_name_for(wgsl)

        # Register the shader for this class pair (idempotent: pygfx
        # caches by (obj_cls, mat_cls) and we just minted both).
        @register_wgpu_render_function(obj_cls, mat_cls)
        class _SceneShader(BaseShader):
            type = "render"
            _wgsl = wgsl
            _fields_for_bindings = fields
            _slot_indices = slot_indices

            def get_bindings(self, wobject, shared, scene=None):
                # `scene` arg was added in a pygfx point-release; older
                # 0.16.x installs still call this with (wobject, shared).
                # Defaulting to None keeps us compatible with both.
                bindings = [
                    Binding("u_stdinfo",  "buffer/uniform", shared.uniform_buffer),
                    Binding("u_wobject",  "buffer/uniform", wobject.uniform_buffer),
                    Binding("u_material", "buffer/uniform", wobject.material.uniform_buffer),
                ]
                # Per-field bindings (textures, samplers).
                for f, s in zip(wobject._fields, wobject._slot_indices):
                    bindings.extend(f.get_bindings(s))
                # Shadow bindings, when this renderer was built with shadows.
                sv = getattr(wobject, "_shadow_volume", None)
                if sv is not None:
                    bindings.extend([
                        Binding("s_shadow", "sampler/filtering",
                                GfxSampler("linear", "clamp"), FRAGMENT_ONLY),
                        Binding("t_shadow", "texture/auto",
                                GfxTextureView(sv.pygfx_texture), FRAGMENT_ONLY),
                    ])
                bindings = {i: b for i, b in enumerate(bindings)}
                self.define_bindings(0, bindings)
                return {0: bindings}

            def get_pipeline_info(self, wobject, shared):
                return {
                    "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
                    "cull_mode": wgpu.CullMode.none,
                }

            def get_render_info(self, wobject, shared):
                return {"indices": (3, 1)}

            def get_code(self):
                return self._wgsl

        _SceneShader.__name__ = _shader_cls_name
        _SceneShader.__qualname__ = _shader_cls_name

        material = mat_cls()
        renderer = obj_cls(material)
        renderer._fields = fields
        renderer._shadow_volume = shadow_volume
        renderer._slot_indices = slot_indices
        renderer._field_mtimes = [f.mtime for f in fields]
        renderer._shader_wgsl = wgsl

        # Initial uniform fill + scene bounds
        renderer.refresh_uniforms()
        renderer.recompute_scene_bounds()
        return renderer

    # -------- Live updates --------

    def fields(self) -> list[Field]:
        return list(self._fields)

    def needs_rebuild_for(self, fields: Iterable[Field]) -> bool:
        """True if the field list's structure has changed (count or kind),
        which requires a fresh SceneRenderer/SceneMaterial pair."""
        new = list(fields)
        if len(new) != len(self._fields):
            return True
        for old, nf in zip(self._fields, new):
            if old.field_kind != nf.field_kind:
                return True
        return False

    def refresh_uniforms(self) -> None:
        """Re-fill all field uniforms (cheap; happens on any field mtime
        change). Does not regenerate the shader."""
        for f, s in zip(self._fields, self._slot_indices):
            f.fill_uniforms(self.material.uniform_buffer, s)
        self.material.uniform_buffer.update_full()

    def maybe_refresh(self) -> bool:
        """If any field's mtime advanced, refresh uniforms + bounds.
        Returns True if anything changed."""
        changed = False
        for i, f in enumerate(self._fields):
            if f.mtime != self._field_mtimes[i]:
                self._field_mtimes[i] = f.mtime
                changed = True
        if changed:
            self.refresh_uniforms()
            self.recompute_scene_bounds()
        return changed

    def recompute_scene_bounds(self) -> None:
        boxes = [f.aabb() for f in self._fields]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return
        lo = np.min(np.stack([b[0] for b in boxes]), axis=0)
        hi = np.max(np.stack([b[1] for b in boxes]), axis=0)
        # Pad a little so jitter at the edges still lands inside.
        pad = (hi - lo).max() * 0.01
        self.material.scene_bounds_min = tuple(lo - pad)
        self.material.scene_bounds_max = tuple(hi + pad)
        # Sample step: smallest of any field's preferred step (or 1mm
        # default for fields that don't have one).
        steps = []
        for f in self._fields:
            if hasattr(f, "sample_step_mm"):
                steps.append(f.sample_step_mm)
        if steps:
            self.material.sample_step = min(steps)

    # -------- Picking --------

    def pick_at(self, ndc_x: float, ndc_y: float, camera, viewport_size):
        """Walk the picking-capable fields and return the nearest hit, or None."""
        # Construct the world-space ray for this NDC point. Use camera
        # matrices directly (not the WGSL pipeline).
        # cam.local.position + (look-at projection) gives us the ray.
        from pylinalg import vec_unproject  # type: ignore
        # Two NDC points along the same line through the pixel
        near_ndc = np.array([ndc_x, ndc_y, 0.0], dtype=np.float64)
        far_ndc  = np.array([ndc_x, ndc_y, 1.0], dtype=np.float64)
        # ndc -> world (perspective)
        cam_proj = np.asarray(camera.projection_matrix, dtype=np.float64)
        cam_world = np.asarray(camera.world.matrix, dtype=np.float64)
        full = cam_world @ np.linalg.inv(cam_proj)
        def ndc_to_world(p):
            v = full @ np.array([*p, 1.0], dtype=np.float64)
            return v[:3] / v[3]
        wn = ndc_to_world(near_ndc)
        wf = ndc_to_world(far_ndc)
        ray_dir = wf - wn
        ray_dir /= np.linalg.norm(ray_dir)
        ray_origin = wn

        nearest = None
        for f in self._fields:
            if not f.supports_picking:
                continue
            hit = f.pick(ray_origin, ray_dir, camera, viewport_size)
            if hit is None:
                continue
            if nearest is None or hit.t < nearest.t:
                nearest = hit
        return nearest

    def drag_continue(self, hit, ndc_x, ndc_y, camera, viewport_size) -> bool:
        """Forward a drag pointer-move to the originating field."""
        cam_proj = np.asarray(camera.projection_matrix, dtype=np.float64)
        cam_world = np.asarray(camera.world.matrix, dtype=np.float64)
        full = cam_world @ np.linalg.inv(cam_proj)
        def ndc_to_world(p):
            v = full @ np.array([*p, 1.0], dtype=np.float64)
            return v[:3] / v[3]
        wn = ndc_to_world([ndc_x, ndc_y, 0.0])
        wf = ndc_to_world([ndc_x, ndc_y, 1.0])
        ray_dir = wf - wn
        ray_dir /= np.linalg.norm(ray_dir)
        ray_origin = wn
        changed = hit.field.drag_update(hit, ray_origin, ray_dir, camera, viewport_size)
        if changed:
            self.maybe_refresh()
        return changed
