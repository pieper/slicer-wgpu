"""
slicer_wgpu.demos.single_volume -- standalone pygfx-compatible volume
renderer ported from STEP (https://github.com/pieper/step) with
RenderCL-inspired fixes.

This is the original Stage-1 single-volume implementation, kept here as a
self-contained reference / demo. Production code uses
slicer_wgpu.scene_renderer (a field-compositing ray tracer that supersedes
this module). See slicer_wgpu.fields for the per-Field abstraction.

Design:
  * Fullscreen triangle (3 verts, no geometry buffer).
  * Ray set up in WORLD space from pygfx's camera matrices.
  * Ray/AABB slab intersection against the volume's RAS bounds.
  * Ray is INSET by one sample step at both ends (RenderCL trick):
    tNear += step; tFar -= step. This keeps gradient neighbors valid
    and prevents first/last slice "edge plane" artifacts.
  * Per-sample: world -> patient -> texture coord via patient_to_texture
    matrix. Strict trivial-reject: samples outside [0,1] contribute
    nothing (belt-and-braces alongside the tNear/tFar inset).
  * VTK-style gradient-opacity LUT modulation: opacity = base_alpha *
    grad_opacity_lut[gradient_magnitude] * (step / unit_distance).
    The LUT is sampled from vtkVolumeProperty.GetGradientOpacity(),
    matching Slicer's preset behavior. Homogeneous regions get near-zero
    multiplier so the first/last slice's flat plane is suppressed even
    when it has real tissue data; surfaces (high gradient) get full
    opacity so skin/bone edges show.
  * Front-to-back alpha compositing.
  * Per-pixel headlight Phong (matches VTK default vtkLight lightType=1).
"""

from __future__ import annotations

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


VERTEX_AND_FRAGMENT = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
FRAGMENT_ONLY = wgpu.ShaderStage.FRAGMENT


# ------------------------------------------------------------------------
# WGSL shader (inline to keep module self-contained)
# ------------------------------------------------------------------------

SHADER_WGSL = """
// Slicer Volume Renderer - STEP port for pygfx
// Bindings and shared structs (u_stdinfo, u_wobject, FragmentOutput,
// ndc_to_world_pos, etc.) are auto-generated from pygfx.std.wgsl.

{$ include 'pygfx.std.wgsl' $}


// ---- Vertex: fullscreen triangle via vertex_index ----

struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    // Classic "3-vertex covers the screen" trick:
    //   idx 0 -> (-1,-1),  idx 1 -> (3,-1),  idx 2 -> (-1,3)
    let x = select(-1.0, 3.0, in.vertex_index == 1u);
    let y = select(-1.0, 3.0, in.vertex_index == 2u);
    var varyings: Varyings;
    varyings.position = vec4<f32>(x, y, 0.0, 1.0);
    return varyings;
}


// ---- Helpers ----

fn sample_lut(value: f32) -> vec4<f32> {
    let t = clamp(
        (value - u_material.clim.x) / max(u_material.clim.y - u_material.clim.x, 1e-6),
        0.0, 1.0,
    );
    return textureSample(t_lut, s_lut, t);
}

fn sample_volume_world(wp: vec3<f32>) -> vec2<f32> {
    // (value, inside_flag) -- flag is 0 if the transformed tex coord is
    // outside [0,1] on any axis (strict STEP-style trivial reject).
    let tex4 = u_material.patient_to_texture * vec4<f32>(wp, 1.0);
    let tex = tex4.xyz;
    if (any(tex < vec3<f32>(0.0)) || any(tex > vec3<f32>(1.0))) {
        return vec2<f32>(0.0, 0.0);
    }
    let v = textureSample(t_volume, s_volume, tex).r;
    return vec2<f32>(v, 1.0);
}

// Clamp-to-edge sampling: used for gradient neighbors so gradient stays
// well-defined at the outermost samples (otherwise trivial-reject would
// zero one side of the central difference and bias the normal).
fn sample_volume_clamped(wp: vec3<f32>) -> f32 {
    let tex4 = u_material.patient_to_texture * vec4<f32>(wp, 1.0);
    let tex = clamp(tex4.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    return textureSample(t_volume, s_volume, tex).r;
}

fn compute_gradient_world(wp: vec3<f32>, h_mm: f32) -> vec3<f32> {
    // Central differences in mm. Divide by (2*h) to get a physical gradient
    // in per-mm units (numerically well-behaved for f32; HU span is ~4000
    // and we divide by ~2mm -> magnitudes O(1000), plenty of room).
    let dx = vec3<f32>(h_mm, 0.0, 0.0);
    let dy = vec3<f32>(0.0, h_mm, 0.0);
    let dz = vec3<f32>(0.0, 0.0, h_mm);
    let gx = sample_volume_clamped(wp + dx) - sample_volume_clamped(wp - dx);
    let gy = sample_volume_clamped(wp + dy) - sample_volume_clamped(wp - dy);
    let gz = sample_volume_clamped(wp + dz) - sample_volume_clamped(wp - dz);
    return vec3<f32>(gx, gy, gz) / (2.0 * h_mm);
}


// ---- Fragment: ray march ----

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    var out: FragmentOutput;

    // @builtin(position) is in framebuffer pixels (physical). Convert to NDC.
    let size = u_stdinfo.physical_size;
    let ndc_x = (varyings.position.x / size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (varyings.position.y / size.y) * 2.0;

    // Inverse-project NDC near & far to world space. We must honor the
    // camera's chirality (pygfx's cam_transform can flip the z convention).
    let cs = sign(
        u_stdinfo.cam_transform[0][0] *
        u_stdinfo.cam_transform[1][1] *
        u_stdinfo.cam_transform[2][2]
    );
    let world_near = ndc_to_world_pos(vec4<f32>(ndc_x, ndc_y, -cs, 1.0));
    let world_far  = ndc_to_world_pos(vec4<f32>(ndc_x, ndc_y,  cs, 1.0));
    let ray_origin = world_near;
    let ray_dir    = normalize(world_far - world_near);

    // Ray/AABB slab intersection in world space
    let bmin = u_material.bounds_min.xyz;
    let bmax = u_material.bounds_max.xyz;
    let inv_dir = vec3<f32>(1.0) / ray_dir;
    let tb = (bmin - ray_origin) * inv_dir;
    let tt = (bmax - ray_origin) * inv_dir;
    let tmn = min(tt, tb);
    let tmx = max(tt, tb);
    var t_near = max(max(tmn.x, tmn.y), tmn.z);
    var t_far  = min(min(tmx.x, tmx.y), tmx.z);

    if (t_far <= t_near || t_far <= 0.0) {
        out.color = u_material.background;
        return out;
    }

    let step = max(u_material.sample_step, 1e-3);
    let unit = max(u_material.opacity_unit_distance, 1e-3);

    // RenderCL-style inset: start one step inside, end one step before exit
    t_near = max(t_near + step, 0.0);
    t_far  = t_far - step;
    if (t_far <= t_near) {
        out.color = u_material.background;
        return out;
    }

    // Per-pixel jitter to kill wood-grain
    let seed = fract(sin(dot(vec3<f32>(varyings.position.xy, 0.0),
                              vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
    var t = t_near + seed * step;

    var integrated = vec4<f32>(0.0);
    var safety: i32 = 0;
    let max_steps: i32 = 2048;

    let k_a = u_material.k_ambient;
    let k_d = u_material.k_diffuse;
    let k_s = u_material.k_specular;
    let sh  = u_material.shininess;

    loop {
        if (t >= t_far)          { break; }
        if (safety >= max_steps) { break; }
        if (integrated.a >= 0.99){ break; }

        let wp = ray_origin + ray_dir * t;
        let s = sample_volume_world(wp);
        if (s.y > 0.5) {
            let tf = sample_lut(s.x);
            let base_a = tf.a;
            let grad = compute_gradient_world(wp, step);
            let grad_len = length(grad);

            var opacity = base_a * (step / unit);
            if (u_material.gradient_opacity_enabled > 0.5) {
                let gmin = u_material.gradient_range.x;
                let gmax = max(u_material.gradient_range.y, gmin + 1e-6);
                let gnorm = clamp((grad_len - gmin) / (gmax - gmin), 0.0, 1.0);
                let gmul = textureSample(t_grad_lut, s_grad_lut, gnorm).r;
                opacity = opacity * gmul;
            }
            opacity = clamp(opacity, 0.0, 1.0);

            if (opacity > 0.001) {
                // Phong lighting is applied in sRGB space because the lighting
                // coefficients (ambient, diffuse, specular) in volume-rendering
                // presets are authored against sRGB color values (that's what
                // the VTK/Slicer UI calibrates). Doing the Phong math on the
                // already-linearized TF color would lift shadows (ambient=0.1
                // on linear 0.6 gives linear 0.06 = sRGB 0.30, which looks too
                // light, not the sRGB 0.08 the preset intends).
                var lit_srgb = tf.rgb * k_a;
                if (grad_len > 1e-6) {
                    var n = grad / grad_len;
                    if (dot(n, -ray_dir) < 0.0) { n = -n; }
                    // Headlight: light is co-located with the camera, so
                    // to_light points from the sample back toward the eye.
                    // Same vector is also the view direction -> matches VTK's
                    // default vtkLight(lightType=1) behavior and stays stable
                    // as the camera orbits the volume.
                    let view_dir = normalize(ray_origin - wp);
                    let to_light = view_dir;
                    let ldotn = dot(to_light, n);
                    if (ldotn > 0.0) {
                        let refl = normalize(2.0 * ldotn * n - to_light);
                        let rdotv = max(0.0, dot(refl, view_dir));
                        lit_srgb = tf.rgb * (k_a + k_d * ldotn)
                                 + vec3<f32>(k_s * pow(rdotv, sh));
                    }
                }
                // Convert LIT sRGB -> linear for the actual front-to-back
                // alpha integration (the physically correct place for linear).
                let lit = srgb2physical(clamp(lit_srgb, vec3<f32>(0.0), vec3<f32>(1.0)));

                integrated.r = integrated.r + (1.0 - integrated.a) * opacity * lit.r;
                integrated.g = integrated.g + (1.0 - integrated.a) * opacity * lit.g;
                integrated.b = integrated.b + (1.0 - integrated.a) * opacity * lit.b;
                integrated.a = integrated.a + (1.0 - integrated.a) * opacity;
            }
        }

        t = t + step;
        safety = safety + 1;
    }

    // Mix background in LINEAR space. Background uniform is stored as sRGB
    // (matches Slicer's UI convention), so convert before the mix.
    let bg = srgb2physical(u_material.background.rgb);
    let final_linear = mix(bg, integrated.rgb, integrated.a);

    // The swapchain format is rgba8unorm-srgb: WebGPU auto-encodes
    // linear -> sRGB on write. Output LINEAR here; a manual OETF would
    // double-encode and over-brighten by ~44%.
    out.color = vec4<f32>(final_linear, 1.0);
    return out;
}
"""


# ------------------------------------------------------------------------
# Material
# ------------------------------------------------------------------------

class SlicerVolumeMaterial(pygfx.Material):
    """Single-volume material for the STEP-port renderer."""

    uniform_type = dict(
        pygfx.Material.uniform_type,
        patient_to_texture="4x4xf4",
        bounds_min="4xf4",
        bounds_max="4xf4",
        clim="4xf4",
        gradient_range="4xf4",
        point_light="4xf4",
        light_direction="4xf4",
        background="4xf4",
        sample_step="f4",
        opacity_unit_distance="f4",
        gradient_opacity_enabled="f4",
        k_ambient="f4",
        k_diffuse="f4",
        k_specular="f4",
        shininess="f4",
        _pad="f4",
    )

    def __init__(self, **kwargs):
        super().__init__()
        self._volume_tex: pygfx.Texture | None = None
        self._lut_tex: pygfx.Texture | None = None
        self._interpolation = "linear"

        self._grad_lut_tex: pygfx.Texture | None = None

        # Defaults (can be overridden by kwargs below)
        self.sample_step = 1.0
        self.opacity_unit_distance = 1.0
        self.gradient_opacity_enabled = 0.0  # 0 disables; 1 uses the grad LUT
        self.clim = (0.0, 255.0)
        self.gradient_range = (0.0, 1.0)
        self.bounds_min = (-100.0, -100.0, -100.0)
        self.bounds_max = (100.0, 100.0, 100.0)
        self.point_light = (250.0, 250.0, 400.0)
        self.light_direction = (0.0, 0.0, -1.0)
        self.k_ambient = 0.4
        self.k_diffuse = 0.7
        self.k_specular = 0.2
        self.shininess = 10.0
        self.background = (0.5, 0.5, 0.7, 1.0)
        self.patient_to_texture = np.eye(4, dtype=np.float32)

        for k, v in kwargs.items():
            setattr(self, k, v)

    # --- uniform helpers ---

    def _set_scalar(self, name, value):
        self.uniform_buffer.data[name] = float(value)
        self.uniform_buffer.update_full()

    def _set_vec4(self, name, value):
        arr = np.zeros(4, dtype=np.float32)
        v = np.asarray(value, dtype=np.float32).ravel()
        n = min(len(v), 4)
        arr[:n] = v[:n]
        if len(v) == 3:
            arr[3] = 1.0
        self.uniform_buffer.data[name] = arr
        self.uniform_buffer.update_full()

    def _set_mat4(self, name, value):
        m = np.asarray(value, dtype=np.float32).reshape(4, 4)
        # wgpu expects column-major; numpy row-major -> transpose
        self.uniform_buffer.data[name] = m.T
        self.uniform_buffer.update_full()

    # scalar properties
    def _sprop(n): return property(lambda self: float(self.uniform_buffer.data[n]),
                                   lambda self, v: self._set_scalar(n, v))
    sample_step              = _sprop("sample_step")
    opacity_unit_distance    = _sprop("opacity_unit_distance")
    gradient_opacity_enabled = _sprop("gradient_opacity_enabled")
    k_ambient             = _sprop("k_ambient")
    k_diffuse             = _sprop("k_diffuse")
    k_specular            = _sprop("k_specular")
    shininess             = _sprop("shininess")
    del _sprop

    @property
    def clim(self):
        d = self.uniform_buffer.data["clim"]
        return float(d[0]), float(d[1])
    @clim.setter
    def clim(self, v):
        self._set_vec4("clim", (float(v[0]), float(v[1]), 0.0, 0.0))

    @property
    def gradient_range(self):
        d = self.uniform_buffer.data["gradient_range"]
        return float(d[0]), float(d[1])
    @gradient_range.setter
    def gradient_range(self, v):
        self._set_vec4("gradient_range", (float(v[0]), float(v[1]), 0.0, 0.0))

    @property
    def bounds_min(self):
        return tuple(float(x) for x in self.uniform_buffer.data["bounds_min"][:3])
    @bounds_min.setter
    def bounds_min(self, v): self._set_vec4("bounds_min", v)

    @property
    def bounds_max(self):
        return tuple(float(x) for x in self.uniform_buffer.data["bounds_max"][:3])
    @bounds_max.setter
    def bounds_max(self, v): self._set_vec4("bounds_max", v)

    @property
    def point_light(self):
        return tuple(float(x) for x in self.uniform_buffer.data["point_light"][:3])
    @point_light.setter
    def point_light(self, v): self._set_vec4("point_light", v)

    @property
    def light_direction(self):
        return tuple(float(x) for x in self.uniform_buffer.data["light_direction"][:3])
    @light_direction.setter
    def light_direction(self, v): self._set_vec4("light_direction", v)

    @property
    def background(self):
        return tuple(float(x) for x in self.uniform_buffer.data["background"])
    @background.setter
    def background(self, v): self._set_vec4("background", v)

    @property
    def patient_to_texture(self):
        return np.array(self.uniform_buffer.data["patient_to_texture"]).T
    @patient_to_texture.setter
    def patient_to_texture(self, m): self._set_mat4("patient_to_texture", m)

    @property
    def volume_texture(self): return self._volume_tex
    @volume_texture.setter
    def volume_texture(self, tex): self._volume_tex = tex

    @property
    def lut_texture(self): return self._lut_tex
    @lut_texture.setter
    def lut_texture(self, tex): self._lut_tex = tex

    @property
    def grad_lut_texture(self): return self._grad_lut_tex
    @grad_lut_texture.setter
    def grad_lut_texture(self, tex): self._grad_lut_tex = tex

    @property
    def interpolation(self): return self._interpolation
    @interpolation.setter
    def interpolation(self, v):
        assert v in ("nearest", "linear")
        self._interpolation = v


# ------------------------------------------------------------------------
# WorldObject
# ------------------------------------------------------------------------

class SlicerVolumeRenderer(pygfx.WorldObject):
    """Screen-space volume renderer WorldObject. No geometry -- the shader
    emits a fullscreen triangle from the vertex index."""

    def __init__(self, material: SlicerVolumeMaterial | None = None, **kwargs):
        super().__init__(geometry=None, **kwargs)
        self.material = material if material is not None else SlicerVolumeMaterial()


# ------------------------------------------------------------------------
# Shader registration
# ------------------------------------------------------------------------

@register_wgpu_render_function(SlicerVolumeRenderer, SlicerVolumeMaterial)
class SlicerVolumeShader(BaseShader):
    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)

    def get_bindings(self, wobject, shared, scene):
        material = wobject.material

        bindings = [
            Binding("u_stdinfo",  "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject",  "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        if material.volume_texture is not None:
            vt = GfxTextureView(material.volume_texture)
            vs = GfxSampler(material.interpolation, "clamp")
            bindings.append(Binding("s_volume", "sampler/filtering", vs, FRAGMENT_ONLY))
            bindings.append(Binding("t_volume", "texture/auto", vt, FRAGMENT_ONLY))

        if material.lut_texture is not None:
            lt = GfxTextureView(material.lut_texture)
            ls = GfxSampler("linear", "clamp")
            bindings.append(Binding("s_lut", "sampler/filtering", ls, FRAGMENT_ONLY))
            bindings.append(Binding("t_lut", "texture/auto", lt, FRAGMENT_ONLY))

        if material.grad_lut_texture is not None:
            glt = GfxTextureView(material.grad_lut_texture)
            gls = GfxSampler("linear", "clamp")
            bindings.append(Binding("s_grad_lut", "sampler/filtering", gls, FRAGMENT_ONLY))
            bindings.append(Binding("t_grad_lut", "texture/auto", glt, FRAGMENT_ONLY))

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        return {"indices": (3, 1)}  # fullscreen triangle

    def get_code(self):
        return SHADER_WGSL


# ------------------------------------------------------------------------
# MRML helpers
# ------------------------------------------------------------------------

def patient_to_texture_matrix(volume_node) -> np.ndarray:
    """4x4 matrix mapping RAS (patient) coords -> texture [0,1] coords.

    Pipeline: RAS -> IJK -> texture. Texture coord for voxel index (i,j,k)
    is ((i+0.5)/I, (j+0.5)/J, (k+0.5)/K) so voxel centers land inside [0,1].
    """
    import vtk

    m = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(m)
    ras_to_ijk = np.array(
        [[m.GetElement(i, j) for j in range(4)] for i in range(4)],
        dtype=np.float64,
    )
    dims = volume_node.GetImageData().GetDimensions()  # (I, J, K) in VTK order
    ijk_to_tex = np.eye(4, dtype=np.float64)
    for axis in range(3):
        ijk_to_tex[axis, axis] = 1.0 / dims[axis]
        ijk_to_tex[axis, 3] = 0.5 / dims[axis]
    return (ijk_to_tex @ ras_to_ijk).astype(np.float32)


def build_lut_texture(n_samples=256, scalar_range=(0.0, 255.0), vr_display_node=None):
    lut = np.zeros((n_samples, 4), dtype=np.float32)
    lo, hi = scalar_range
    if vr_display_node is not None:
        vp = vr_display_node.GetVolumePropertyNode().GetVolumeProperty()
        color_fn = vp.GetRGBTransferFunction()
        op_fn = vp.GetScalarOpacity()
        color_tab = np.zeros(n_samples * 3, dtype=np.float64)
        op_tab = np.zeros(n_samples, dtype=np.float64)
        color_fn.GetTable(lo, hi, n_samples, color_tab)
        op_fn.GetTable(lo, hi, n_samples, op_tab)
        lut[:, 0:3] = color_tab.reshape(n_samples, 3).astype(np.float32)
        lut[:, 3] = op_tab.astype(np.float32)
    else:
        lut[:, 0:3] = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)[:, None]
        lut[:, 3] = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    return pygfx.Texture(lut, dim=1)


def build_gradient_opacity_lut(vr_display_node, n_samples=256):
    """Build a 1D LUT for gradient-opacity modulation from the MRML volume
    property node. Returns (lut_texture, (range_lo, range_hi))."""
    vp = vr_display_node.GetVolumePropertyNode().GetVolumeProperty()
    go_fn = vp.GetGradientOpacity(0)
    lut = np.ones(n_samples, dtype=np.float32)  # default: fully enabled
    gmin, gmax = 0.0, 1.0
    if go_fn is not None and go_fn.GetSize() > 0:
        r = go_fn.GetRange()
        gmin, gmax = float(r[0]), float(r[1])
        if gmax <= gmin:
            gmax = gmin + 1.0
        tab = np.zeros(n_samples, dtype=np.float64)
        go_fn.GetTable(gmin, gmax, n_samples, tab)
        lut[:] = tab.astype(np.float32)
    return pygfx.Texture(lut.reshape(n_samples, 1), dim=1), (gmin, gmax)


def scene_light_direction(three_d_view) -> tuple:
    """Return the (dir.x, dir.y, dir.z) world-space direction of the first
    enabled VTK light in a Slicer 3D view, or (0, 0, -1) if none found."""
    try:
        rw = three_d_view.renderWindow()
        renderer = rw.GetRenderers().GetItemAsObject(0)
        lights = renderer.GetLights()
        lights.InitTraversal()
        for _ in range(lights.GetNumberOfItems()):
            l = lights.GetNextItem()
            if l is None or not l.GetSwitch():
                continue
            pos = l.GetPosition()
            fp = l.GetFocalPoint()
            d = (fp[0] - pos[0], fp[1] - pos[1], fp[2] - pos[2])
            n = (d[0]**2 + d[1]**2 + d[2]**2) ** 0.5
            if n > 1e-6:
                return (d[0]/n, d[1]/n, d[2]/n)
    except Exception:
        pass
    return (0.0, 0.0, -1.0)


def build_renderer_for_volume(volume_node, vr_display_node=None) -> SlicerVolumeRenderer:
    """Create a SlicerVolumeRenderer configured for a MRML volume node."""
    import slicer

    arr = slicer.util.arrayFromVolume(volume_node).astype(np.float32, copy=False)
    dmin, dmax = float(arr.min()), float(arr.max())

    # Clim from the TF domain when available; fall back to volume range.
    if vr_display_node is not None:
        vp = vr_display_node.GetVolumePropertyNode().GetVolumeProperty()
        op_range = vp.GetScalarOpacity().GetRange()
        tf_lo, tf_hi = float(op_range[0]), float(op_range[1])
        smin = max(tf_lo, dmin)
        smax = min(tf_hi, dmax)
        if smax <= smin:
            smin, smax = dmin, dmax
    else:
        smin, smax = dmin, dmax

    vol_tex = pygfx.Texture(arr, dim=3)
    lut_tex = build_lut_texture(256, (smin, smax), vr_display_node)

    mat = SlicerVolumeMaterial(clim=(smin, smax))
    mat.volume_texture = vol_tex
    mat.lut_texture = lut_tex

    bounds = [0.0] * 6
    volume_node.GetBounds(bounds)
    mat.bounds_min = (bounds[0], bounds[2], bounds[4])
    mat.bounds_max = (bounds[1], bounds[3], bounds[5])

    mat.patient_to_texture = patient_to_texture_matrix(volume_node)

    spacing = volume_node.GetSpacing()
    min_spacing = float(min(spacing))
    mat.sample_step = min_spacing

    if vr_display_node is not None:
        vp = vr_display_node.GetVolumePropertyNode().GetVolumeProperty()
        mat.opacity_unit_distance = float(vp.GetScalarOpacityUnitDistance())
        if vp.GetShade():
            mat.k_ambient  = float(vp.GetAmbient())
            mat.k_diffuse  = float(vp.GetDiffuse())
            mat.k_specular = float(vp.GetSpecular())
            mat.shininess  = float(vp.GetSpecularPower())
    else:
        mat.opacity_unit_distance = min_spacing

    if vr_display_node is not None:
        grad_lut, grad_range = build_gradient_opacity_lut(vr_display_node)
        mat.grad_lut_texture = grad_lut
        mat.gradient_range = grad_range
        mat.gradient_opacity_enabled = 1.0
    else:
        mat.gradient_opacity_enabled = 0.0

    # The shader uses a per-pixel headlight, but we still capture the VTK
    # scene light direction so callers that want a world-fixed light can
    # swap the shader without touching Python.
    try:
        lm = slicer.app.layoutManager()
        if lm.threeDViewCount > 0:
            mat.light_direction = scene_light_direction(lm.threeDWidget(0).threeDView())
    except Exception:
        mat.light_direction = (0.0, 0.0, -1.0)

    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2
    extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    mat.point_light = (cx + extent, cy + extent, cz + extent)

    return SlicerVolumeRenderer(mat)
