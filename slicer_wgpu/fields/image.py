"""ImageField: a 3D scalar volume + colour transfer-function LUT +
optional gradient-opacity LUT, contributing to the per-sample compositing
loop of the SceneRenderer.

The sampling/transfer-function WGSL is factored out of the single-volume
renderer in slicer_wgpu.demos.single_volume so it can be combined with
other fields (more volumes, fiducials, transforms, ...) at the same ray
sample point.
"""

from __future__ import annotations

import numpy as np
import pygfx
import wgpu

from pygfx.renderers.wgpu import (
    Binding,
    GfxSampler,
    GfxTextureView,
)

from .base import Field

FRAGMENT_ONLY = wgpu.ShaderStage.FRAGMENT


def _patient_to_texture_matrix(volume_node) -> np.ndarray:
    """4x4 mapping RAS -> tex coords [0,1]^3 (voxel centres)."""
    import vtk
    m = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(m)
    ras_to_ijk = np.array(
        [[m.GetElement(i, j) for j in range(4)] for i in range(4)],
        dtype=np.float64,
    )
    dims = volume_node.GetImageData().GetDimensions()
    ijk_to_tex = np.eye(4, dtype=np.float64)
    for axis in range(3):
        ijk_to_tex[axis, axis] = 1.0 / dims[axis]
        ijk_to_tex[axis, 3] = 0.5 / dims[axis]
    return (ijk_to_tex @ ras_to_ijk).astype(np.float32)


def _build_lut_array(vr_display_node, n_samples, scalar_range):
    """Sample (color, opacity) for n_samples scalar values into an (n,4) f32 array."""
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
    return lut


def _build_gradient_opacity_lut_array(vr_display_node, n_samples=256):
    """(lut_array (n,1), (gmin, gmax)). Default flat = 1 if no GO function."""
    lut = np.ones(n_samples, dtype=np.float32)
    gmin, gmax = 0.0, 1.0
    if vr_display_node is not None:
        vp = vr_display_node.GetVolumePropertyNode().GetVolumeProperty()
        go_fn = vp.GetGradientOpacity(0)
        if go_fn is not None and go_fn.GetSize() > 0:
            r = go_fn.GetRange()
            gmin, gmax = float(r[0]), float(r[1])
            if gmax <= gmin:
                gmax = gmin + 1.0
            tab = np.zeros(n_samples, dtype=np.float64)
            go_fn.GetTable(gmin, gmax, n_samples, tab)
            lut[:] = tab.astype(np.float32)
    return lut.reshape(n_samples, 1), (gmin, gmax)


class ImageField(Field):
    """A scalar 3D volume rendered with a colour TF + gradient-opacity LUT.

    Construct with `ImageField.from_volume_node(volume_node, vr_display_node)`
    to populate textures and uniforms from MRML; or assemble the parts
    manually for testing.
    """

    field_kind = "img"

    def __init__(
        self,
        volume_array: np.ndarray | None = None,
        lut_array: np.ndarray | None = None,
        grad_lut_array: np.ndarray | None = None,
        clim: tuple[float, float] = (0.0, 255.0),
        gradient_range: tuple[float, float] = (0.0, 1.0),
        bounds_min=(-100.0, -100.0, -100.0),
        bounds_max=(100.0, 100.0, 100.0),
        patient_to_texture: np.ndarray | None = None,
        sample_step_mm: float = 1.0,
        opacity_unit_distance: float = 1.0,
        gradient_opacity_enabled: bool = False,
        k_ambient: float = 0.4,
        k_diffuse: float = 0.7,
        k_specular: float = 0.2,
        shininess: float = 10.0,
        visible: bool = True,
        interpolation: str = "linear",
        world_from_local: np.ndarray | None = None,
    ):
        super().__init__()
        self._volume_tex = (
            pygfx.Texture(volume_array.astype(np.float32, copy=False), dim=3)
            if volume_array is not None else None
        )
        self._lut_tex = (
            pygfx.Texture(lut_array.astype(np.float32, copy=False), dim=1)
            if lut_array is not None else None
        )
        self._grad_lut_tex = (
            pygfx.Texture(grad_lut_array.astype(np.float32, copy=False), dim=1)
            if grad_lut_array is not None else None
        )
        self._interpolation = interpolation

        self.clim = tuple(float(x) for x in clim)
        self.gradient_range = tuple(float(x) for x in gradient_range)
        self.bounds_min = tuple(float(x) for x in bounds_min)
        self.bounds_max = tuple(float(x) for x in bounds_max)
        self.patient_to_texture = (
            np.eye(4, dtype=np.float32) if patient_to_texture is None
            else np.asarray(patient_to_texture, dtype=np.float32)
        )
        # Transform from the volume's local (RAS) space to world space --
        # i.e. the 4x4 that a vtkMRMLTransformNode.GetMatrixTransformToWorld
        # returns for the volume. Identity means "volume is parked at the
        # world origin" (Slicer's default). We fold its inverse into the
        # sampling matrix at fill_uniforms() time so the WGSL side stays
        # untouched and the Phong gradient naturally captures any
        # non-rigid stretch.
        self.world_from_local = (
            np.eye(4, dtype=np.float32) if world_from_local is None
            else np.asarray(world_from_local, dtype=np.float32).reshape(4, 4)
        )
        self.sample_step_mm = float(sample_step_mm)
        self.opacity_unit_distance = float(opacity_unit_distance)
        self.gradient_opacity_enabled = bool(gradient_opacity_enabled)
        self.k_ambient = float(k_ambient)
        self.k_diffuse = float(k_diffuse)
        self.k_specular = float(k_specular)
        self.shininess = float(shininess)
        self.visible = bool(visible)

    # -------- Construction from MRML --------

    @classmethod
    def from_volume_node(cls, volume_node, vr_display_node=None) -> "ImageField":
        import slicer
        arr = slicer.util.arrayFromVolume(volume_node).astype(np.float32, copy=False)
        dmin, dmax = float(arr.min()), float(arr.max())

        if vr_display_node is not None:
            vp = vr_display_node.GetVolumePropertyNode().GetVolumeProperty()
            tf_lo, tf_hi = vp.GetScalarOpacity().GetRange()
            smin = max(float(tf_lo), dmin)
            smax = min(float(tf_hi), dmax)
            if smax <= smin:
                smin, smax = dmin, dmax
        else:
            smin, smax = dmin, dmax

        lut = _build_lut_array(vr_display_node, 256, (smin, smax))
        grad_lut, grad_range = _build_gradient_opacity_lut_array(vr_display_node)

        bounds = [0.0] * 6
        volume_node.GetBounds(bounds)
        bounds_min = (bounds[0], bounds[2], bounds[4])
        bounds_max = (bounds[1], bounds[3], bounds[5])

        spacing = volume_node.GetSpacing()
        sample_step = float(min(spacing))

        ka, kd, ks, sh = 0.4, 0.7, 0.2, 10.0
        unit_dist = sample_step
        if vr_display_node is not None:
            vp = vr_display_node.GetVolumePropertyNode().GetVolumeProperty()
            unit_dist = float(vp.GetScalarOpacityUnitDistance())
            if vp.GetShade():
                ka = float(vp.GetAmbient())
                kd = float(vp.GetDiffuse())
                ks = float(vp.GetSpecular())
                sh = float(vp.GetSpecularPower())

        visible = (
            vr_display_node is None or vr_display_node.GetVisibility() == 1
        )

        return cls(
            volume_array=arr,
            lut_array=lut,
            grad_lut_array=grad_lut,
            clim=(smin, smax),
            gradient_range=grad_range,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            patient_to_texture=_patient_to_texture_matrix(volume_node),
            sample_step_mm=sample_step,
            opacity_unit_distance=unit_dist,
            gradient_opacity_enabled=(vr_display_node is not None),
            k_ambient=ka, k_diffuse=kd, k_specular=ks, shininess=sh,
            visible=visible,
        )

    # -------- Field protocol --------

    def uniform_type(self, slot_idx: int) -> dict:
        p = f"img{slot_idx}"
        return {
            f"{p}_patient_to_texture": "4x4xf4",
            f"{p}_clim":               "4xf4",
            f"{p}_gradient_range":     "4xf4",
            f"{p}_bounds_min":         "4xf4",
            f"{p}_bounds_max":         "4xf4",
            f"{p}_opacity_unit_distance": "f4",
            f"{p}_sample_step":        "f4",
            f"{p}_gradient_opacity_enabled": "f4",
            f"{p}_visible":            "f4",
            f"{p}_k_ambient":          "f4",
            f"{p}_k_diffuse":          "f4",
            f"{p}_k_specular":         "f4",
            f"{p}_shininess":          "f4",
            # Pad to 16-byte boundary so the next field lands cleanly.
            f"{p}__pad0":              "f4",
            f"{p}__pad1":              "f4",
            f"{p}__pad2":              "f4",
        }

    def get_bindings(self, slot_idx: int) -> list:
        out = []
        if self._volume_tex is not None:
            out.append(Binding(
                f"s_vol{slot_idx}", "sampler/filtering",
                GfxSampler(self._interpolation, "clamp"), FRAGMENT_ONLY))
            out.append(Binding(
                f"t_vol{slot_idx}", "texture/auto",
                GfxTextureView(self._volume_tex), FRAGMENT_ONLY))
        if self._lut_tex is not None:
            out.append(Binding(
                f"s_lut{slot_idx}", "sampler/filtering",
                GfxSampler("linear", "clamp"), FRAGMENT_ONLY))
            out.append(Binding(
                f"t_lut{slot_idx}", "texture/auto",
                GfxTextureView(self._lut_tex), FRAGMENT_ONLY))
        if self._grad_lut_tex is not None:
            out.append(Binding(
                f"s_grad_lut{slot_idx}", "sampler/filtering",
                GfxSampler("linear", "clamp"), FRAGMENT_ONLY))
            out.append(Binding(
                f"t_grad_lut{slot_idx}", "texture/auto",
                GfxTextureView(self._grad_lut_tex), FRAGMENT_ONLY))
        return out

    def sampling_wgsl(self, slot_idx: int) -> str:
        i = slot_idx
        return f"""
fn sample_volume_world_img{i}(wp: vec3<f32>) -> vec2<f32> {{
    let tex4 = u_material.img{i}_patient_to_texture * vec4<f32>(wp, 1.0);
    let tex = tex4.xyz;
    if (any(tex < vec3<f32>(0.0)) || any(tex > vec3<f32>(1.0))) {{
        return vec2<f32>(0.0, 0.0);
    }}
    let v = textureSample(t_vol{i}, s_vol{i}, tex).r;
    return vec2<f32>(v, 1.0);
}}
fn sample_volume_clamped_img{i}(wp: vec3<f32>) -> f32 {{
    let tex4 = u_material.img{i}_patient_to_texture * vec4<f32>(wp, 1.0);
    let tex = clamp(tex4.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    return textureSample(t_vol{i}, s_vol{i}, tex).r;
}}
fn compute_gradient_world_img{i}(wp: vec3<f32>, h_mm: f32) -> vec3<f32> {{
    let dx = vec3<f32>(h_mm, 0.0, 0.0);
    let dy = vec3<f32>(0.0, h_mm, 0.0);
    let dz = vec3<f32>(0.0, 0.0, h_mm);
    let gx = sample_volume_clamped_img{i}(wp + dx) - sample_volume_clamped_img{i}(wp - dx);
    let gy = sample_volume_clamped_img{i}(wp + dy) - sample_volume_clamped_img{i}(wp - dy);
    let gz = sample_volume_clamped_img{i}(wp + dz) - sample_volume_clamped_img{i}(wp - dz);
    return vec3<f32>(gx, gy, gz) / (2.0 * h_mm);
}}
fn sample_field_img{i}(wp: vec3<f32>, ray_dir: vec3<f32>, ray_origin: vec3<f32>) -> FieldSample {{
    var out: FieldSample;
    out.hit = false;
    out.color_srgb = vec3<f32>(0.0);
    out.opacity = 0.0;
    if (u_material.img{i}_visible < 0.5) {{ return out; }}

    let s = sample_volume_world_img{i}(wp);
    if (s.y < 0.5) {{ return out; }}

    let clim = u_material.img{i}_clim;
    let t_lut = clamp((s.x - clim.x) / max(clim.y - clim.x, 1e-6), 0.0, 1.0);
    let tf = textureSample(t_lut{i}, s_lut{i}, t_lut);
    let base_a = tf.a;
    if (base_a <= 0.0) {{ return out; }}

    let step = max(u_material.img{i}_sample_step, 1e-3);
    let unit = max(u_material.img{i}_opacity_unit_distance, 1e-3);
    var opacity = base_a * (step / unit);

    let grad = compute_gradient_world_img{i}(wp, step);
    let grad_len = length(grad);

    if (u_material.img{i}_gradient_opacity_enabled > 0.5) {{
        let gmin = u_material.img{i}_gradient_range.x;
        let gmax = max(u_material.img{i}_gradient_range.y, gmin + 1e-6);
        let gnorm = clamp((grad_len - gmin) / (gmax - gmin), 0.0, 1.0);
        let gmul = textureSample(t_grad_lut{i}, s_grad_lut{i}, gnorm).r;
        opacity = opacity * gmul;
    }}
    opacity = clamp(opacity, 0.0, 1.0);
    if (opacity <= 0.001) {{ return out; }}

    // Phong in sRGB (the TF is authored in sRGB).
    var lit = tf.rgb * u_material.img{i}_k_ambient;
    if (grad_len > 1e-6) {{
        var n = grad / grad_len;
        if (dot(n, -ray_dir) < 0.0) {{ n = -n; }}
        let view_dir = normalize(ray_origin - wp);
        let to_light = view_dir;  // headlight
        let ldotn = dot(to_light, n);
        if (ldotn > 0.0) {{
            let refl = normalize(2.0 * ldotn * n - to_light);
            let rdotv = max(0.0, dot(refl, view_dir));
            lit = tf.rgb * (u_material.img{i}_k_ambient + u_material.img{i}_k_diffuse * ldotn)
                + vec3<f32>(u_material.img{i}_k_specular * pow(rdotv, u_material.img{i}_shininess));
        }}
    }}
    out.hit = true;
    out.color_srgb = clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0));
    out.opacity = opacity;
    return out;
}}
"""

    def tf_wgsl(self, slot_idx: int) -> str:
        # All TF + lighting work happens inside sample_field_imgN; this fn
        # is just an identity adapter so the generic compositing loop can
        # call it uniformly.
        i = slot_idx
        return f"""
fn tf_field_img{i}(s: FieldSample) -> vec4<f32> {{
    return vec4<f32>(s.color_srgb, s.opacity);
}}
"""

    # -------- TF fast path --------

    def refresh_from_display_node(self, volume_node, vr_display_node) -> None:
        """Re-read the colour / opacity / gradient-opacity TFs and
        shading parameters from the MRML display node and rewrite the
        two 1D LUT textures in place. Keeps the expensive 3D
        `_volume_tex` and the same ImageField instance, so live TF
        edits (Shift slider, preset swap, per-frame threshold sweeps)
        cost O(lut size) rather than O(voxels).

        Called by VolumeRenderingDisplayer's ModifiedEvent /
        InteractionEvent handler on the VolumePropertyNode.
        """
        import slicer
        vp = vr_display_node.GetVolumePropertyNode().GetVolumeProperty()
        arr = slicer.util.arrayFromVolume(volume_node)
        dmin, dmax = float(arr.min()), float(arr.max())
        tf_lo, tf_hi = vp.GetScalarOpacity().GetRange()
        smin = max(float(tf_lo), dmin)
        smax = min(float(tf_hi), dmax)
        if smax <= smin:
            smin, smax = dmin, dmax

        lut = _build_lut_array(vr_display_node, 256, (smin, smax))
        grad_lut, grad_range = _build_gradient_opacity_lut_array(
            vr_display_node)

        if self._lut_tex is not None:
            self._lut_tex.set_data(lut.astype(np.float32, copy=False))
        if self._grad_lut_tex is not None:
            self._grad_lut_tex.set_data(grad_lut.astype(np.float32, copy=False))

        self.clim = (smin, smax)
        self.gradient_range = grad_range
        self.opacity_unit_distance = float(vp.GetScalarOpacityUnitDistance())
        if vp.GetShade():
            self.k_ambient = float(vp.GetAmbient())
            self.k_diffuse = float(vp.GetDiffuse())
            self.k_specular = float(vp.GetSpecular())
            self.shininess = float(vp.GetSpecularPower())
        self.visible = (vr_display_node.GetVisibility() == 1)
        self.touch()

    # -------- Transform API --------

    def set_world_from_local(self, world_from_local: np.ndarray) -> None:
        """Update the volume's world-from-local transform (the 4x4 a
        vtkMRMLTransformNode.GetMatrixTransformToWorld would return).
        Folded into the sampling matrix on the next uniform refresh.
        Stretching or skewing this matrix makes the Phong gradient
        respond to the deformation automatically because the world-space
        central-difference neighbours map to non-uniformly-spaced
        texture coords.
        """
        M = np.asarray(world_from_local, dtype=np.float32).reshape(4, 4)
        if np.allclose(M, self.world_from_local):
            return
        self.world_from_local = M
        self.touch()

    # -------- CPU-side state --------

    def fill_uniforms(self, ub, slot_idx: int) -> None:
        p = f"img{slot_idx}"
        # Compose: wp (world) -> (world_to_local) -> (patient_to_texture).
        # The shader uniform is still called *_patient_to_texture; we just
        # reuse its slot for the full world-to-texture matrix so no WGSL
        # change is needed.
        world_to_local = np.linalg.inv(
            np.asarray(self.world_from_local, dtype=np.float64))
        world_to_texture = (
            np.asarray(self.patient_to_texture, dtype=np.float64) @ world_to_local
        ).astype(np.float32)
        ub.data[f"{p}_patient_to_texture"] = world_to_texture.T  # column-major
        cmin, cmax = self.clim
        ub.data[f"{p}_clim"] = np.array([cmin, cmax, 0.0, 0.0], dtype=np.float32)
        gmin, gmax = self.gradient_range
        ub.data[f"{p}_gradient_range"] = np.array([gmin, gmax, 0.0, 0.0], dtype=np.float32)
        ub.data[f"{p}_bounds_min"] = np.array([*self.bounds_min, 1.0], dtype=np.float32)
        ub.data[f"{p}_bounds_max"] = np.array([*self.bounds_max, 1.0], dtype=np.float32)
        ub.data[f"{p}_opacity_unit_distance"] = np.float32(self.opacity_unit_distance)
        ub.data[f"{p}_sample_step"] = np.float32(self.sample_step_mm)
        ub.data[f"{p}_gradient_opacity_enabled"] = np.float32(
            1.0 if self.gradient_opacity_enabled else 0.0)
        ub.data[f"{p}_visible"] = np.float32(1.0 if self.visible else 0.0)
        ub.data[f"{p}_k_ambient"] = np.float32(self.k_ambient)
        ub.data[f"{p}_k_diffuse"] = np.float32(self.k_diffuse)
        ub.data[f"{p}_k_specular"] = np.float32(self.k_specular)
        ub.data[f"{p}_shininess"] = np.float32(self.shininess)

    def aabb(self):
        # If a non-identity transform is set, union the 8 transformed
        # corners of the local AABB. For identity this collapses to the
        # plain local bounds; non-rigid transforms expand the AABB as
        # needed so the ray-march doesn't clip the deformed volume.
        lo = np.asarray(self.bounds_min, dtype=np.float64)
        hi = np.asarray(self.bounds_max, dtype=np.float64)
        M = np.asarray(self.world_from_local, dtype=np.float64)
        if np.allclose(M, np.eye(4)):
            return lo, hi
        corners = np.array([
            [lo[0], lo[1], lo[2], 1.0], [hi[0], lo[1], lo[2], 1.0],
            [lo[0], hi[1], lo[2], 1.0], [hi[0], hi[1], lo[2], 1.0],
            [lo[0], lo[1], hi[2], 1.0], [hi[0], lo[1], hi[2], 1.0],
            [lo[0], hi[1], hi[2], 1.0], [hi[0], hi[1], hi[2], 1.0],
        ], dtype=np.float64)
        w = (M @ corners.T).T[:, :3]
        return w.min(axis=0), w.max(axis=0)
