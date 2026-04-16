"""Light-space transmittance volume for soft volumetric shadows.

Stage-1 scope: ImageFields only, single directional light, 256^3 fixed
resolution, no cone softening / no TAA. Rebuilt on a dirty tick (light
changed, field mtimes changed, scene bounds changed). The rebuilt
texture is sampled trilinearly during the fragment ray march so soft
shadow edges fall out of the volume interpolation for free.

Compute-only uniform layout
---------------------------
Rather than mirror ``pygfx.Material.uniform_buffer``'s packed struct
into the compute pass -- which would couple us to pygfx's internal
field ordering + padding rules -- the compute pipeline has its own
minimal per-ImageField uniform. Each field uploads its world->texture
matrix, clim, opacity_unit_distance, and visibility through this
dedicated buffer; the volume / LUT / gradient-LUT textures and their
samplers are pulled directly from the ``pygfx.Texture`` objects the
fragment shader already binds (via ``ensure_wgpu_object``), so data is
shared, not duplicated.

The compute shader omits gradient-opacity modulation. Stage 1 eats the
fidelity cost (soft shadows are already blurred by trilinear lookup);
Stage 2 can re-add it if we find cases where ignoring gradient opacity
leaves artefacts.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pygfx
import wgpu

from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object, update_resource


# One per-ImageField compute uniform = 4x4 matrix (64B) + 2 vec4s (32B)
# + 4 scalars padded to a final vec4 (16B) = 112B, rounded up to 128B
# for std140 tail alignment.
_IMAGE_FIELD_UBO_BYTES = 128


_COMPUTE_TEMPLATE = """
struct ShadowUniforms {
    scene_bounds_min: vec4<f32>,
    scene_bounds_max: vec4<f32>,
    light_direction: vec4<f32>,
    step_size: f32,
    max_steps: u32,
    n_fields: u32,
    resolution: u32,
};

struct ImageFieldCompute {
    world_to_texture: mat4x4<f32>,
    clim: vec4<f32>,
    gradient_range: vec4<f32>,
    opacity_unit_distance: f32,
    sample_step: f32,
    gradient_opacity_enabled: f32,
    visible: f32,
};

@group(0) @binding(0) var<uniform> u_shadow: ShadowUniforms;
@group(0) @binding(1) var t_out: texture_storage_3d<r32float, write>;

__FIELD_BINDINGS__

__FIELD_FUNCTIONS__

fn ray_aabb(o: vec3<f32>, d: vec3<f32>,
            bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let inv = vec3<f32>(1.0) / d;
    let tb = (bmin - o) * inv;
    let tt = (bmax - o) * inv;
    let tmn = min(tt, tb);
    let tmx = max(tt, tb);
    return vec2<f32>(
        max(max(tmn.x, tmn.y), tmn.z),
        min(min(tmx.x, tmx.y), tmx.z),
    );
}

fn sample_opacity_all(wp: vec3<f32>, step: f32) -> f32 {
    var op: f32 = 0.0;
__FIELD_OPACITY_ACCUM__
    return clamp(op, 0.0, 0.999);
}

@compute @workgroup_size(4, 4, 4)
fn build_shadow(@builtin(global_invocation_id) id: vec3<u32>) {
    let R = u_shadow.resolution;
    if (id.x >= R || id.y >= R || id.z >= R) { return; }

    let bmin = u_shadow.scene_bounds_min.xyz;
    let bmax = u_shadow.scene_bounds_max.xyz;

    // Texel-center in world space.
    let uvw = (vec3<f32>(f32(id.x), f32(id.y), f32(id.z)) + vec3<f32>(0.5))
            / vec3<f32>(f32(R));
    let wp0 = mix(bmin, bmax, uvw);

    // light_direction points FROM surface TO light. Step toward the light
    // from the texel position until we exit the scene bounds or exceed
    // the safety step count.
    let dir = normalize(u_shadow.light_direction.xyz);
    let trange = ray_aabb(wp0, dir, bmin, bmax);
    // We start inside the box, so trange.x is negative (entering time).
    let t_exit = max(trange.y, 0.0);
    if (t_exit <= 0.0) {
        textureStore(t_out, vec3<i32>(id), vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }

    let step = max(u_shadow.step_size, 1e-3);
    // Offset half a step so the self-shading term (lit sample) isn't also
    // the first shadow-occluder sample.
    var t: f32 = 0.5 * step;
    var tau: f32 = 0.0;
    var safety: u32 = 0u;

    loop {
        if (t >= t_exit) { break; }
        if (safety >= u_shadow.max_steps) { break; }

        let wp = wp0 + dir * t;
        let alpha = sample_opacity_all(wp, step);
        // alpha = 1 - exp(-mu * step) => mu*step = -ln(1 - alpha)
        tau = tau + (-log(1.0 - alpha));
        if (tau > 6.0) { break; }   // transmittance < 0.25%, early out.

        t = t + step;
        safety = safety + 1u;
    }

    let transmittance = exp(-tau);
    textureStore(t_out, vec3<i32>(id), vec4<f32>(transmittance, 0.0, 0.0, 0.0));
}
"""


def _emit_field_bindings(n_fields: int, first_binding: int) -> str:
    """Per-ImageField bindings: 1 uniform buf + 3 sampler+texture pairs = 7 slots."""
    lines = []
    for i in range(n_fields):
        base = first_binding + 7 * i
        lines.append(
            f"@group(0) @binding({base + 0}) var<uniform> u_if{i}: ImageFieldCompute;"
            f"\n@group(0) @binding({base + 1}) var s_vol{i}: sampler;"
            f"\n@group(0) @binding({base + 2}) var t_vol{i}: texture_3d<f32>;"
            f"\n@group(0) @binding({base + 3}) var s_lut{i}: sampler;"
            f"\n@group(0) @binding({base + 4}) var t_lut{i}: texture_1d<f32>;"
            f"\n@group(0) @binding({base + 5}) var s_grad_lut{i}: sampler;"
            f"\n@group(0) @binding({base + 6}) var t_grad_lut{i}: texture_1d<f32>;"
        )
    return "\n".join(lines)


def _emit_field_functions(n_fields: int) -> str:
    chunks = []
    for i in range(n_fields):
        chunks.append(f"""
fn sample_opacity_if{i}(wp: vec3<f32>, step: f32) -> f32 {{
    if (u_if{i}.visible < 0.5) {{ return 0.0; }}
    let tex4 = u_if{i}.world_to_texture * vec4<f32>(wp, 1.0);
    let tex = tex4.xyz;
    if (any(tex < vec3<f32>(0.0)) || any(tex > vec3<f32>(1.0))) {{ return 0.0; }}
    let v = textureSampleLevel(t_vol{i}, s_vol{i}, tex, 0.0).r;
    let clim = u_if{i}.clim;
    let t_lut_c = clamp((v - clim.x) / max(clim.y - clim.x, 1e-6), 0.0, 1.0);
    let tf = textureSampleLevel(t_lut{i}, s_lut{i}, t_lut_c, 0.0);
    let base_a = tf.a;
    if (base_a <= 0.0) {{ return 0.0; }}
    let unit = max(u_if{i}.opacity_unit_distance, 1e-3);
    return clamp(base_a * (step / unit), 0.0, 1.0);
}}
""")
    return "".join(chunks)


def _emit_field_opacity_accum(n_fields: int) -> str:
    if n_fields == 0:
        return "    // (no fields)"
    lines = []
    for i in range(n_fields):
        lines.append(f"    op = op + sample_opacity_if{i}(wp, step);")
    return "\n".join(lines)


def _pack_image_field_uniform(field) -> bytes:
    """Pack an ImageField's compute uniforms into _IMAGE_FIELD_UBO_BYTES bytes.

    Layout matches ``struct ImageFieldCompute`` in WGSL:
        mat4x4<f32> world_to_texture  (64 B, column-major)
        vec4<f32>   clim              (16 B)
        vec4<f32>   gradient_range    (16 B)
        f32 opacity_unit_distance
        f32 sample_step
        f32 gradient_opacity_enabled
        f32 visible
    """
    buf = np.zeros(_IMAGE_FIELD_UBO_BYTES // 4, dtype=np.float32)
    # world_to_texture == patient_to_texture @ world_to_local
    world_to_local = np.linalg.inv(
        np.asarray(field.world_from_local, dtype=np.float64))
    m = (np.asarray(field.patient_to_texture, dtype=np.float64) @ world_to_local)
    # WGSL matrices are column-major; store the transpose.
    buf[0:16] = m.T.astype(np.float32).ravel()
    cmin, cmax = field.clim
    buf[16:20] = np.array([cmin, cmax, 0.0, 0.0], dtype=np.float32)
    gmin, gmax = field.gradient_range
    buf[20:24] = np.array([gmin, gmax, 0.0, 0.0], dtype=np.float32)
    buf[24] = float(field.opacity_unit_distance)
    buf[25] = float(field.sample_step_mm)
    buf[26] = 1.0 if field.gradient_opacity_enabled else 0.0
    buf[27] = 1.0 if field.visible else 0.0
    return buf.tobytes()


class ShadowVolume:
    """World-space 3D transmittance texture + compute pipeline.

    Lifecycle:
        sv = ShadowVolume(device, resolution)
        sv.build_pipeline_for_image_fields(image_fields)   # compile per-config
        sv.build(bmin, bmax, light_dir, image_fields)      # per-rebuild
        sv.texture_view                                    # bind into fragment
        sv.sampler
    """

    def __init__(self, device, resolution: int = 256):
        self.device = device
        self.resolution = int(resolution)

        # A pygfx.Texture wraps our wgpu GPUTexture, so the fragment shader
        # can sample it via pygfx's Binding mechanism, while the compute
        # pass writes it via a raw texture_view.
        usage = int(wgpu.TextureUsage.STORAGE_BINDING
                    | wgpu.TextureUsage.TEXTURE_BINDING
                    | wgpu.TextureUsage.COPY_SRC)
        self.pygfx_texture = pygfx.Texture(
            data=None, dim=3,
            size=(self.resolution, self.resolution, self.resolution),
            format="r32float", usage=usage,
        )
        ensure_wgpu_object(self.pygfx_texture)
        self.texture = self.pygfx_texture._wgpu_object
        self.texture_view = self.texture.create_view()
        self.sampler = device.create_sampler(
            label="slicer_wgpu.ShadowVolume.sampler",
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            address_mode_w=wgpu.AddressMode.clamp_to_edge,
        )

        # Scene-wide shadow uniforms.
        self.uniform_buffer = device.create_buffer(
            label="slicer_wgpu.ShadowVolume.uniforms",
            size=64,  # 3 vec4 + 4 scalars = 64
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self._pipeline = None
        self._bind_group_layout = None
        self._n_fields = 0
        self._per_field_ubos: list = []      # wgpu.GPUBuffer, one per field

    # -------- Pipeline lifecycle --------

    def build_pipeline_for_image_fields(self, image_fields: Sequence) -> None:
        """Compile the compute pipeline for this specific set of ImageFields.

        This populates per-field uniform buffers as well; the fragment shader
        and compute pass share the underlying wgpu.GPUTextures (volume,
        LUT, gradient LUT) by calling ``ensure_wgpu_object`` on each
        ``pygfx.Texture``.
        """
        n = len(image_fields)
        self._n_fields = n

        # Allocate per-field UBOs.
        self._per_field_ubos = [
            self.device.create_buffer(
                label=f"slicer_wgpu.ShadowVolume.if{i}.ubo",
                size=_IMAGE_FIELD_UBO_BYTES,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            for i in range(n)
        ]

        # Build the WGSL source.
        first_field_binding = 2
        wgsl = (_COMPUTE_TEMPLATE
                .replace("__FIELD_BINDINGS__",
                         _emit_field_bindings(n, first_field_binding))
                .replace("__FIELD_FUNCTIONS__", _emit_field_functions(n))
                .replace("__FIELD_OPACITY_ACCUM__",
                         _emit_field_opacity_accum(n)))
        self._wgsl = wgsl

        shader = self.device.create_shader_module(
            label="slicer_wgpu.ShadowVolume.compute", code=wgsl)

        # Bind-group layout: 0=u_shadow, 1=t_out, then 7 per field.
        entries = [
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {
                 "access": wgpu.StorageTextureAccess.write_only,
                 "format": wgpu.TextureFormat.r32float,
                 "view_dimension": wgpu.TextureViewDimension.d3,
             }},
        ]
        for i in range(n):
            base = first_field_binding + 7 * i
            entries.extend([
                {"binding": base + 0,
                 "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
                {"binding": base + 1,
                 "visibility": wgpu.ShaderStage.COMPUTE,
                 "sampler": {"type": wgpu.SamplerBindingType.filtering}},
                {"binding": base + 2,
                 "visibility": wgpu.ShaderStage.COMPUTE,
                 "texture": {
                     "sample_type": wgpu.TextureSampleType.float,
                     "view_dimension": wgpu.TextureViewDimension.d3,
                     "multisampled": False,
                 }},
                {"binding": base + 3,
                 "visibility": wgpu.ShaderStage.COMPUTE,
                 "sampler": {"type": wgpu.SamplerBindingType.filtering}},
                {"binding": base + 4,
                 "visibility": wgpu.ShaderStage.COMPUTE,
                 "texture": {
                     "sample_type": wgpu.TextureSampleType.float,
                     "view_dimension": wgpu.TextureViewDimension.d1,
                     "multisampled": False,
                 }},
                {"binding": base + 5,
                 "visibility": wgpu.ShaderStage.COMPUTE,
                 "sampler": {"type": wgpu.SamplerBindingType.filtering}},
                {"binding": base + 6,
                 "visibility": wgpu.ShaderStage.COMPUTE,
                 "texture": {
                     "sample_type": wgpu.TextureSampleType.float,
                     "view_dimension": wgpu.TextureViewDimension.d1,
                     "multisampled": False,
                 }},
            ])

        bgl = self.device.create_bind_group_layout(
            label="slicer_wgpu.ShadowVolume.bgl", entries=entries)
        pipeline_layout = self.device.create_pipeline_layout(
            label="slicer_wgpu.ShadowVolume.pl", bind_group_layouts=[bgl])
        self._pipeline = self.device.create_compute_pipeline(
            label="slicer_wgpu.ShadowVolume.pipeline",
            layout=pipeline_layout,
            compute={"module": shader, "entry_point": "build_shadow"},
        )
        self._bind_group_layout = bgl

    # -------- Per-frame build --------

    def _write_scene_uniforms(self, bmin, bmax, light_dir,
                              step_size, max_steps):
        buf = np.zeros(16, dtype=np.float32)
        buf[0:3] = np.asarray(bmin, dtype=np.float32)
        buf[4:7] = np.asarray(bmax, dtype=np.float32)
        ld = np.asarray(light_dir, dtype=np.float32)
        ld = ld / max(float(np.linalg.norm(ld)), 1e-6)
        buf[8:11] = ld
        buf[12] = float(step_size)
        ints = np.zeros(4, dtype=np.uint32)
        ints[0] = int(max_steps)
        ints[1] = int(self._n_fields)
        ints[2] = int(self.resolution)
        raw = bytearray(buf.tobytes())
        raw[13 * 4: 13 * 4 + 12] = ints[:3].tobytes()
        self.device.queue.write_buffer(self.uniform_buffer, 0, bytes(raw))

    def _write_field_uniforms(self, image_fields):
        for i, field in enumerate(image_fields):
            self.device.queue.write_buffer(
                self._per_field_ubos[i], 0, _pack_image_field_uniform(field))

    def build(self, bmin, bmax, light_dir, image_fields,
              step_size: float | None = None,
              max_steps: int = 4096) -> None:
        """Dispatch the compute shader to refresh the transmittance texture.

        ``light_dir`` points FROM surface TO light. ``step_size`` defaults
        to one texel-diagonal of the shadow volume if not supplied.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "ShadowVolume.build called before build_pipeline_for_image_fields")
        if len(image_fields) != self._n_fields:
            raise RuntimeError(
                f"ShadowVolume.build: field count {len(image_fields)} "
                f"does not match pipeline ({self._n_fields}).")

        if step_size is None:
            extents = np.asarray(bmax, dtype=np.float64) - np.asarray(bmin, dtype=np.float64)
            step_size = float(np.linalg.norm(extents) / max(self.resolution, 1))
            step_size = max(step_size, 1e-3)

        self._write_scene_uniforms(bmin, bmax, light_dir, step_size, max_steps)
        self._write_field_uniforms(image_fields)

        # Walk each field, reach into its pygfx Textures, make sure the
        # wgpu upload has happened, and pull the GPU handles.
        entries = [
            {"binding": 0, "resource": {
                "buffer": self.uniform_buffer, "offset": 0,
                "size": self.uniform_buffer.size}},
            {"binding": 1, "resource": self.texture_view},
        ]
        first_field_binding = 2
        for i, field in enumerate(image_fields):
            base = first_field_binding + 7 * i
            # If pygfx hasn't rendered the field yet, the texture was never
            # marked as needing TEXTURE_BINDING, so the upload produced a
            # COPY_DST-only GPUTexture. Set the flag before upload so the
            # created GPUTexture is bindable in our compute pass.
            for pgt in (field._volume_tex, field._lut_tex, field._grad_lut_tex):
                if pgt is None:
                    continue
                if pgt._wgpu_object is None:
                    pgt._wgpu_usage |= wgpu.TextureUsage.TEXTURE_BINDING
                ensure_wgpu_object(pgt)
                # ensure_wgpu_object only creates the GPUTexture; the actual
                # pixel upload happens in update_resource (normally driven
                # by the render path before each draw). Run it explicitly
                # here so we can read from the texture in our compute pass
                # even if no render has happened yet.
                update_resource(pgt)
            vol_gpu = field._volume_tex._wgpu_object
            lut_gpu = field._lut_tex._wgpu_object
            grad_gpu = field._grad_lut_tex._wgpu_object
            vol_view = vol_gpu.create_view()
            lut_view = lut_gpu.create_view()
            grad_view = grad_gpu.create_view()
            # Matching sampler flavours to the fragment shader's usage:
            # volume uses field._interpolation; LUTs use linear.
            vol_sampler = self.device.create_sampler(
                label=f"slicer_wgpu.ShadowVolume.if{i}.s_vol",
                mag_filter=wgpu.FilterMode.linear,
                min_filter=wgpu.FilterMode.linear,
                address_mode_u=wgpu.AddressMode.clamp_to_edge,
                address_mode_v=wgpu.AddressMode.clamp_to_edge,
                address_mode_w=wgpu.AddressMode.clamp_to_edge,
            )
            lut_sampler = self.device.create_sampler(
                label=f"slicer_wgpu.ShadowVolume.if{i}.s_lut",
                mag_filter=wgpu.FilterMode.linear,
                min_filter=wgpu.FilterMode.linear,
                address_mode_u=wgpu.AddressMode.clamp_to_edge,
            )
            grad_sampler = self.device.create_sampler(
                label=f"slicer_wgpu.ShadowVolume.if{i}.s_grad_lut",
                mag_filter=wgpu.FilterMode.linear,
                min_filter=wgpu.FilterMode.linear,
                address_mode_u=wgpu.AddressMode.clamp_to_edge,
            )
            entries.extend([
                {"binding": base + 0, "resource": {
                    "buffer": self._per_field_ubos[i],
                    "offset": 0,
                    "size": self._per_field_ubos[i].size}},
                {"binding": base + 1, "resource": vol_sampler},
                {"binding": base + 2, "resource": vol_view},
                {"binding": base + 3, "resource": lut_sampler},
                {"binding": base + 4, "resource": lut_view},
                {"binding": base + 5, "resource": grad_sampler},
                {"binding": base + 6, "resource": grad_view},
            ])

        bind_group = self.device.create_bind_group(
            label="slicer_wgpu.ShadowVolume.bg",
            layout=self._bind_group_layout, entries=entries)

        encoder = self.device.create_command_encoder(
            label="slicer_wgpu.ShadowVolume.encoder")
        cpass = encoder.begin_compute_pass(
            label="slicer_wgpu.ShadowVolume.pass")
        cpass.set_pipeline(self._pipeline)
        cpass.set_bind_group(0, bind_group, [], 0, 0)
        wg = 4
        groups = (self.resolution + wg - 1) // wg
        cpass.dispatch_workgroups(groups, groups, groups)
        cpass.end()
        self.device.queue.submit([encoder.finish()])
