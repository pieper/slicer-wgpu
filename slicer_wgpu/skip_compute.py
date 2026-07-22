"""Compute-pass builders for space-skipping occupancy textures.

BlockMinMaxBuilder
    GPU compute pass that reduces a 3D scalar volume into a coarse
    3D texture of (min, max) per block. Each output texel covers a
    ``ceil(vol_dim / block_dim)``-sized region of the volume.

build_block_alpha
    CPU helper that, given the min/max block texture and a TF LUT,
    produces a per-block ``r8unorm`` occupancy mask: 1 where the
    block's scalar range maps to non-zero TF opacity, 0 otherwise.
    Cheap enough (~32 K lookups) to run on the CPU every TF change.
"""

from __future__ import annotations

import numpy as np
import wgpu
import pygfx

from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object, update_resource

# ---------------------------------------------------------------------------
# GPU compute pass: volume → 32^3 (min, max) block texture
# ---------------------------------------------------------------------------

_BLOCK_MINMAX_WGSL = """
struct Params {
    vol_dims: vec4<u32>,       // (W, H, D, 0)
    block_dims: vec4<u32>,     // (BW, BH, BD, 0)
};

@group(0) @binding(0) var<uniform> u_params: Params;
@group(0) @binding(1) var t_vol: texture_3d<f32>;
@group(0) @binding(2) var t_out: texture_storage_3d<rg32float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(t_out);
    if (any(gid >= out_dims)) { return; }

    let vol_dims = u_params.vol_dims.xyz;
    let block = u_params.block_dims.xyz;

    // Voxel range covered by this output texel.
    let lo = gid * block;
    let hi = min(lo + block, vol_dims);

    var vmin: f32 = 1e30;
    var vmax: f32 = -1e30;

    for (var z = lo.z; z < hi.z; z = z + 1u) {
        for (var y = lo.y; y < hi.y; y = y + 1u) {
            for (var x = lo.x; x < hi.x; x = x + 1u) {
                let v = textureLoad(t_vol, vec3<i32>(vec3<u32>(x, y, z)), 0).r;
                vmin = min(vmin, v);
                vmax = max(vmax, v);
            }
        }
    }

    textureStore(t_out, vec3<i32>(gid), vec4<f32>(vmin, vmax, 0.0, 0.0));
}
"""


class BlockMinMaxBuilder:
    """Build a 3D (min, max) texture from a scalar volume.

    Parameters
    ----------
    device : wgpu.GPUDevice
    block_resolution : int
        Output texture resolution along each axis (default 32).
    """

    def __init__(self, device: wgpu.GPUDevice, block_resolution: int = 32):
        self.device = device
        self.block_resolution = block_resolution
        self._pipeline = None
        self._bgl = None

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return
        shader = self.device.create_shader_module(
            label="slicer_wgpu.BlockMinMax.shader", code=_BLOCK_MINMAX_WGSL)
        self._bgl = self.device.create_bind_group_layout(
            label="slicer_wgpu.BlockMinMax.bgl",
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
                 "texture": {"sample_type": wgpu.TextureSampleType.unfilterable_float,
                             "view_dimension": wgpu.TextureViewDimension.d3,
                             "multisampled": False}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
                 "storage_texture": {
                     "access": wgpu.StorageTextureAccess.write_only,
                     "format": wgpu.TextureFormat.rg32float,
                     "view_dimension": wgpu.TextureViewDimension.d3}},
            ])
        pl = self.device.create_pipeline_layout(
            label="slicer_wgpu.BlockMinMax.pl",
            bind_group_layouts=[self._bgl])
        self._pipeline = self.device.create_compute_pipeline(
            label="slicer_wgpu.BlockMinMax.pipeline",
            layout=pl,
            compute={"module": shader, "entry_point": "main"})

    def build(self, volume_pygfx_tex: pygfx.Texture) -> wgpu.GPUTexture:
        """Run the reduction and return a wgpu 3D texture (rg32float).

        The returned texture has dimensions ``(R, R, R)`` where R is
        ``self.block_resolution``, and each texel contains
        ``(scalar_min, scalar_max)`` for the corresponding block.
        """
        self._ensure_pipeline()

        # Ensure the pygfx volume texture has TEXTURE_BINDING usage.
        # pygfx's default is COPY_DST only; we need to read it in compute.
        # Follow the same pattern as shadows.py: set the flag before the
        # wgpu object is created, or accept the existing one if it
        # already has the right flags (the fragment shader path adds
        # TEXTURE_BINDING during its own ensure_wgpu_object call).
        if volume_pygfx_tex._wgpu_object is None:
            volume_pygfx_tex._wgpu_usage |= wgpu.TextureUsage.TEXTURE_BINDING
        ensure_wgpu_object(volume_pygfx_tex)
        update_resource(volume_pygfx_tex)
        vol_tex = volume_pygfx_tex._wgpu_object
        if not (vol_tex.usage & wgpu.TextureUsage.TEXTURE_BINDING):
            raise RuntimeError(
                "Volume texture missing TEXTURE_BINDING usage. "
                "Build skip textures before the first pygfx render, "
                "or rebuild after a module reload.")

        vol_w = vol_tex.width
        vol_h = vol_tex.height
        vol_d = vol_tex.depth_or_array_layers
        R = self.block_resolution

        # Block size in voxels along each axis.
        bw = max((vol_w + R - 1) // R, 1)
        bh = max((vol_h + R - 1) // R, 1)
        bd = max((vol_d + R - 1) // R, 1)

        # Params uniform.
        params = np.zeros(8, dtype=np.uint32)
        params[0], params[1], params[2] = vol_w, vol_h, vol_d
        params[4], params[5], params[6] = bw, bh, bd
        param_buf = self.device.create_buffer_with_data(
            label="slicer_wgpu.BlockMinMax.params",
            data=params.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM)

        # Output texture.
        out_tex = self.device.create_texture(
            label="slicer_wgpu.BlockMinMax.out",
            size=(R, R, R),
            dimension="3d",
            format=wgpu.TextureFormat.rg32float,
            usage=(wgpu.TextureUsage.STORAGE_BINDING
                   | wgpu.TextureUsage.TEXTURE_BINDING
                   | wgpu.TextureUsage.COPY_SRC))

        vol_view = vol_tex.create_view()
        out_view = out_tex.create_view()

        bg = self.device.create_bind_group(
            label="slicer_wgpu.BlockMinMax.bg",
            layout=self._bgl,
            entries=[
                {"binding": 0, "resource": {
                    "buffer": param_buf, "offset": 0, "size": param_buf.size}},
                {"binding": 1, "resource": vol_view},
                {"binding": 2, "resource": out_view},
            ])

        encoder = self.device.create_command_encoder(
            label="slicer_wgpu.BlockMinMax.enc")
        cpass = encoder.begin_compute_pass(
            label="slicer_wgpu.BlockMinMax.pass")
        cpass.set_pipeline(self._pipeline)
        cpass.set_bind_group(0, bg, [], 0, 0)
        wg = 4
        groups = (R + wg - 1) // wg
        cpass.dispatch_workgroups(groups, groups, groups)
        cpass.end()
        self.device.queue.submit([encoder.finish()])

        return out_tex


# ---------------------------------------------------------------------------
# CPU helper: (min, max) block texture + TF LUT → per-block alpha mask
# ---------------------------------------------------------------------------

def build_block_alpha(
    minmax_data: np.ndarray,
    lut_array: np.ndarray,
    clim: tuple[float, float],
    threshold: float = 0.001,
) -> np.ndarray:
    """Produce a uint8 occupancy mask from a min/max block texture.

    Parameters
    ----------
    minmax_data : ndarray, shape (D, H, W, 2), float32
        Per-block (min_scalar, max_scalar) values, read back from the
        GPU ``rg32float`` texture.
    lut_array : ndarray, shape (N, 4), float32
        The transfer-function LUT (RGBA).  Only the alpha column is used.
    clim : (float, float)
        The scalar range mapped to [0, 1] for the LUT index.
    threshold : float
        Blocks whose max TF alpha is below this are marked empty (0).

    Returns
    -------
    ndarray, shape (D, H, W), uint8
        1 where the block may be opaque, 0 where it can be skipped.
    """
    cmin, cmax = float(clim[0]), float(clim[1])
    crange = max(cmax - cmin, 1e-12)
    N = lut_array.shape[0]
    alpha = lut_array[:, 3].astype(np.float32)  # just the alpha column

    block_min = minmax_data[..., 0]
    block_max = minmax_data[..., 1]

    # Map block scalar range to LUT index range.
    idx_lo = np.clip((block_min - cmin) / crange, 0.0, 1.0) * (N - 1)
    idx_hi = np.clip((block_max - cmin) / crange, 0.0, 1.0) * (N - 1)
    ilo = np.floor(idx_lo).astype(np.int32)
    ihi = np.ceil(idx_hi).astype(np.int32)

    # For each block, check if ANY LUT entry in [ilo, ihi] has alpha > threshold.
    out = np.zeros(block_min.shape, dtype=np.uint8)
    it = np.nditer(ilo, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        lo_i = int(ilo[idx])
        hi_i = int(ihi[idx])
        if hi_i >= lo_i and np.any(alpha[lo_i:hi_i + 1] > threshold):
            out[idx] = 1
        it.iternext()

    return out
