"""TransformField: a 3D displacement vector field that warps any Field's
sampling position during the ray march.

Unlike ImageField / FiducialField, a TransformField is a *modifier*, not
a compositor: it does not contribute color or opacity. Instead, another
Field holds a reference to a TransformField and the renderer inlines a
``transform_point_<kind><slot>(wp) = wp + displacement_grid<M>(wp)``
helper that is called at the top of the receiver's sampling path. This
follows the STEP ``transformPoint`` pattern from
``step/fields/field.js`` -- the receiver lookups its own data at the
warped point, so a grid transform deforms the apparent shape of the
volume, the position of fiducials, etc.

The vector field is a 3D RGBA32F texture of surface-to-warp-point
displacements in world (RAS) mm. World->texture uses the standard
``patient_to_texture`` 4x4. Outside the [0,1]^3 box the warp returns the
zero vector (identity) so only the region covered by the MRML grid is
actually deformed.
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


def _patient_to_texture_from_grid(grid_image_data) -> np.ndarray:
    """4x4 mapping world RAS -> tex coords [0,1]^3 for a vtkImageData whose
    IJK basis carries the RAS->IJK direction matrix via GetDirectionMatrix
    + GetOrigin + GetSpacing (what vtkGridTransform.GetDisplacementGrid
    returns). Mirrors the math used for volume textures but inlined here
    because grid transforms carry their own origin/spacing.
    """
    origin = np.asarray(grid_image_data.GetOrigin(), dtype=np.float64)
    spacing = np.asarray(grid_image_data.GetSpacing(), dtype=np.float64)
    dim = grid_image_data.GetDimensions()
    # vtkImageData.GetDirectionMatrix() is a nullary getter returning a
    # vtkMatrix3x3; older VTK builds don't have it and default to identity.
    if hasattr(grid_image_data, "GetDirectionMatrix"):
        direction = grid_image_data.GetDirectionMatrix()
        D = np.array([[direction.GetElement(i, j) for j in range(3)]
                      for i in range(3)], dtype=np.float64)
    else:
        D = np.eye(3, dtype=np.float64)
    # IJK->RAS = T(origin) * R(direction) * S(spacing)
    ijk_to_ras = np.eye(4, dtype=np.float64)
    ijk_to_ras[:3, :3] = D @ np.diag(spacing)
    ijk_to_ras[:3, 3] = origin
    ras_to_ijk = np.linalg.inv(ijk_to_ras)
    ijk_to_tex = np.eye(4, dtype=np.float64)
    for axis in range(3):
        ijk_to_tex[axis, axis] = 1.0 / dim[axis]
        ijk_to_tex[axis, 3] = 0.5 / dim[axis]
    return (ijk_to_tex @ ras_to_ijk).astype(np.float32)


class TransformField(Field):
    """3D displacement vector field. Non-compositing: referenced by other
    Fields via their ``transform_field`` attribute. The SceneRenderer
    deduplicates these across all fields, allocates slot indices
    ``grid0, grid1, ...``, emits their WGSL once, and inlines a
    ``transform_point_<kind><slot>`` helper per receiver that adds the
    sampled displacement to the incoming world position.
    """

    field_kind = "grid"

    def __init__(self,
                 displacement_array: np.ndarray | None = None,
                 patient_to_texture: np.ndarray | None = None,
                 bounds_min=(-100.0, -100.0, -100.0),
                 bounds_max=(100.0, 100.0, 100.0),
                 gain: float = 1.0):
        super().__init__()
        # ``displacement_array`` is (D, H, W, 4) float32 -- rgba, with rgb
        # = (dx, dy, dz) in RAS mm and a=unused (padded for GPU alignment).
        # Accept None to build an empty placeholder useful in tests.
        self._tex = (
            pygfx.Texture(
                displacement_array.astype(np.float32, copy=False), dim=3)
            if displacement_array is not None else None
        )
        self.patient_to_texture = (
            np.eye(4, dtype=np.float32) if patient_to_texture is None
            else np.asarray(patient_to_texture, dtype=np.float32)
        )
        self.bounds_min = tuple(float(x) for x in bounds_min)
        self.bounds_max = tuple(float(x) for x in bounds_max)
        self.gain = float(gain)

    # ---------- Construct from a vtkMRMLGridTransformNode ----------

    @classmethod
    def from_grid_transform_node(cls, grid_node) -> "TransformField":
        """Build a TransformField from a vtkMRMLGridTransformNode. The
        displacement grid is extracted via
        ``GetTransformFromParent()->GetDisplacementGrid()`` (RAS frame,
        one vec3 of mm displacement per voxel).
        """
        import vtk.util.numpy_support as vnp
        # GridTransform stores the displacement in its "from-parent" side,
        # which Slicer exposes identity-wrapped around the vtkGridTransform
        # core. Fall back to "to-parent" if the from side is empty.
        core = grid_node.GetTransformFromParent()
        displacement_grid = None
        if hasattr(core, "GetDisplacementGrid"):
            displacement_grid = core.GetDisplacementGrid()
        if displacement_grid is None:
            core = grid_node.GetTransformToParent()
            if hasattr(core, "GetDisplacementGrid"):
                displacement_grid = core.GetDisplacementGrid()
        if displacement_grid is None:
            raise ValueError(
                f"{grid_node.GetName()}: no displacement grid on either "
                "transform direction")

        # Reject MRML's default placeholder (a 1-voxel grid that the
        # vtkMRMLGridTransformNode wrapper allocates before the user has
        # attached a real displacement). Without this guard the first
        # NodeAddedEvent yields a (1,1,1) TransformField that then
        # survives as if real; the caller is expected to retry on the
        # next Modified once the grid has actual voxels.
        if displacement_grid.GetNumberOfPoints() <= 1:
            raise ValueError(
                f"{grid_node.GetName()}: displacement grid is a 1-voxel "
                "placeholder; not yet populated")

        dims = displacement_grid.GetDimensions()          # (nx, ny, nz)
        pd = displacement_grid.GetPointData()
        vec_arr = pd.GetScalars() if pd is not None else None
        if vec_arr is None:
            # Try to grab the first vector array on PointData
            pd = displacement_grid.GetPointData()
            if pd is not None and pd.GetNumberOfArrays() > 0:
                vec_arr = pd.GetArray(0)
        if vec_arr is None:
            raise ValueError(
                f"{grid_node.GetName()}: displacement grid has no "
                "vectors attached")

        raw = vnp.vtk_to_numpy(vec_arr).astype(np.float32, copy=False)
        # raw is (N, n_components). N = nx*ny*nz. n_components is usually 3.
        n_comp = raw.shape[1] if raw.ndim == 2 else 1
        n_vox = dims[0] * dims[1] * dims[2]
        raw = raw.reshape(n_vox, n_comp)
        rgba = np.zeros((n_vox, 4), dtype=np.float32)
        rgba[:, 0:min(n_comp, 3)] = raw[:, 0:min(n_comp, 3)]
        # vtkImageData layout: IJK with I fastest. Texture dim order is
        # (W, H, D) along (I, J, K). pygfx.Texture with dim=3 expects
        # (D, H, W, C) so we reshape accordingly.
        rgba = rgba.reshape(dims[2], dims[1], dims[0], 4)

        # Bounds from IJK -> RAS corners (for scene AABB participation).
        p2t = _patient_to_texture_from_grid(displacement_grid)
        corners_tex = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
             [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float64)
        tex_to_pat = np.linalg.inv(p2t.astype(np.float64))
        world_corners = np.array([
            (tex_to_pat @ np.array([*c, 1.0]))[:3] for c in corners_tex])
        bmin = tuple(float(x) for x in world_corners.min(axis=0))
        bmax = tuple(float(x) for x in world_corners.max(axis=0))

        return cls(
            displacement_array=rgba,
            patient_to_texture=p2t,
            bounds_min=bmin,
            bounds_max=bmax,
        )

    # ---------- Field protocol ----------

    def uniform_type(self, slot_idx: int) -> dict:
        p = f"grid{slot_idx}"
        return {
            f"{p}_patient_to_texture": "4x4xf4",
            f"{p}_gain":                "f4",
            # Pad to next 16 byte boundary.
            f"{p}__pad0": "f4",
            f"{p}__pad1": "f4",
            f"{p}__pad2": "f4",
        }

    def get_bindings(self, slot_idx: int) -> list:
        out = []
        if self._tex is not None:
            out.append(Binding(
                f"s_grid{slot_idx}", "sampler/filtering",
                GfxSampler("linear", "clamp"), FRAGMENT_ONLY))
            out.append(Binding(
                f"t_grid{slot_idx}", "texture/auto",
                GfxTextureView(self._tex), FRAGMENT_ONLY))
        return out

    def sampling_wgsl(self, slot_idx: int) -> str:
        i = slot_idx
        return f"""
fn displacement_grid{i}(wp: vec3<f32>) -> vec3<f32> {{
    let tex4 = u_material.grid{i}_patient_to_texture * vec4<f32>(wp, 1.0);
    let tex = tex4.xyz;
    if (any(tex < vec3<f32>(0.0)) || any(tex > vec3<f32>(1.0))) {{
        return vec3<f32>(0.0);
    }}
    let d = textureSampleLevel(t_grid{i}, s_grid{i}, tex, 0.0).xyz;
    return u_material.grid{i}_gain * d;
}}
"""

    def tf_wgsl(self, slot_idx: int) -> str:
        # A transform-only field emits no compositing TF.
        return ""

    # ---------- CPU-side ----------

    def fill_uniforms(self, uniform_buffer, slot_idx: int) -> None:
        p = f"grid{slot_idx}"
        uniform_buffer.data[f"{p}_patient_to_texture"] = (
            np.asarray(self.patient_to_texture, dtype=np.float32).T)
        uniform_buffer.data[f"{p}_gain"] = np.float32(self.gain)

    def aabb(self):
        return (np.asarray(self.bounds_min, dtype=np.float64),
                np.asarray(self.bounds_max, dtype=np.float64))

    def set_displacement(self, displacement_array: np.ndarray,
                         patient_to_texture: np.ndarray | None = None):
        """Replace the displacement texture contents. Bumps mtime so the
        renderer re-uploads the GPU texture on the next draw. If the
        world->texture mapping also changed (new dims / origin), pass
        the new 4x4 in ``patient_to_texture``.
        """
        if self._tex is None or self._tex.data.shape != displacement_array.shape:
            self._tex = pygfx.Texture(
                displacement_array.astype(np.float32, copy=False), dim=3)
        else:
            self._tex.set_data(displacement_array.astype(np.float32, copy=False))
        if patient_to_texture is not None:
            self.patient_to_texture = np.asarray(
                patient_to_texture, dtype=np.float32)
        self.touch()

    def set_gain(self, gain: float):
        self.gain = float(gain)
        self.touch()
