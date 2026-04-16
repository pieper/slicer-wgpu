"""Field ABC.

A Field is one sample-able contribution to the per-ray compositing loop in
the SceneRenderer's ray march. Each Field generates its own WGSL fragment
that gets unrolled into the loop body, plus its own slice of the material
uniform layout and any texture/sampler bindings it needs.

Concrete subclasses (so far):
- ImageField     : a 3D scalar volume + transfer function + gradient-opacity
                   LUT (slicer_wgpu.fields.image)
- FiducialField  : up to N procedural SDF spheres in world space, drawn with
                   plastic-Phong shading (slicer_wgpu.fields.fiducial)

Coordinate convention: every Field samples in world space (RAS millimetres
in the Slicer pipeline). The Field is responsible for transforming to its
own data space if needed.

Picking is opt-in. Fields that override `supports_picking = True` and
implement `pick()` / `drag_update()` participate in interactive editing
through the SceneRenderer's pointer event handlers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class PickHit:
    """Result of Field.pick(). Stores enough state for a follow-up drag.

    Attributes:
        field: the Field instance that was hit (back-reference for routing).
        item_index: per-Field opaque integer (e.g. sphere index in a
            FiducialField, voxel coord in an ImageField).
        world_pos: the world-space hit point (RAS mm).
        t: ray parameter at the hit (for occlusion comparisons across fields).
        extra: optional per-Field state needed during drag.
    """

    __slots__ = ("field", "item_index", "world_pos", "t", "extra")

    def __init__(self, field, item_index, world_pos, t, extra=None):
        self.field = field
        self.item_index = item_index
        self.world_pos = np.asarray(world_pos, dtype=np.float64)
        self.t = float(t)
        self.extra = extra if extra is not None else {}


class Field(ABC):
    """Abstract sample-able thing along a ray.

    Subclasses must provide WGSL generation + uniform/binding helpers
    (uniform_type / get_bindings / sampling_wgsl / tf_wgsl) plus CPU-side
    helpers (fill_uniforms, aabb, mtime). Picking is optional.

    Slot index: at WGSL generation time the SceneRenderer assigns each
    Field an integer slot within its Field type (e.g. ImageField slot 0,
    ImageField slot 1, FiducialField slot 0). The slot is woven into all
    generated identifiers so multiple instances of the same Field type
    don't collide.
    """

    # Override in subclasses to enable interactive picking/dragging.
    supports_picking = False

    # WGSL family name. The SceneRenderer uses this to assign slot indices
    # within a family and to namespace generated function and uniform names.
    field_kind: str = ""

    def __init__(self):
        self._mtime = 0
        # Optional TransformField that warps the sampling position of
        # this field. The SceneRenderer emits a
        # ``transform_point_<kind><slot>(wp)`` helper per field that
        # either returns ``wp`` (no transform) or
        # ``wp + displacement_grid<M>(wp)`` (transform attached). Every
        # subclass's sampling_wgsl is expected to call this helper at
        # the top of its sampling path.
        self.transform_field = None

    # ---- WGSL generation hooks ----

    @abstractmethod
    def uniform_type(self, slot_idx: int) -> dict:
        """Per-slot fields to merge into the SceneMaterial's uniform_type
        dict. Keys are flat field names (typically prefixed with the slot
        identifier, e.g. "img0_clim"); values are pygfx uniform-type
        strings such as "f4", "4xf4", "4x4xf4", "256*4xf4".
        """

    @abstractmethod
    def get_bindings(self, slot_idx: int) -> list:
        """Return a list of pygfx.renderers.wgpu.Binding objects for this
        field's slot (textures/samplers). May be empty for procedural
        fields. Bindings are appended after the standard u_stdinfo /
        u_wobject / u_material trio.
        """

    @abstractmethod
    def sampling_wgsl(self, slot_idx: int) -> str:
        """WGSL source defining `sample_field_<kind><slot>(world_pos)
        -> FieldSample` plus any helpers. The SceneRenderer's main fragment
        shader calls these in the per-step compositing loop.
        """

    @abstractmethod
    def tf_wgsl(self, slot_idx: int) -> str:
        """WGSL source defining `tf_field_<kind><slot>(s: FieldSample)
        -> vec4<f32>` returning (color.rgb in sRGB, alpha in [0,1]).
        """

    # ---- CPU-side state ----

    @abstractmethod
    def fill_uniforms(self, uniform_buffer, slot_idx: int) -> None:
        """Write this field's current state into the SceneMaterial's
        uniform buffer. Called whenever the field's mtime advances.
        """

    @abstractmethod
    def aabb(self) -> tuple[np.ndarray, np.ndarray] | None:
        """World-space AABB (min_xyz, max_xyz) for ray clipping.
        Return None if the field's extent is unbounded or unknown
        (renderer falls back to scene-wide bounds).
        """

    @property
    def mtime(self) -> int:
        """Monotonic counter. Bump (via touch()) whenever uniforms /
        textures need to be re-uploaded or the AABB changes."""
        return self._mtime

    def touch(self) -> None:
        self._mtime += 1

    # ---- Optional structural / shader-cache hint ----

    def shader_signature(self, slot_idx: int) -> str:
        """A string that uniquely identifies the WGSL this field will
        generate at this slot. Two Fields with identical signatures can
        share a compiled shader pipeline; the SceneRenderer uses this to
        decide when to recompile.
        """
        return f"{self.field_kind}@{slot_idx}"

    # ---- Picking (optional) ----

    def pick(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
             camera, viewport_size) -> PickHit | None:
        """Intersect the field's geometry against a world-space ray.
        Return a PickHit if the ray hits something interactive on this
        field, else None. Only called if supports_picking is True.
        """
        return None

    def drag_update(self, hit: PickHit, ray_origin: np.ndarray,
                    ray_dir: np.ndarray, camera, viewport_size) -> bool:
        """Update the field's state from a follow-up pointer position
        during a drag that originated from `hit`. Return True if the
        field changed and a redraw is needed.
        """
        return False
