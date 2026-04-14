"""FiducialDisplayer: observes vtkMRMLMarkupsFiducialNode and produces
FiducialField instances for the SceneRenderer.

One Field per markup node. Each Field carries all of that node's control
points as spheres. Per-control-point colour comes from the markup's
display colour (with selected/unselected variants honoured); per-point
radius is derived from the display node's glyph scale.

Pick + drag: when a sphere is dragged in the pygfx pane, the
SceneRenderer routes drag_update() back into FiducialField, which
updates its in-memory sphere centre. After the drag we push the change
back into MRML by calling SetNthControlPointPosition.
"""

from __future__ import annotations

import numpy as np
import vtk

from .base import Displayer
from ..fields.fiducial import FiducialField


# Default radius used when the display node's glyph scale can't be
# converted (RAS mm). Slicer's default glyph scale is in pixel screen
# units, which doesn't translate cleanly to a fixed-radius world sphere;
# we use a conservative default here and let the host adjust per-Field if
# they want a different look.
DEFAULT_FIDUCIAL_RADIUS_MM = 3.0


def _displayed_color(disp_node, selected: bool):
    """Per-control-point colour from the markups display node, with the
    "Selected" override honoured. Returns RGBA tuple."""
    if disp_node is None:
        return (0.95, 0.7, 0.2, 1.0)  # warm pushpin yellow as a fallback
    if selected:
        rgb = disp_node.GetSelectedColor()
    else:
        rgb = disp_node.GetColor()
    opacity = disp_node.GetOpacity() if hasattr(disp_node, "GetOpacity") else 1.0
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(opacity))


def _node_to_spheres(node, default_radius=DEFAULT_FIDUCIAL_RADIUS_MM):
    """Read all visible control points of a markups node into
    (centers Nx3, radii N, colors Nx4)."""
    disp = node.GetDisplayNode()
    n = node.GetNumberOfControlPoints()
    centers, radii, colors = [], [], []
    pos = [0.0, 0.0, 0.0]
    for i in range(n):
        if not node.GetNthControlPointVisibility(i):
            continue
        node.GetNthControlPointPosition(i, pos)
        centers.append(tuple(pos))
        radii.append(default_radius)
        sel = bool(node.GetNthControlPointSelected(i))
        colors.append(_displayed_color(disp, sel))
    if not centers:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32))
    return (np.asarray(centers, dtype=np.float32),
            np.asarray(radii, dtype=np.float32),
            np.asarray(colors, dtype=np.float32))


class FiducialDisplayer(Displayer):
    """Observes vtkMRMLMarkupsFiducialNode -> one FiducialField each.

    For other markups types (curves/planes/ROI) the same Displayer
    pattern applies; this class is fiducial-only because each shape kind
    needs its own field/shader. Subclass and override `_make_field` /
    `_update_field` for the others.
    """

    node_class = "vtkMRMLMarkupsFiducialNode"

    def __init__(self, *args, default_radius_mm=DEFAULT_FIDUCIAL_RADIUS_MM, **kwargs):
        self._default_radius = float(default_radius_mm)
        # nid -> {"node_id": ..., "default_radius": ...}
        self._meta: dict[str, dict] = {}
        super().__init__(*args, **kwargs)

    def _extra_watch(self, node, tags) -> None:
        disp = node.GetDisplayNode()
        if disp is not None:
            tags.append((disp, disp.AddObserver(
                vtk.vtkCommand.ModifiedEvent, self._handle_node_modified)))
        # Markup point modifications fire as PointModifiedEvent on the node
        # itself, which is already covered by the generic ModifiedEvent.

    def _make_field(self, node):
        radius = self._meta.get(node.GetID(), {}).get("default_radius", self._default_radius)
        centers, radii, colors = _node_to_spheres(node, default_radius=radius)
        field = FiducialField(centers=centers, radii=radii, colors=colors)
        # Stash a back-reference so drag_update can route changes to MRML.
        field.mrml_node_id = node.GetID()
        self._meta[node.GetID()] = {"default_radius": radius}
        return field

    def _update_field(self, node, field) -> bool:
        radius = self._meta.get(node.GetID(), {}).get("default_radius", self._default_radius)
        centers, radii, colors = _node_to_spheres(node, default_radius=radius)
        field.set_spheres(centers, radii, colors)
        # Number of spheres and visibility can change but we keep the
        # same FiducialField slot, so no structural change for the
        # SceneRenderer.
        return False

    # -------- Per-node sizing convenience --------

    def set_default_radius(self, node_id: str, radius_mm: float) -> None:
        """Override the default sphere radius for one node."""
        self._meta.setdefault(node_id, {})["default_radius"] = float(radius_mm)
        node = self.mrml_scene.GetNodeByID(node_id)
        if node is None or node_id not in self.fields_by_nid:
            return
        self._update_field(node, self.fields_by_nid[node_id])
        self._on_field_modified(self.fields_by_nid[node_id])

    # -------- Bidirectional drag: write back to MRML --------

    def commit_drag(self, field, sphere_idx: int) -> None:
        """Push a sphere-centre change in the FiducialField back into
        its source vtkMRMLMarkupsFiducialNode control point. Called by
        the SceneRendererBridge after a successful drag."""
        nid = getattr(field, "mrml_node_id", None)
        if nid is None:
            return
        node = self.mrml_scene.GetNodeByID(nid)
        if node is None:
            return
        center = field.get_center(sphere_idx).astype(float)
        # Suppress our own ModifiedEvent so we don't re-read what we
        # just wrote (would clobber radius/colour).
        try:
            node.RemoveObserver(self._node_observer_tags[nid][0][1])
        except Exception:
            pass
        try:
            node.SetNthControlPointPosition(sphere_idx, float(center[0]),
                                            float(center[1]), float(center[2]))
        finally:
            # Re-attach the observer
            tag = node.AddObserver(vtk.vtkCommand.ModifiedEvent, self._handle_node_modified)
            self._node_observer_tags[nid][0] = (node, tag)
            self._caller_to_nid[id(node)] = nid
