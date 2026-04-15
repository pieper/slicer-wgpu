"""VolumeRenderingDisplayer: observes vtkMRMLVolumeRenderingDisplayNode
and produces ImageField instances for the SceneRenderer.
"""

from __future__ import annotations

import numpy as np
import slicer
import vtk

from .base import Displayer
from ..fields.image import ImageField


def _vtk_to_numpy(m: vtk.vtkMatrix4x4) -> np.ndarray:
    return np.array(
        [[m.GetElement(i, j) for j in range(4)] for i in range(4)],
        dtype=np.float64,
    )


def _world_from_local_for_volume(volume_node) -> np.ndarray:
    """Concatenated parent-transform-to-world matrix for a volume node;
    identity if the volume has no parent transform."""
    t_node = volume_node.GetParentTransformNode()
    if t_node is None:
        return np.eye(4, dtype=np.float64)
    m = vtk.vtkMatrix4x4()
    t_node.GetMatrixTransformToWorld(m)
    return _vtk_to_numpy(m)


class VolumeRenderingDisplayer(Displayer):
    node_class = "vtkMRMLVolumeRenderingDisplayNode"

    def _extra_watch(self, node, tags) -> None:
        vp = node.GetVolumePropertyNode()
        if vp is not None:
            tags.append((vp, vp.AddObserver(
                vtk.vtkCommand.ModifiedEvent, self._handle_node_modified)))
            # The Volume Rendering module's Shift slider (and other live
            # TF edits) mutates the transfer functions with
            # IgnoreVolumePropertyChanges=true, so vtkMRMLVolumePropertyNode
            # does NOT re-fire ModifiedEvent during the drag; it fires
            # InteractionEvent instead (per ProcessMRMLEvents). Observe
            # those as well so our LUT stays in sync — matching what the
            # VTK vtkMRMLVolumeRenderingDisplayableManager does.
            for ev in (vtk.vtkCommand.InteractionEvent,
                       vtk.vtkCommand.EndInteractionEvent):
                tags.append((vp, vp.AddObserver(ev, self._handle_node_modified)))
        # vtkMRMLTransformableNode relays a TransformModifiedEvent on
        # the volume itself whenever any ancestor in its transform
        # chain changes (either a re-parent or a matrix mutation),
        # which is what we need to refresh the Field's world_from_local.
        vol = node.GetVolumeNode()
        if vol is not None:
            tags.append((vol, vol.AddObserver(
                slicer.vtkMRMLTransformableNode.TransformModifiedEvent,
                self._handle_transform_modified)))

    def _handle_transform_modified(self, caller, event) -> None:
        """Fast path: only the transform changed, so update
        world_from_local in place rather than rebuilding the whole
        Field (which would re-upload textures + LUTs every drag tick)."""
        nid = self._caller_to_nid.get(id(caller))
        if nid is None:
            return
        field = self.fields_by_nid.get(nid)
        if field is None:
            return
        try:
            M = _world_from_local_for_volume(caller)
            field.set_world_from_local(M)
        except Exception as e:
            print(f"VolumeRenderingDisplayer transform update: {e}")
            return
        self._on_field_modified(field)

    def _make_field(self, node):
        vol_node = node.GetVolumeNode()
        if vol_node is None:
            return None
        try:
            field = ImageField.from_volume_node(vol_node, node)
            field.set_world_from_local(_world_from_local_for_volume(vol_node))
            return field
        except Exception as e:
            print(f"VolumeRenderingDisplayer._make_field: {e}")
            return None

    def _update_field(self, node, field) -> bool:
        # Any TF / property change rebuilds the LUT; cleanest is to drop
        # the field and let _make_field rebuild it on next add. That
        # counts as a structural change so the SceneRenderer can decide.
        nid = node.GetID()
        del self.fields_by_nid[nid]
        new_field = self._make_field(node)
        if new_field is None:
            return True
        self.fields_by_nid[nid] = new_field
        return True
