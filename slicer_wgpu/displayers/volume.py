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

    def __init__(self, *args, transform_provider=None, **kwargs):
        # ``transform_provider`` is typically a TransformDisplayer. It
        # resolves a volume node to its TransformField (if a grid
        # transform is in the parent chain). Passed in rather than
        # imported to keep Displayer instantiation order explicit.
        self._transform_provider = transform_provider
        super().__init__(*args, **kwargs)

    def _attach_transform_field(self, field, volume_node) -> bool:
        """Point ``field.transform_field`` at the right TransformField
        for this volume's parent-transform chain. Returns True if the
        attached TransformField identity changed (which is a structural
        change for the SceneRenderer)."""
        new_tf = None
        if self._transform_provider is not None:
            try:
                new_tf = self._transform_provider.transform_field_for_volume(
                    volume_node)
            except Exception as e:
                print(f"VolumeRenderingDisplayer transform resolve: {e}")
                new_tf = None
        changed = id(getattr(field, "transform_field", None)) != id(new_tf)
        field.transform_field = new_tf
        return changed

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
            # InteractionEvent so our LUT stays in sync during the drag,
            # matching what the VTK DM does.
            tags.append((vp, vp.AddObserver(
                vtk.vtkCommand.InteractionEvent, self._handle_node_modified)))
            # Start/End Interaction bracket the drag. Forward them to
            # the host so the renderer can trade quality for frame rate
            # while the slider is moving and restore full quality when
            # the user lets go.
            tags.append((vp, vp.AddObserver(
                vtk.vtkCommand.StartInteractionEvent,
                lambda c, e: self._on_interaction_start())))
            tags.append((vp, vp.AddObserver(
                vtk.vtkCommand.EndInteractionEvent,
                self._handle_end_interaction)))
        # vtkMRMLTransformableNode relays a TransformModifiedEvent on
        # the volume itself whenever any ancestor in its transform
        # chain changes (either a re-parent or a matrix mutation),
        # which is what we need to refresh the Field's world_from_local.
        vol = node.GetVolumeNode()
        if vol is not None:
            tags.append((vol, vol.AddObserver(
                slicer.vtkMRMLTransformableNode.TransformModifiedEvent,
                self._handle_transform_modified)))

    def _handle_end_interaction(self, caller, event) -> None:
        # On release: refresh the Field at full quality, THEN tell the
        # host to restore pixel ratio / sample step. The TF fast path
        # is cheap (LUT-only), so re-running it here guarantees the
        # final still-frame reflects the last interaction state before
        # quality comes back up.
        nid = self._caller_to_nid.get(id(caller))
        if nid is not None:
            node = self.mrml_scene.GetNodeByID(nid)
            field = self.fields_by_nid.get(nid)
            if node is not None and field is not None:
                try:
                    self._update_field(node, field)
                    self._on_field_modified(field)
                except Exception as e:
                    print(f"VolumeRenderingDisplayer end-interaction "
                          f"refresh failed: {e}")
        self._on_interaction_end()

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
        # Re-parenting can move a grid transform in or out of the chain.
        # Detect that and escalate to a structural change so the renderer
        # rebuilds with / without the grid binding.
        structural = self._attach_transform_field(field, caller)
        if structural:
            self._on_structure_changed()
        else:
            self._on_field_modified(field)

    def _make_field(self, node):
        vol_node = node.GetVolumeNode()
        if vol_node is None:
            return None
        try:
            field = ImageField.from_volume_node(vol_node, node)
            field.set_world_from_local(_world_from_local_for_volume(vol_node))
            self._attach_transform_field(field, vol_node)
            return field
        except Exception as e:
            print(f"VolumeRenderingDisplayer._make_field: {e}")
            return None

    def _update_field(self, node, field) -> bool:
        # Fast path: a VolumePropertyNode Modified/InteractionEvent means
        # the TF (scalar opacity, colour, gradient opacity, shading
        # params) changed but the 3D volume data did not. Rewrite the
        # 1D LUT textures and scalar uniforms in place on the same
        # ImageField instance -- O(lut size), not O(voxels) -- and
        # return False so the SceneRenderer just re-fills uniforms and
        # redraws rather than reinstantiating the whole Field (which
        # would re-upload the volume texture every Shift-slider tick).
        vol_node = node.GetVolumeNode()
        if vol_node is not None:
            try:
                field.refresh_from_display_node(vol_node, node)
                # Display-node updates can't change the parent transform,
                # but an external re-parent may have happened between
                # ticks. Re-resolve the grid TF so we notice.
                structural = self._attach_transform_field(field, vol_node)
                return structural
            except Exception as e:
                print(f"VolumeRenderingDisplayer TF fast path failed, "
                      f"falling back to full rebuild: {e}")

        # Fallback: full rebuild. Happens when refresh raised (e.g.
        # stale pygfx texture reference after an install teardown) or
        # the volume node disappeared.
        nid = node.GetID()
        del self.fields_by_nid[nid]
        new_field = self._make_field(node)
        if new_field is None:
            return True
        self.fields_by_nid[nid] = new_field
        return True
