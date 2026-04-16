"""TransformDisplayer: observes vtkMRMLGridTransformNode and maintains
one TransformField per MRML grid transform. The TransformField texture
is refreshed on Modified events so live edits to the displacement grid
propagate without a shader rebuild; replacing the displacement grid's
dimensions does trigger a rebuild because the pygfx Texture is recreated.

Unlike the other Displayers in this package, this one does NOT
contribute renderable Fields to the SceneRenderer. Its TransformFields
are *modifiers* attached to other Fields (ImageField etc.) via the
``transform_field`` attribute. The SceneRenderer discovers them by
walking ``field.transform_field`` and treats them uniformly. We still
subclass Displayer to reuse the scene-scan + node-observer plumbing,
but override ``fields()`` to return an empty iterable so the manager's
`_gather_fields()` ignores them.

Sister displayers (VolumeRenderingDisplayer, FiducialDisplayer) query
``transform_field_for_volume(node)`` to resolve a given receiver's
parent-transform chain to a TransformField (if any grid transform is
in the chain).
"""

from __future__ import annotations

import slicer
import vtk

from .base import Displayer
from ..fields.transform import TransformField


class TransformDisplayer(Displayer):
    node_class = "vtkMRMLGridTransformNode"

    def fields(self):
        # Modifiers, not renderable Fields.
        return ()

    def _extra_watch(self, node, tags) -> None:
        # The MRML node fires a plain ModifiedEvent on metadata churn,
        # but a vtkMRMLTransformNode uses a dedicated TransformModifiedEvent
        # when its inner transform is replaced (SetAndObserveTransformFromParent)
        # or mutated (displacement grid voxels change). Subscribe to that
        # so we rebuild the TransformField right after the user attaches
        # a real displacement grid. The plain ModifiedEvent observer
        # installed by the base class still catches node re-parenting.
        tags.append((node, node.AddObserver(
            slicer.vtkMRMLTransformNode.TransformModifiedEvent,
            self._handle_transform_core_modified)))

    def _handle_transform_core_modified(self, caller, event) -> None:
        nid = self._caller_to_nid.get(id(caller))
        if nid is None:
            return
        node = self.mrml_scene.GetNodeByID(nid)
        if node is None:
            return
        if nid not in self.fields_by_nid:
            # First successful populate: go through _handle_node_added
            # so _watch_node bookkeeping stays consistent.
            self._handle_node_added(node)
            return
        field = self.fields_by_nid[nid]
        try:
            structural = self._update_field(node, field)
        except Exception as e:
            print(f"TransformDisplayer transform-core modified: {e}")
            return
        if structural:
            self._on_structure_changed()
        else:
            self._on_field_modified(field)

    def _make_field(self, node):
        try:
            tf = TransformField.from_grid_transform_node(node)
            return tf
        except Exception as e:
            print(f"TransformDisplayer._make_field({node.GetName()}): {e}")
            return None

    def _update_field(self, node, field) -> bool:
        """Re-read the displacement grid from the node. If the new grid
        has the same dimensions we just update the texture contents
        (cheap; touch() signals the renderer to re-upload); if the dims
        changed we return structural=True so the manager rebuilds the
        shader against the new TransformField/texture pair.
        """
        try:
            new_tf = TransformField.from_grid_transform_node(node)
        except Exception as e:
            print(f"TransformDisplayer._update_field: {e}")
            return False

        old_tex = field._tex
        new_tex = new_tf._tex
        dims_match = (old_tex is not None and new_tex is not None
                      and old_tex.data.shape == new_tex.data.shape)
        if dims_match:
            field.set_displacement(
                new_tex.data, patient_to_texture=new_tf.patient_to_texture)
            field.bounds_min = new_tf.bounds_min
            field.bounds_max = new_tf.bounds_max
            return False
        # Replace the whole instance: tell the host this is structural so
        # the SceneRenderer rebuilds bindings against the new Texture.
        nid = node.GetID()
        self.fields_by_nid[nid] = new_tf
        return True

    # -------- Public lookup for sister displayers ----------

    def transform_field_for_volume(self, volume_node):
        """Walk ``volume_node``'s parent-transform chain and return the
        first TransformField that corresponds to a grid transform in the
        chain, or None if no grid transform is upstream. If ``volume_node``
        is None, returns None.
        """
        if volume_node is None:
            return None
        t = volume_node.GetParentTransformNode()
        while t is not None:
            if t.IsA("vtkMRMLGridTransformNode"):
                return self.fields_by_nid.get(t.GetID())
            t = t.GetParentTransformNode()
        return None
