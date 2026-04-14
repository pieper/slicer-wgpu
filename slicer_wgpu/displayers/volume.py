"""VolumeRenderingDisplayer: observes vtkMRMLVolumeRenderingDisplayNode
and produces ImageField instances for the SceneRenderer.
"""

from __future__ import annotations

import vtk

from .base import Displayer
from ..fields.image import ImageField


class VolumeRenderingDisplayer(Displayer):
    node_class = "vtkMRMLVolumeRenderingDisplayNode"

    def _extra_watch(self, node, tags) -> None:
        vp = node.GetVolumePropertyNode()
        if vp is not None:
            tags.append((vp, vp.AddObserver(
                vtk.vtkCommand.ModifiedEvent, self._handle_node_modified)))

    def _make_field(self, node):
        vol_node = node.GetVolumeNode()
        if vol_node is None:
            return None
        try:
            return ImageField.from_volume_node(vol_node, node)
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
