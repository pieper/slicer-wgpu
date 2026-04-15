"""Displayer ABC.

A Displayer is to a Field what a DisplayableManager is to a vtkActor: it
observes a category of MRML nodes (one node class per Displayer) and
maintains the corresponding Fields on a SceneRenderer.

When a node appears, the Displayer builds a Field; when the node is
modified, it updates the Field's state (or rebuilds it); when the node
disappears, it drops the Field. The host (SceneRendererBridge) is
notified after every batch of changes via a callback so it can decide
whether the SceneRenderer needs a structural rebuild (field count or
kind changed) or just a uniform refresh.

Subclasses implement four hooks:
    node_class      class-level: the MRML class to observe
    _make_field     -> build a new Field for a freshly-discovered node
    _update_field   -> push current MRML state into an existing Field
    _extra_watch    optional: register additional VTK observers
                    (e.g. on display nodes)
"""

from __future__ import annotations

from typing import Callable, Iterable

import slicer
import vtk
import weakref

from ..fields.base import Field


class Displayer:
    """Mirror MRML nodes of one class onto Fields on a SceneRenderer."""

    node_class: str = ""  # subclass sets

    def __init__(self, mrml_scene=None,
                 on_structure_changed: Callable[[], None] | None = None,
                 on_field_modified: Callable[[Field], None] | None = None,
                 on_interaction_start: Callable[[], None] | None = None,
                 on_interaction_end: Callable[[], None] | None = None):
        self.mrml_scene = mrml_scene if mrml_scene is not None else slicer.mrmlScene
        # nid -> Field
        self.fields_by_nid: dict[str, Field] = {}
        self._scene_observer_tags: list[int] = []
        # nid -> [(vtk_object, observer_tag), ...]
        self._node_observer_tags: dict[str, list] = {}
        # id(vtk_obj) -> nid (O(1) reverse lookup)
        self._caller_to_nid: dict[int, str] = {}

        self._on_structure_changed = on_structure_changed or (lambda: None)
        self._on_field_modified = on_field_modified or (lambda f: None)
        # Optional: host-level "user is actively dragging" hooks so the
        # renderer can drop pixel ratio / raise sample step mid-drag and
        # restore at rest. Modelled after VTK's DesiredUpdateRate path
        # but kept explicit instead of auto-tuned.
        self._on_interaction_start = on_interaction_start or (lambda: None)
        self._on_interaction_end = on_interaction_end or (lambda: None)

        self_ref = weakref.ref(self)

        @vtk.calldata_type(vtk.VTK_OBJECT)
        def _node_added(caller, event, calldata):
            s = self_ref()
            if s is not None:
                s._handle_scene_node_added(caller, event, calldata)

        @vtk.calldata_type(vtk.VTK_OBJECT)
        def _node_removed(caller, event, calldata):
            s = self_ref()
            if s is not None:
                s._handle_scene_node_removed(caller, event, calldata)

        self._node_added_cb = _node_added
        self._node_removed_cb = _node_removed
        self._scene_observer_tags.append(
            self.mrml_scene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, _node_added)
        )
        self._scene_observer_tags.append(
            self.mrml_scene.AddObserver(slicer.vtkMRMLScene.NodeRemovedEvent, _node_removed)
        )

        self._scan_scene()

    # -------- Scene scan + observers --------

    def _scan_scene(self) -> None:
        n = self.mrml_scene.GetNumberOfNodesByClass(self.node_class)
        for i in range(n):
            node = self.mrml_scene.GetNthNodeByClass(i, self.node_class)
            if node is not None:
                self._handle_node_added(node)

    def cleanup(self) -> None:
        for tag in self._scene_observer_tags:
            try:
                self.mrml_scene.RemoveObserver(tag)
            except Exception:
                pass
        self._scene_observer_tags = []
        for nid, pairs in list(self._node_observer_tags.items()):
            for obj, tag in pairs:
                try:
                    obj.RemoveObserver(tag)
                except Exception:
                    pass
        self._node_observer_tags.clear()
        self._caller_to_nid.clear()
        had_fields = bool(self.fields_by_nid)
        self.fields_by_nid.clear()
        if had_fields:
            self._on_structure_changed()

    def _handle_scene_node_added(self, caller, event, node) -> None:
        if node is None or not node.IsA(self.node_class):
            return
        self._handle_node_added(node)

    def _handle_scene_node_removed(self, caller, event, node) -> None:
        if node is None:
            return
        nid = node.GetID()
        if nid in self.fields_by_nid or nid in self._node_observer_tags:
            self._handle_node_removed(nid)

    def _handle_node_added(self, node) -> None:
        nid = node.GetID()
        if nid not in self._node_observer_tags:
            self._watch_node(node, nid)
        if nid in self.fields_by_nid:
            return
        try:
            field = self._make_field(node)
        except Exception as e:
            print(f"{self.__class__.__name__}._make_field failed for {nid}: {e}")
            return
        if field is None:
            return  # Not ready yet (e.g. display node not attached); will re-try
        self.fields_by_nid[nid] = field
        self._on_structure_changed()

    def _handle_node_removed(self, nid: str) -> None:
        for obj, tag in self._node_observer_tags.pop(nid, []):
            try:
                obj.RemoveObserver(tag)
            except Exception:
                pass
            self._caller_to_nid.pop(id(obj), None)
        if nid in self.fields_by_nid:
            del self.fields_by_nid[nid]
            self._on_structure_changed()

    def _watch_node(self, node, nid: str) -> None:
        tags = [(node, node.AddObserver(vtk.vtkCommand.ModifiedEvent, self._handle_node_modified))]
        self._extra_watch(node, tags)
        self._node_observer_tags[nid] = tags
        for obj, _ in tags:
            self._caller_to_nid[id(obj)] = nid

    def _extra_watch(self, node, tags: list) -> None:
        """Subclasses may append additional (vtk_object, observer_tag) pairs."""

    def _handle_node_modified(self, caller, event) -> None:
        nid = self._caller_to_nid.get(id(caller))
        if nid is None:
            return
        node = self.mrml_scene.GetNodeByID(nid)
        if node is None:
            return
        if nid not in self.fields_by_nid:
            self._handle_node_added(node)
            return
        field = self.fields_by_nid[nid]
        try:
            structural = self._update_field(node, field)
        except Exception as e:
            print(f"{self.__class__.__name__}._update_field failed for {nid}: {e}")
            return
        if structural:
            self._on_structure_changed()
        else:
            self._on_field_modified(field)

    # -------- Subclass hooks --------

    def _make_field(self, node) -> Field | None:
        """Construct a brand-new Field for this MRML node. Return None if
        the node isn't ready (we'll get another shot on the next Modified)."""
        raise NotImplementedError

    def _update_field(self, node, field: Field) -> bool:
        """Push current MRML state into the existing Field. Return True if
        the change is *structural* (e.g. a node visibility flip that
        requires the SceneRenderer to recompile its shader / rebind);
        False if it's just a uniform/data update.
        """
        return False

    # -------- Convenience --------

    def fields(self) -> Iterable[Field]:
        return self.fields_by_nid.values()
