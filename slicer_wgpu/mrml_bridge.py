"""
slicer_wgpu.mrml_bridge -- mirror a Slicer MRML scene into a pygfx view.

Provides:
    PygfxView                            -- widget + renderer + scene + camera
    DisplayableManager                   -- base class for MRML -> scene-graph bridges
    ModelDisplayableManager              -- vtkMRMLModelNode  (mesh)
    SegmentationDisplayableManager       -- vtkMRMLSegmentationNode (mesh)
    SceneRendererManager                 -- owns the SceneRenderer + per-Field
                                            Displayers (vtkMRMLVolumeRenderingDisplayNode,
                                            vtkMRMLMarkupsFiducialNode, ...).
                                            Routes pointer events for pick-and-drag.
    CameraDisplayableManager             -- vtkMRMLCameraNode (bidirectional)
    ViewDisplayableManager               -- vtkMRMLViewNode (background, cube)
    DualView                             -- custom layout + managers

Usage:
    from slicer_wgpu import mrml_bridge
    dv = mrml_bridge.install()
    # ... load data; both VTK and pygfx views mirror the scene ...
    # mrml_bridge.uninstall() to remove
"""

import numpy as np
import weakref
import slicer
import vtk
import vtk.util.numpy_support as vnp
import qt
import pygfx
import pylinalg as la
from rendercanvas.qt import QRenderWidget

from .scene_renderer import SceneRenderer
from .displayers import VolumeRenderingDisplayer, FiducialDisplayer


# ------------------------------------------------------------------------
# PygfxView: widget + pygfx renderer + scene + camera + lights
# ------------------------------------------------------------------------

class PygfxView:
    """A pygfx render widget with scene, camera, orbit controller, and lights."""

    def __init__(self, parent=None):
        self.widget = QRenderWidget(parent) if parent is not None else QRenderWidget()
        # Match qMRMLThreeDView's size constraints so the controller (blue) bar
        # above us keeps its sizeHint height and we absorb all extra vertical
        # space on resize.
        self.widget.setMinimumSize(0, 0)
        _sp = qt.QSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Expanding)
        self.widget.setSizePolicy(_sp)

        self.renderer = pygfx.renderers.WgpuRenderer(self.widget)
        self.scene = pygfx.Scene()

        # Lights
        self.scene.add(pygfx.AmbientLight("#ffffff", 0.4))
        self._dir_light = pygfx.DirectionalLight("#ffffff", 2.0)
        self._dir_light.local.position = (200, 300, 400)
        self.scene.add(self._dir_light)

        # Camera + controller
        self.camera = pygfx.PerspectiveCamera(50, 4 / 3, depth_range=(0.1, 5000))
        self.camera.local.position = (0, 500, 0)
        self.controller = pygfx.OrbitController(self.camera, register_events=self.renderer)

        self._closed = False
        self._dirty = True

        try:
            self.renderer.add_event_handler(self._on_controller_event, "pointer_move", "wheel")
        except Exception:
            pass

        self.widget.request_draw(self._animate)

    def _on_controller_event(self, ev):
        self.request_redraw()

    def _animate(self):
        if self._closed:
            return
        try:
            self.renderer.render(self.scene, self.camera)
        except Exception as e:
            print(f"PygfxView._animate render error: {e}")

    def request_redraw(self):
        if self._closed:
            return
        self._dirty = True
        try:
            self.widget.request_draw()
        except Exception:
            pass

    def close(self):
        self._closed = True
        try:
            self.widget.draw_frame = None
        except Exception:
            pass

    def reset_camera(self):
        has_content = any(
            not isinstance(c, (pygfx.AmbientLight, pygfx.DirectionalLight, pygfx.PointLight, pygfx.Background))
            for c in self.scene.children
        )
        if has_content:
            try:
                self.camera.show_object(self.scene, view_dir=(0, -500, 0), up=(0, 0, 1))
            except Exception as e:
                print(f"reset_camera: {e}")
        self.request_redraw()

    def add(self, obj):
        self.scene.add(obj)
        self.request_redraw()

    def remove(self, obj):
        try:
            self.scene.remove(obj)
        except Exception:
            pass
        self.request_redraw()


# ------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------

def vtk_matrix_to_numpy(vtk_mat):
    m = slicer.util.arrayFromVTKMatrix(vtk_mat)
    return m.astype(np.float32, copy=False)


_GEOMETRY_CACHE = {}  # id(polydata) -> (mtime, pygfx.Geometry)


def polydata_to_pygfx_geometry(poly_data):
    """Extract (indices, positions, normals) numpy arrays from vtkPolyData.
    Returns a pygfx.Geometry or None if polydata is empty. Cached by mtime."""
    if poly_data is None or poly_data.GetNumberOfPoints() == 0:
        return None

    key = id(poly_data)
    mtime = poly_data.GetMTime()
    cached = _GEOMETRY_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    polys = poly_data.GetPolys()
    pd = poly_data
    needs_triangulate = False
    if polys is not None and polys.GetNumberOfCells() > 0:
        data_len = polys.GetData().GetNumberOfTuples()
        expected_tri_len = polys.GetNumberOfCells() * 4
        if data_len != expected_tri_len:
            needs_triangulate = True
    else:
        return None

    if needs_triangulate:
        tf = vtk.vtkTriangleFilter()
        tf.SetInputData(poly_data)
        tf.Update()
        pd = tf.GetOutput()

    if pd.GetPointData().GetNormals() is None:
        nf = vtk.vtkPolyDataNormals()
        nf.SetInputData(pd)
        nf.SplittingOff()
        nf.ConsistencyOn()
        nf.Update()
        pd = nf.GetOutput()

    points_arr = vnp.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float32, copy=False)
    if points_arr.size == 0:
        return None

    polys2 = pd.GetPolys()
    if polys2 is None or polys2.GetNumberOfCells() == 0:
        return None
    idx_raw = vnp.vtk_to_numpy(polys2.GetData()).astype(np.uint32, copy=False)
    idx_raw = np.delete(idx_raw, slice(None, None, 4))
    indices = np.ascontiguousarray(idx_raw.reshape(-1, 3))

    norm_vtk = pd.GetPointData().GetNormals()
    normals = vnp.vtk_to_numpy(norm_vtk).astype(np.float32, copy=False) if norm_vtk is not None else None

    geom = pygfx.Geometry(indices=indices, positions=points_arr, normals=normals)
    _GEOMETRY_CACHE[key] = (mtime, geom)
    if len(_GEOMETRY_CACHE) > 128:
        _GEOMETRY_CACHE.pop(next(iter(_GEOMETRY_CACHE)))
    return geom


# ------------------------------------------------------------------------
# DisplayableManager: base class
# ------------------------------------------------------------------------

class DisplayableManager:
    """Base class that mirrors MRML nodes of one class into a PygfxView."""

    node_class = None  # subclass sets

    def __init__(self, view, mrml_scene=None):
        self.view = view
        self.mrml_scene = mrml_scene if mrml_scene is not None else slicer.mrmlScene
        self.entries = {}
        self._scene_observer_tags = []
        self._node_observer_tags = {}
        self._caller_to_nid = {}

        self_ref = weakref.ref(self)

        @vtk.calldata_type(vtk.VTK_OBJECT)
        def _node_added_wrapper(caller, event, calldata):
            s = self_ref()
            if s is not None:
                s._on_scene_node_added(caller, event, calldata)

        @vtk.calldata_type(vtk.VTK_OBJECT)
        def _node_removed_wrapper(caller, event, calldata):
            s = self_ref()
            if s is not None:
                s._on_scene_node_removed(caller, event, calldata)

        self._node_added_wrapper = _node_added_wrapper
        self._node_removed_wrapper = _node_removed_wrapper
        self._scene_observer_tags.append(
            self.mrml_scene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, _node_added_wrapper)
        )
        self._scene_observer_tags.append(
            self.mrml_scene.AddObserver(slicer.vtkMRMLScene.NodeRemovedEvent, _node_removed_wrapper)
        )

        self._scan_scene()

    def _scan_scene(self):
        n = self.mrml_scene.GetNumberOfNodesByClass(self.node_class)
        for i in range(n):
            node = self.mrml_scene.GetNthNodeByClass(i, self.node_class)
            if node is not None:
                self._on_node_added(node)

    def cleanup(self):
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
        for nid in list(self.entries.keys()):
            self._remove_entry(nid)

    def _on_scene_node_added(self, caller, event, calldata):
        node = calldata
        if node is None or not node.IsA(self.node_class):
            return
        self._on_node_added(node)

    def _on_scene_node_removed(self, caller, event, calldata):
        node = calldata
        if node is None:
            return
        nid = node.GetID()
        if nid in self.entries or nid in self._node_observer_tags:
            self._remove_entry(nid)

    def _on_node_added(self, node):
        nid = node.GetID()
        if nid not in self._node_observer_tags:
            self._watch_node(node, nid)
        self._try_add(node, nid)

    def _try_add(self, node, nid):
        if nid in self.entries:
            return
        try:
            entry = self._add_node(node)
        except Exception as e:
            print(f"{self.__class__.__name__}._add_node failed for {nid}: {e}")
            return
        if entry is None:
            return
        self.entries[nid] = entry
        root = entry.get("root")
        if root is not None:
            self.view.add(root)
        else:
            self.view.request_redraw()

    def _watch_node(self, node, nid):
        tags = [(node, node.AddObserver(vtk.vtkCommand.ModifiedEvent, self._on_node_modified))]
        self._extra_watch(node, tags)
        self._node_observer_tags[nid] = tags
        for obj, _ in tags:
            self._caller_to_nid[id(obj)] = nid

    def _extra_watch(self, node, tags):
        pass

    def _on_node_modified(self, caller, event):
        nid = self._caller_to_nid.get(id(caller))
        if nid is None:
            return
        node = self.mrml_scene.GetNodeByID(nid)
        if node is None:
            return
        if nid not in self.entries:
            self._try_add(node, nid)
            return
        try:
            self._update_node(node, self.entries[nid])
        except Exception as e:
            print(f"{self.__class__.__name__}._update_node failed for {nid}: {e}")
        self.view.request_redraw()

    def _remove_entry(self, nid):
        entry = self.entries.pop(nid, None)
        if entry is not None:
            root = entry.get("root")
            if root is not None:
                self.view.remove(root)
            try:
                self._remove_node(entry)
            except Exception:
                pass
        for obj, tag in self._node_observer_tags.pop(nid, []):
            try:
                obj.RemoveObserver(tag)
            except Exception:
                pass
            self._caller_to_nid.pop(id(obj), None)

    def _add_node(self, node):
        raise NotImplementedError

    def _update_node(self, node, entry):
        pass

    def _remove_node(self, entry):
        pass


# ------------------------------------------------------------------------
# ModelDisplayableManager
# ------------------------------------------------------------------------

class ModelDisplayableManager(DisplayableManager):
    node_class = "vtkMRMLModelNode"

    def _get_display(self, node):
        return node.GetDisplayNode()

    def _extra_watch(self, node, tags):
        disp = self._get_display(node)
        if disp is not None:
            tags.append((disp, disp.AddObserver(vtk.vtkCommand.ModifiedEvent, self._on_node_modified)))

    def _add_node(self, node):
        pd = node.GetPolyData()
        geom = polydata_to_pygfx_geometry(pd)
        if geom is None:
            return None
        disp = self._get_display(node)
        color = [0.7, 0.7, 0.7]
        opacity = 1.0
        visible = True
        if disp is not None:
            color = disp.GetColor()
            opacity = disp.GetOpacity()
            visible = disp.GetVisibility() == 1
        material = pygfx.MeshPhongMaterial(
            color=[color[0], color[1], color[2], opacity],
            shininess=30,
        )
        if opacity < 1.0:
            material.alpha_mode = "blend"
        mesh = pygfx.Mesh(geom, material)
        mesh.visible = visible
        return {"root": mesh, "mesh": mesh, "material": material, "pd_id": id(pd), "pd_mtime": pd.GetMTime()}

    def _update_node(self, node, entry):
        mesh = entry["mesh"]
        material = entry["material"]
        disp = self._get_display(node)
        if disp is not None:
            color = disp.GetColor()
            opacity = disp.GetOpacity()
            material.color = [color[0], color[1], color[2], opacity]
            if opacity < 1.0:
                material.alpha_mode = "blend"
            mesh.visible = disp.GetVisibility() == 1
        pd = node.GetPolyData()
        if pd is not None:
            new_mtime = pd.GetMTime()
            if id(pd) != entry.get("pd_id") or new_mtime != entry.get("pd_mtime"):
                new_geom = polydata_to_pygfx_geometry(pd)
                if new_geom is not None:
                    mesh.geometry = new_geom
                entry["pd_id"] = id(pd)
                entry["pd_mtime"] = new_mtime


# ------------------------------------------------------------------------
# VolumeRenderingDisplayableManager
# ------------------------------------------------------------------------

class SceneRendererManager:
    """Owns one SceneRenderer + a set of Displayers; arbitrates structural
    rebuilds vs. uniform refreshes, and routes pointer events to picking-
    capable Fields for interactive drag.

    This replaces the old per-volume DisplayableManager: instead of one
    pygfx WorldObject per MRML volume, every contributing MRML node maps
    to a Field on the single SceneRenderer that does the per-sample
    compositing.
    """

    def __init__(self, view):
        self.view = view
        self._renderer: SceneRenderer | None = None
        # Suppress structure-change callbacks while we populate the
        # displayer list: each Displayer's __init__ scans the scene and
        # would call back into _rebuild_renderer before _displayers is
        # assigned.
        self._initializing = True
        self._displayers = []
        self._displayers.append(VolumeRenderingDisplayer(
            on_structure_changed=self._on_structure_changed,
            on_field_modified=self._on_field_modified,
        ))
        self._displayers.append(FiducialDisplayer(
            on_structure_changed=self._on_structure_changed,
            on_field_modified=self._on_field_modified,
        ))
        self._initializing = False

        # Active drag state, set by pick on pointer_down
        self._drag_hit = None

        # Subscribe pointer events for pick-and-drag. We register at the
        # rendercanvas widget level so we can decide before the controller
        # whether to consume the event.
        # pygfx 0.16 added `order=` so our handlers run before the
        # controller's. On older (0.15.x) builds the kwarg isn't
        # accepted, so fall back to registering without it — we lose
        # priority but pick/drag still works because the controller
        # accepts/ignores based on button.
        def _add(handler, kind):
            try:
                view.renderer.add_event_handler(handler, kind, order=-10)
            except TypeError:
                view.renderer.add_event_handler(handler, kind)
        try:
            _add(self._on_pointer_down, "pointer_down")
            _add(self._on_pointer_move, "pointer_move")
            _add(self._on_pointer_up, "pointer_up")
        except Exception as e:
            print(f"SceneRendererManager: pointer handlers failed: {e}")

        # Initial scan already populated displayers with whatever was in
        # the scene at construction time; build the renderer now.
        self._rebuild_renderer()

    def cleanup(self):
        for d in self._displayers:
            try:
                d.cleanup()
            except Exception:
                pass
        if self._renderer is not None:
            try:
                self.view.remove(self._renderer)
            except Exception:
                pass
            self._renderer = None

    # ----- Structure / refresh routing -----

    def _gather_fields(self):
        out = []
        for d in self._displayers:
            out.extend(d.fields())
        return out

    def _on_structure_changed(self):
        if getattr(self, "_initializing", False):
            return
        self._rebuild_renderer()

    def _on_field_modified(self, field):
        if self._renderer is None:
            return
        # Per-frame style update: re-fill uniforms for the changed field.
        for f, s in zip(self._renderer.fields(), self._renderer._slot_indices):
            if f is field:
                f.fill_uniforms(self._renderer.material.uniform_buffer, s)
        self._renderer.material.uniform_buffer.update_full()
        self._renderer.recompute_scene_bounds()
        self.view.request_redraw()

    def _rebuild_renderer(self):
        fields = self._gather_fields()

        # Drop the existing renderer (the SceneMaterial subclass is
        # specific to the previous field configuration).
        if self._renderer is not None:
            try:
                self.view.remove(self._renderer)
            except Exception:
                pass
            self._renderer = None

        if not fields:
            self.view.request_redraw()
            return

        try:
            self._renderer = SceneRenderer.build_for_fields(fields)
        except Exception as e:
            print(f"SceneRendererManager: build_for_fields failed: {e}")
            return

        self.view.add(self._renderer)
        self.view.request_redraw()

    # ----- Picking (pointer event handlers) -----

    def _ndc_from_event(self, ev):
        """Convert a rendercanvas pointer event (logical pixels) to NDC."""
        sz = self.view.widget.get_logical_size()
        if sz[0] <= 0 or sz[1] <= 0:
            return None
        ndc_x = (float(ev["x"]) / sz[0]) * 2.0 - 1.0
        ndc_y = 1.0 - (float(ev["y"]) / sz[1]) * 2.0
        return ndc_x, ndc_y

    def _on_pointer_down(self, ev):
        # Only left button does pick-and-drag
        if int(ev.get("button", 0)) != 1:
            return
        if self._renderer is None:
            return
        ndc = self._ndc_from_event(ev)
        if ndc is None:
            return
        sz = self.view.widget.get_logical_size()
        hit = self._renderer.pick_at(ndc[0], ndc[1], self.view.camera, sz)
        if hit is None:
            return
        self._drag_hit = hit
        # Stop the controller from interpreting this as the start of a
        # rotate. rendercanvas event dicts are mutable; setting a flag
        # in our own dict is fine, but we can't mark it "consumed" the
        # way browser events can. Workaround: temporarily disable the
        # controller while a drag is active.
        if hasattr(self.view, "controller"):
            self.view.controller.enabled = False

    def _on_pointer_move(self, ev):
        if self._drag_hit is None or self._renderer is None:
            return
        ndc = self._ndc_from_event(ev)
        if ndc is None:
            return
        sz = self.view.widget.get_logical_size()
        if self._renderer.drag_continue(self._drag_hit, ndc[0], ndc[1],
                                        self.view.camera, sz):
            self.view.request_redraw()

    def _on_pointer_up(self, ev):
        if self._drag_hit is None:
            return
        # On release, push the new sphere position back into MRML via the
        # owning displayer (FiducialDisplayer.commit_drag).
        for d in self._displayers:
            if isinstance(d, FiducialDisplayer):
                if self._drag_hit.field in d.fields():
                    d.commit_drag(self._drag_hit.field, self._drag_hit.item_index)
                    break
        self._drag_hit = None
        if hasattr(self.view, "controller"):
            self.view.controller.enabled = True

    # ----- Test/inspection --------

    @property
    def renderer(self) -> SceneRenderer | None:
        return self._renderer


# ------------------------------------------------------------------------
# SegmentationDisplayableManager
# ------------------------------------------------------------------------

class SegmentationDisplayableManager(DisplayableManager):
    node_class = "vtkMRMLSegmentationNode"

    def _extra_watch(self, node, tags):
        disp = node.GetDisplayNode()
        if disp is not None:
            tags.append((disp, disp.AddObserver(vtk.vtkCommand.ModifiedEvent, self._on_node_modified)))
        seg = node.GetSegmentation()
        if seg is not None:
            tags.append((seg, seg.AddObserver(vtk.vtkCommand.ModifiedEvent, self._on_node_modified)))

    def _ensure_closed_surface(self, seg_node):
        seg = seg_node.GetSegmentation()
        closed = slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
        if not seg.ContainsRepresentation(closed):
            seg.CreateRepresentation(closed)
        return closed

    def _add_node(self, node):
        closed = self._ensure_closed_surface(node)
        group = pygfx.Group()
        entry = {
            "root": group,
            "group": group,
            "segments": {},
            "segment_ids_cached": tuple(),
        }
        self._rebuild_segments(node, entry, closed, force=True)
        return entry

    def _rebuild_segments(self, node, entry, closed_repr=None, force=False):
        if closed_repr is None:
            closed_repr = self._ensure_closed_surface(node)
        seg = node.GetSegmentation()
        disp = node.GetDisplayNode()
        group = entry["group"]

        current_ids = tuple(seg.GetNthSegmentID(i) for i in range(seg.GetNumberOfSegments()))
        structural_change = force or current_ids != entry["segment_ids_cached"]

        if structural_change:
            current_set = set(current_ids)
            for sid in list(entry["segments"].keys()):
                if sid not in current_set:
                    old = entry["segments"].pop(sid)
                    try:
                        group.remove(old["mesh"])
                    except Exception:
                        pass
            entry["segment_ids_cached"] = current_ids

        for sid in current_ids:
            segment = seg.GetSegment(sid)
            if segment is None:
                continue
            pd = segment.GetRepresentation(closed_repr)
            if pd is None:
                continue

            color = segment.GetColor()
            visible = True
            opacity = 1.0
            if disp is not None:
                visible = bool(disp.GetSegmentVisibility(sid) and disp.GetVisibility3D())
                if hasattr(disp, "GetSegmentOpacity3D"):
                    opacity = float(disp.GetSegmentOpacity3D(sid))

            existing = entry["segments"].get(sid)
            if existing is None:
                geom = polydata_to_pygfx_geometry(pd)
                if geom is None:
                    continue
                material = pygfx.MeshPhongMaterial(
                    color=[color[0], color[1], color[2], opacity],
                    shininess=20,
                )
                if opacity < 1.0:
                    material.alpha_mode = "blend"
                mesh = pygfx.Mesh(geom, material)
                mesh.visible = visible
                group.add(mesh)
                entry["segments"][sid] = {
                    "mesh": mesh, "material": material, "pd_mtime": pd.GetMTime(),
                }
            else:
                existing["material"].color = [color[0], color[1], color[2], opacity]
                if opacity < 1.0:
                    existing["material"].alpha_mode = "blend"
                existing["mesh"].visible = visible
                new_mtime = pd.GetMTime()
                if new_mtime != existing.get("pd_mtime"):
                    geom = polydata_to_pygfx_geometry(pd)
                    if geom is not None:
                        existing["mesh"].geometry = geom
                    existing["pd_mtime"] = new_mtime

    def _update_node(self, node, entry):
        self._rebuild_segments(node, entry)


# ------------------------------------------------------------------------
# CameraDisplayableManager
# ------------------------------------------------------------------------

class CameraDisplayableManager(DisplayableManager):
    """Bidirectional sync between the pygfx camera and a specific MRML camera
    node (by layout label). Pushes pygfx interactions back to MRML only on
    gesture end to avoid event storms during orbit/zoom."""

    node_class = "vtkMRMLCameraNode"

    def __init__(self, view, mrml_scene=None, layout_name="1"):
        self._layout_name = layout_name
        self._tracked_node_id = None
        self._suppress_mrml_event = False
        self._suppress_pygfx_push = False
        super().__init__(view, mrml_scene)

        try:
            view.renderer.add_event_handler(self._on_pygfx_event, "pointer_up", "key_down")
            view.renderer.add_event_handler(self._on_pygfx_wheel, "wheel")
        except Exception as e:
            print(f"CameraDisplayableManager: could not register pygfx handler: {e}")

        self._wheel_timer = qt.QTimer()
        self._wheel_timer.setSingleShot(True)
        self._wheel_timer.setInterval(120)
        self._wheel_timer.timeout.connect(self._flush_wheel_push)

    def _add_node(self, node):
        # GetActiveTag() was deprecated in favour of GetLayoutName() in
        # recent Slicer builds. GetLayoutName() returns the bare name
        # ("1"), GetActiveTag() returns "vtkMRMLViewNode" + name.
        if hasattr(node, "GetLayoutName"):
            if node.GetLayoutName() != self._layout_name:
                return None
        else:
            if node.GetActiveTag() != f"vtkMRMLViewNode{self._layout_name}":
                return None
        self._tracked_node_id = node.GetID()
        self._apply_mrml_to_pygfx(node)
        return {"root": None, "camera_node": node}

    def _update_node(self, node, entry):
        if self._suppress_mrml_event:
            return
        self._apply_mrml_to_pygfx(node)

    # ----- FOV convention conversion -----
    #
    # VTK's vtkCamera.ViewAngle is the full VERTICAL field of view in
    # degrees (when UseHorizontalViewAngle is off, which is Slicer's
    # default). pygfx's PerspectiveCamera.fov is an aspect-averaged
    # reference angle where (width + height) = 4*near*tan(fov/2) at
    # the near plane: see pygfx._perspective._update_projection_matrix.
    # For any aspect other than 1.0 these are not the same number, so
    # copying VTK.ViewAngle straight into pygfx.fov makes the pygfx
    # vertical FOV smaller than VTK's and the view looks flatter /
    # further away. We bridge the two with these two helpers.

    @staticmethod
    def _vtk_vfov_to_pygfx_fov(vtk_fov_deg, aspect):
        """Convert a VTK vertical FOV (deg) to the pygfx fov (deg) that
        produces the same vertical FOV at the given widget aspect."""
        import math
        if aspect <= 0:
            aspect = 1.0
        half_vtk = math.radians(0.5 * float(vtk_fov_deg))
        half_pyg = math.atan((1.0 + aspect) * 0.5 * math.tan(half_vtk))
        return math.degrees(2.0 * half_pyg)

    @staticmethod
    def _pygfx_fov_to_vtk_vfov(pygfx_fov_deg, aspect):
        """Inverse of the above."""
        import math
        if aspect <= 0:
            aspect = 1.0
        half_pyg = math.radians(0.5 * float(pygfx_fov_deg))
        half_vtk = math.atan(2.0 * math.tan(half_pyg) / (1.0 + aspect))
        return math.degrees(2.0 * half_vtk)

    def _current_widget_aspect(self):
        try:
            w, h = self.view.widget.get_logical_size()
        except Exception:
            return 1.0
        return (w / h) if h > 0 else 1.0

    def _apply_mrml_to_pygfx(self, cam_node):
        vtk_cam = cam_node.GetCamera()
        pos = vtk_cam.GetPosition()
        focal = vtk_cam.GetFocalPoint()
        vtk_vfov = float(vtk_cam.GetViewAngle())
        pygfx_cam = self.view.camera

        # Keep pygfx's reference aspect locked to the widget's current
        # aspect so `maintain_aspect` doesn't second-guess us, then pick
        # a pygfx fov that maps to the requested VTK vertical FOV.
        aspect = self._current_widget_aspect()
        pygfx_cam.aspect = aspect
        pygfx_cam.fov = self._vtk_vfov_to_pygfx_fov(vtk_vfov, aspect)
        pygfx_cam.local.position = tuple(pos)
        self._suppress_pygfx_push = True
        try:
            pygfx_cam.look_at(tuple(focal))
            if hasattr(self.view.controller, "target"):
                self.view.controller.target = tuple(focal)
        finally:
            self._suppress_pygfx_push = False
        self.view.request_redraw()

    def _on_pygfx_event(self, ev):
        if self._suppress_pygfx_push or self._tracked_node_id is None:
            return
        cam_node = self.mrml_scene.GetNodeByID(self._tracked_node_id)
        if cam_node is None:
            return
        self._push_pygfx_to_mrml(cam_node)

    def _on_pygfx_wheel(self, ev):
        self._wheel_timer.stop()
        self._wheel_timer.start()

    def _flush_wheel_push(self):
        if self._tracked_node_id is None:
            return
        cam_node = self.mrml_scene.GetNodeByID(self._tracked_node_id)
        if cam_node is not None:
            self._push_pygfx_to_mrml(cam_node)

    def _push_pygfx_to_mrml(self, cam_node):
        pygfx_cam = self.view.camera
        pos = tuple(float(x) for x in pygfx_cam.local.position)
        target = getattr(self.view.controller, "target", None)
        if target is None:
            fwd = pygfx_cam.local.forward
            target = tuple(pos[i] + float(fwd[i]) * 300.0 for i in range(3))
        else:
            target = tuple(float(x) for x in target)
        up = tuple(float(x) for x in pygfx_cam.local.up)
        # Convert pygfx's aspect-averaged fov back to a VTK vertical FOV
        # so the side-by-side VTK pane renders at the same effective
        # vertical angle.
        aspect = pygfx_cam.aspect if pygfx_cam.aspect > 0 else \
                 self._current_widget_aspect()
        vtk_vfov = self._pygfx_fov_to_vtk_vfov(float(pygfx_cam.fov), aspect)

        vtk_cam = cam_node.GetCamera()
        # vtkMRMLViewLinkLogic only broadcasts a camera modification to
        # other views (honouring each view's LinkedControl flag) when
        # the source camera reports Interacting == true AND non-zero
        # InteractionFlags. Mirror the same pattern VTK's interactor
        # uses during a real orbit so pygfx-originated pushes travel
        # through Slicer's standard link propagation.
        self._suppress_mrml_event = True
        was_modifying = cam_node.StartModify()
        try:
            vtk_cam.SetPosition(*pos)
            vtk_cam.SetFocalPoint(*target)
            vtk_cam.SetViewUp(*up)
            vtk_cam.SetViewAngle(vtk_vfov)
            cam_node.SetInteractionFlags(
                slicer.vtkMRMLCameraNode.CameraInteractionFlag)
            cam_node.SetInteracting(1)
            # EndModify fires one coalesced ModifiedEvent with the flags
            # set, so ViewLinkLogic broadcasts iff the user has Link on.
            cam_node.EndModify(was_modifying)
            cam_node.SetInteracting(0)
            # Tell the tracked view's displayable manager to recompute
            # its auto-clipping range for the new camera pose. Same
            # event Slicer's own interactor styles fire after a drag.
            cam_node.ResetClippingRange()
        finally:
            self._suppress_mrml_event = False

    def cleanup(self):
        try:
            self._wheel_timer.stop()
        except Exception:
            pass
        super().cleanup()


# ------------------------------------------------------------------------
# ViewDisplayableManager: background + orientation triad
# ------------------------------------------------------------------------

def _make_axis_cube():
    group = pygfx.Group()
    size = 12.0
    cube_mesh = pygfx.Mesh(
        pygfx.box_geometry(size, size, size),
        pygfx.MeshPhongMaterial(color=[0.7, 0.7, 0.75, 0.35]),
    )
    cube_mesh.material.alpha_mode = "blend"
    group.add(cube_mesh)

    def axis_line(start, end, color):
        positions = np.array([start, end], dtype=np.float32)
        return pygfx.Line(pygfx.Geometry(positions=positions), pygfx.LineMaterial(color=color, thickness=3.0))

    half = size * 0.75
    group.add(axis_line([-half, 0, 0], [half, 0, 0], "#ff5555"))
    group.add(axis_line([0, -half, 0], [0, half, 0], "#55ff55"))
    group.add(axis_line([0, 0, -half], [0, 0, half], "#5599ff"))
    return group


class ViewDisplayableManager(DisplayableManager):
    node_class = "vtkMRMLViewNode"

    def __init__(self, view, mrml_scene=None, layout_name="1"):
        self._layout_name = layout_name
        super().__init__(view, mrml_scene)

    def _add_node(self, node):
        if node.GetLayoutName() != self._layout_name:
            return None
        bg_mat = pygfx.BackgroundMaterial((0, 0, 0, 1), (0, 0, 0, 1))
        bg = pygfx.Background(None, bg_mat)
        self.view.scene.add(bg)
        self.view._background_obj = bg

        cube = _make_axis_cube()

        entry = {
            "root": cube, "cube": cube, "bg": bg, "bg_material": bg_mat,
            "view_node_id": node.GetID(),
            "last_bg1": None, "last_bg2": None,
        }
        self._apply_state(node, entry)
        return entry

    def _update_node(self, node, entry):
        self._apply_state(node, entry)

    def _apply_state(self, view_node, entry):
        bg1 = tuple(view_node.GetBackgroundColor())
        bg2 = tuple(view_node.GetBackgroundColor2())
        if bg1 != entry["last_bg1"] or bg2 != entry["last_bg2"]:
            bg_mat = entry["bg_material"]
            try:
                bg_mat.color_top = (bg1[0], bg1[1], bg1[2], 1.0)
                bg_mat.color_bottom = (bg2[0], bg2[1], bg2[2], 1.0)
            except AttributeError:
                new_mat = pygfx.BackgroundMaterial(
                    (bg1[0], bg1[1], bg1[2], 1.0),
                    (bg2[0], bg2[1], bg2[2], 1.0),
                )
                entry["bg"].material = new_mat
                entry["bg_material"] = new_mat
            entry["last_bg1"] = bg1
            entry["last_bg2"] = bg2
        cube = entry.get("cube")
        if cube is not None:
            cube.visible = bool(view_node.GetBoxVisible() or view_node.GetAxisLabelsVisible())

    def _remove_node(self, entry):
        bg = entry.get("bg")
        if bg is not None:
            try:
                self.view.scene.remove(bg)
            except Exception:
                pass
            if getattr(self.view, "_background_obj", None) is bg:
                self.view._background_obj = None


# ------------------------------------------------------------------------
# Slicer-style controller bindings
# ------------------------------------------------------------------------

def configure_slicer_controls(controller):
    """left=rotate, middle=pan, right=zoom, shift+left=pan, wheel=zoom.

    Right-click zoom uses a float multiplier so pygfx's Action class reduces
    the 2D drag delta to its Y component (see Action.delta in
    pygfx/controllers/_base.py). That matches VTK's
    vtkInteractorStyleTrackballCamera dolly, which reads only dy. Sign is
    positive so drag-down = zoom-in (closer): the grabbed point moves toward
    the camera as the cursor moves down.
    """
    try:
        controller.controls["mouse1"] = ("rotate", "drag", (0.005, 0.005))
        # Pan multiplier is tuned to match VTK's trackball pan (~1 mm per
        # screen pixel at a typical volume-view distance). OrbitController's
        # pan path inflates the raw pixel delta by distance_to_target * 0.01
        # when _custom_target is set (which our CameraDisplayableManager does
        # on every MRML sync), so the 0.01 on our side roughly cancels that
        # inflation and leaves a VTK-equivalent feel regardless of zoom.
        controller.controls["mouse3"] = ("pan", "drag", (0.01, 0.01))
        controller.controls["shift+mouse1"] = ("pan", "drag", (0.01, 0.01))
        controller.controls["mouse2"] = ("zoom", "drag", 0.01)
        controller.controls["control+mouse1"] = ("zoom", "drag", 0.01)
        controller.controls["wheel"] = ("zoom", "push", -0.001)
        controller.controls["alt+wheel"] = ("fov", "push", -0.01)
    except Exception as e:
        print(f"configure_slicer_controls: {e}")


# ------------------------------------------------------------------------
# Layout + DualView
# ------------------------------------------------------------------------

CUSTOM_LAYOUT_ID = 5033
CUSTOM_LAYOUT_XML = """
<layout type="vertical" split="true">
 <item splitSize="500">
  <layout type="horizontal">
   <item>
    <view class="vtkMRMLViewNode" singletontag="1">
     <property name="viewlabel" action="default">1</property>
    </view>
   </item>
   <item>
    <view class="vtkMRMLViewNode" singletontag="2">
     <property name="viewlabel" action="default">2</property>
     <property name="viewcolor" action="default">#33AAFF</property>
    </view>
   </item>
  </layout>
 </item>
 <item splitSize="500">
  <layout type="horizontal">
   <item>
    <view class="vtkMRMLSliceNode" singletontag="Red">
     <property name="orientation" action="default">Axial</property>
     <property name="viewlabel" action="default">R</property>
     <property name="viewcolor" action="default">#F34A33</property>
    </view>
   </item>
   <item>
    <view class="vtkMRMLSliceNode" singletontag="Yellow">
     <property name="orientation" action="default">Sagittal</property>
     <property name="viewlabel" action="default">Y</property>
     <property name="viewcolor" action="default">#EDD54C</property>
    </view>
   </item>
   <item>
    <view class="vtkMRMLSliceNode" singletontag="Green">
     <property name="orientation" action="default">Coronal</property>
     <property name="viewlabel" action="default">G</property>
     <property name="viewcolor" action="default">#6EB04B</property>
    </view>
   </item>
  </layout>
 </item>
</layout>
"""


class DualView:
    _instance = None

    def __init__(self):
        self.view = None
        self.managers = []
        self._prior_layout = None

    @classmethod
    def install(cls):
        if cls._instance is not None:
            return cls._instance
        inst = cls()
        inst._install()
        cls._instance = inst
        return inst

    @classmethod
    def uninstall(cls):
        if cls._instance is not None:
            cls._instance._uninstall()
            cls._instance = None
            return
        # No tracked instance in THIS module -- but a previous import
        # cycle (sys.modules.pop + re-import) may have orphaned widgets
        # in the 3D view layout. Sweep those up too, so uninstall() is a
        # robust "make the layout clean" call regardless of history.
        cls()._purge_stale_pygfx_widgets()

    def _install(self):
        self._register_layout()
        self._prior_layout = slicer.app.layoutManager().layout
        slicer.app.layoutManager().setLayout(CUSTOM_LAYOUT_ID)
        slicer.app.processEvents()

        self.view = PygfxView()
        self._inject_widget()

        configure_slicer_controls(self.view.controller)

        self.view.widget.add_event_handler(
            self._on_double_click, "double_click"
        )

        # Field-based renderer (volumes + fiducials + future) lives on a
        # single SceneRenderer driven by Displayers. Mesh-style nodes
        # (models, segmentations) keep their own scene-graph managers.
        # CameraDisplayableManager tracks our own view node's camera
        # (layoutName "1"). Each MRML view in a layout has its own
        # camera -- Slicer's standard 3D-view Link button in the view
        # controller is what users use to tie independent views
        # together. _sync_camera_to_mrml() below initialises every
        # camera to the same state at install time so the two panes
        # start out visually identical; subsequent linking behaviour
        # is up to Slicer's native link toggle.
        self.managers = [
            ViewDisplayableManager(self.view, layout_name="1"),
            ModelDisplayableManager(self.view),
            SegmentationDisplayableManager(self.view),
            SceneRendererManager(self.view),
            CameraDisplayableManager(self.view, layout_name="1"),
        ]

        self.view.reset_camera()
        # reset_camera() repositioned the pygfx camera (show_object on
        # the scene). First push that state into the tracked MRML
        # camera node (our own view), then copy the same state into
        # every other camera node so all panes in the custom layout
        # start out looking at the scene the same way.
        self._sync_camera_to_mrml()
        self._initialize_all_cameras_to_match()

    def _sync_camera_to_mrml(self):
        """Push the current pygfx camera state into the tracked MRML
        camera node so the paired VTK view mirrors our framing. Only
        the tracked camera (our own view node, layoutName '1') is
        touched -- Slicer's other view nodes keep their own cameras
        exactly as they were, and the user can toggle the standard 3D
        view controller Link button to couple them."""
        cam_mgr = next(
            (m for m in self.managers if isinstance(m, CameraDisplayableManager)),
            None,
        )
        if cam_mgr is None or cam_mgr._tracked_node_id is None:
            return
        node = slicer.mrmlScene.GetNodeByID(cam_mgr._tracked_node_id)
        if node is None:
            return
        cam_mgr._push_pygfx_to_mrml(node)

    def _initialize_all_cameras_to_match(self):
        """Copy the tracked camera's state into every other camera
        node so all panes in the layout start from the same viewpoint.
        One-shot install-time call. Skips any camera that is already
        at the same position/focal/up/fov so we don't kick a redundant
        VTK render when Slicer's link logic has already propagated."""
        cam_mgr = next(
            (m for m in self.managers if isinstance(m, CameraDisplayableManager)),
            None,
        )
        if cam_mgr is None or cam_mgr._tracked_node_id is None:
            return
        tracked = slicer.mrmlScene.GetNodeByID(cam_mgr._tracked_node_id)
        if tracked is None:
            return
        src = tracked.GetCamera()
        pos = src.GetPosition()
        focal = src.GetFocalPoint()
        up = src.GetViewUp()
        vfov = src.GetViewAngle()

        def _eq_vec(a, b, tol=1e-6):
            return all(abs(float(a[i]) - float(b[i])) <= tol for i in range(len(a)))

        nodes = [n for n in slicer.util.getNodesByClass("vtkMRMLCameraNode")
                 if n is not tracked]
        for node in nodes:
            dst = node.GetCamera()
            if (_eq_vec(dst.GetPosition(), pos) and
                    _eq_vec(dst.GetFocalPoint(), focal) and
                    _eq_vec(dst.GetViewUp(), up) and
                    abs(dst.GetViewAngle() - vfov) <= 1e-6):
                continue  # already matches; no Modified, no extra render
            was_modifying = node.StartModify()
            try:
                dst.SetPosition(*pos)
                dst.SetFocalPoint(*focal)
                dst.SetViewUp(*up)
                dst.SetViewAngle(vfov)
                # Explicit Modified inside the batch -- vtkCamera's
                # SetXXX fires on the vtkCamera, but we also want the
                # MRML node to report a change so downstream pipelines
                # (clipping range, schedule render) run.
                node.Modified()
            finally:
                node.EndModify(was_modifying)
            # Jumping the camera a long way (default (0, 500, 0) ->
            # volume-framed ~1.2m) leaves VTK's auto-clipping planes
            # stuck on the old bounds, producing corrupted front/back
            # clipping until the next user interaction recomputes
            # them. vtkMRMLCameraDisplayableManager observes this event
            # and calls renderer->ResetCameraClippingRange() for us.
            node.ResetClippingRange()
            # Let Qt/VTK flush the camera-modified + reset-clipping +
            # schedule-render + actual-render chain before we move on.
            slicer.app.processEvents()

    def _uninstall(self):
        for m in self.managers:
            try:
                m.cleanup()
            except Exception:
                pass
        self.managers = []

        if self.view is not None:
            try:
                self.view.close()
            except Exception:
                pass
        # Purge our widget *and* any leftover pygfx widgets -- the latter
        # handles the case where a previous install was orphaned by a
        # sys.modules reload (old class, old _instance lost, widgets
        # still parked in the layout). Runs regardless of self.view.
        self._purge_stale_pygfx_widgets()
        self.view = None

        if self._prior_layout is not None:
            try:
                slicer.app.layoutManager().setLayout(self._prior_layout)
            except Exception:
                pass

    def _register_layout(self):
        lm_node = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
        if not lm_node.IsLayoutDescription(CUSTOM_LAYOUT_ID):
            lm_node.AddLayoutDescription(CUSTOM_LAYOUT_ID, CUSTOM_LAYOUT_XML)

    def _on_double_click(self, ev):
        """Toggle maximized-view mode, matching Slicer's native 3D view."""
        tdw = self._find_pygfx_threeDWidget()
        if tdw is None:
            return
        view_node = tdw.mrmlViewNode() if hasattr(tdw, "mrmlViewNode") else tdw.threeDView().mrmlViewNode()
        layout_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLayoutNode")
        if layout_node is None or view_node is None:
            return
        if layout_node.IsMaximizedViewNode(view_node):
            layout_node.RemoveMaximizedViewNode(view_node)
        else:
            layout_node.AddMaximizedViewNode(view_node)

    # Distinctive objectName so we can recognise our widgets even across
    # sys.modules reloads where the DualView class itself gets rebound
    # (at which point DualView._instance is unreachable from the fresh
    # module but the Qt widgets stay alive until something removes them).
    PYGFX_WIDGET_OBJECT_NAME = "slicer_wgpu.DualView.pygfx_widget"

    # Which MRML view-node layoutName designates the pygfx pane. We
    # bind by layoutName rather than by `threeDWidget(0)` index because
    # Slicer allocates threeDWidget indices in view-node-creation order,
    # which isn't correlated with the XML-driven on-screen left/right
    # placement -- so hardcoding index 0 can flip sides across sessions.
    PYGFX_VIEW_LAYOUT_NAME = "1"

    def _find_pygfx_threeDWidget(self):
        """Return the qMRMLThreeDWidget whose view node has layoutName
        == PYGFX_VIEW_LAYOUT_NAME, i.e. the on-screen slot we own.
        Returns None if the custom layout isn't active yet."""
        lm = slicer.app.layoutManager()
        for i in range(lm.threeDViewCount):
            tdw = lm.threeDWidget(i)
            try:
                view_node = tdw.mrmlViewNode()
            except Exception:
                view_node = None
            if view_node is not None and \
                    view_node.GetLayoutName() == self.PYGFX_VIEW_LAYOUT_NAME:
                return tdw
        return None

    def _purge_stale_pygfx_widgets(self):
        """Remove every pygfx widget (and every `QRenderWidget`-lookalike)
        that a previous install() dropped into the pygfx pane's layout,
        and un-hide the native VTK threeDView. Safe to call whether or
        not a prior install exists; used both on install (to clean up
        orphans from reloaded modules) and uninstall."""
        tdw = self._find_pygfx_threeDWidget()
        if tdw is None:
            return
        layout = tdw.layout()
        if layout is None:
            return
        # Snapshot first -- layout.count() shifts as we remove.
        victims = []
        for i in range(layout.count()):
            w = layout.itemAt(i).widget() if layout.itemAt(i) else None
            if w is None:
                continue
            cls = type(w).__name__
            if (w.objectName == self.PYGFX_WIDGET_OBJECT_NAME
                    or cls == "QRenderWidget"):
                victims.append(w)
        for w in victims:
            try:
                layout.removeWidget(w)
            except Exception:
                pass
            try:
                w.hide()
                w.setParent(None)
                w.deleteLater()
            except Exception:
                pass
        # Make sure the native VTK view is visible again.
        try:
            tdw.threeDView().show()
        except Exception:
            pass

    def _inject_widget(self):
        lm = slicer.app.layoutManager()
        if lm.threeDViewCount < 2:
            raise RuntimeError(f"Expected 2 threeDViews in custom layout, got {lm.threeDViewCount}")
        tdw = self._find_pygfx_threeDWidget()
        if tdw is None:
            raise RuntimeError(
                f"No threeDWidget with view-node layoutName="
                f"{self.PYGFX_VIEW_LAYOUT_NAME!r}. Custom layout wasn't applied.")
        # Sweep out any orphaned pygfx widgets a reloaded-module install
        # may have left behind; without this the new widget shares the
        # layout cell with the old one and both render at half height.
        self._purge_stale_pygfx_widgets()
        tdw.threeDView().hide()
        self.view.widget.setObjectName(self.PYGFX_WIDGET_OBJECT_NAME)
        tdw.layout().addWidget(self.view.widget)
        self.view.widget.show()


# ------------------------------------------------------------------------
# Convenience
# ------------------------------------------------------------------------

def install():
    return DualView.install()


def uninstall():
    DualView.uninstall()
