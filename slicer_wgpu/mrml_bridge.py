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
# Tell the pieper/rendercanvas fork to use its PythonQt backend BEFORE
# rendercanvas.qt is imported. The fork's select_qt_lib() honours this
# env var first, falling back to `"PythonQt" in sys.modules` detection
# otherwise -- and the sys.modules path is unreliable on some Slicer
# builds (Mac preview in particular) where `import PythonQt` works in
# a Python console but the top-level name isn't surfaced the same way
# during a scripted module's top-level import.
import os as _os_env
_os_env.environ.setdefault("_RENDERCANVAS_QT_LIB", "PythonQt")
# Still import PythonQt so QApplication.instance() detection works in
# the fork's qt_lib_has_app() helper.
import PythonQt  # noqa: F401
import PythonQt.QtCore  # noqa: F401
import pygfx
import pylinalg as la
from rendercanvas.qt import QRenderWidget, QtLoop, WA_PaintOnScreen


class _VtkStyleOrbitController(pygfx.OrbitController):
    """OrbitController variant whose azimuth axis is the camera's
    *current* view-up instead of the fixed ``reference_up``.

    This matches vtkInteractorStyleTrackballCamera's behavior: if you
    orbit up over the pole, the camera's view-up flips, and subsequent
    horizontal motion continues to rotate around that (now-flipped) up.
    pygfx's stock OrbitController azimuths around a world-fixed
    reference_up, which near the pole feels "stuck" and doesn't match
    Slicer's default 3D view interaction.

    Elevation is unchanged: still about the camera's local right axis.
    """

    def _update_rotate(self, delta):
        assert isinstance(delta, tuple) and len(delta) == 2

        delta_azimuth, delta_elevation = delta
        camera_state = self._get_camera_state()

        position = camera_state["position"]
        rotation = camera_state["rotation"]

        # VTK-style: azimuth axis is the camera's *current* up in world
        # space, derived from the rotation quaternion, not the fixed
        # reference_up. Elevation axis is the camera's local right
        # (+X), matching stock OrbitController.
        current_up = la.vec_transform_quat((0, 1, 0), rotation)

        # Clip elevation the same way stock OrbitController does so we
        # don't flip past the pole unintentionally. We still compute
        # against reference_up for the clip because it's a stable
        # world-space axis.
        forward = la.vec_transform_quat((0, 0, -1), rotation)
        ref_up = camera_state["reference_up"]
        elevation = la.vec_angle(forward, ref_up) - 0.5 * np.pi
        bounds = -89 * np.pi / 180, 89 * np.pi / 180
        new_elevation = elevation + delta_elevation
        if new_elevation < bounds[0]:
            delta_elevation = bounds[0] - elevation
        elif new_elevation > bounds[1]:
            delta_elevation = bounds[1] - elevation

        r_azimuth = la.quat_from_axis_angle(current_up, -delta_azimuth)
        r_elevation = la.quat_from_axis_angle((1, 0, 0), -delta_elevation)

        rot1 = rotation
        rot2 = la.quat_mul(r_azimuth, la.quat_mul(rot1, r_elevation))

        pos1 = position
        if self._custom_target is not None:
            target_pos = self._custom_target
            pos1_to_target = target_pos - pos1
            pos1_to_target_rotated = la.vec_transform_quat(pos1_to_target, r_azimuth)
            right = la.vec_transform_quat((1, 0, 0), rot1)
            r_elevation_world = la.quat_from_axis_angle(right, -delta_elevation)
            pos1_to_target_final = la.vec_transform_quat(
                pos1_to_target_rotated, r_elevation_world)
            pos2 = target_pos - pos1_to_target_final
        else:
            pos2target1 = self._get_target_vec(camera_state, rotation=rot1)
            pos2target2 = self._get_target_vec(camera_state, rotation=rot2)
            pos2 = pos1 + pos2target1 - pos2target2

        self._set_camera_state({"position": pos2, "rotation": rot2})


class _ScreenQRenderWidget(QRenderWidget):
    """QRenderWidget that prefers wgpu's 'screen' present method over
    rendercanvas's default 'bitmap' on platforms where we've verified
    it works. Screen-present lets wgpu draw directly into a
    CAMetalLayer / HWND / X11 surface owned by this widget, skipping
    the per-frame GPU->CPU readback + QImage blit that bitmap-present
    does (~1.4 MB transfer per frame for a typical view).

    rendercanvas defaults to bitmap to sidestep Qt compositor issues
    it has encountered across platforms (see rendercanvas/qt.py
    _rc_get_present_info for the rationale). We opt in only where
    we've tested the screen path doesn't trip wgpu's surface
    validation:

      * darwin (macOS / Metal): verified
      * win32 (Windows / DXGI): expected to work (well-supported
        wgpu target), enabled
      * linux: DISABLED by default -- wgpu's X11 surface config path
        panics on the container/VNC test environments ("Error in
        wgpuSurfaceConfigure: Validation Error", Rust panic from
        wgpu-core, unrecoverable from Python). This is a known
        upstream state; see:
          - wgpu-py #776 (exactly this error)
          - wgpu-py #688 (Qt/Linux WSI meta-issue, maintainer says
            "wait")
          - rendercanvas #175 (upstream made bitmap the Qt default
            in 2026-02 on the strength of async bitmap present in
            rendercanvas #138)

        Falls back to bitmap on Linux until we have a specific
        Linux platform verified. Expected to work on curated NVIDIA
        Linux reference systems (DGX Spark / IGX Orin / Holoscan
        devkits -- Ubuntu 24.04 GNOME-on-Xorg, NVIDIA driver with
        full VK_KHR_xlib_surface + xcb_surface + wayland_surface,
        where NVIDIA's own Holoviz runs on Vulkan WSI) -- when
        testing there, extend _SCREEN_ENABLED_PLATFORMS or flip
        via env var before first widget construction.

    Subclass and override _SCREEN_ENABLED_PLATFORMS to change the
    default set per deployment.
    """

    _SCREEN_ENABLED_PLATFORMS = frozenset(("darwin", "win32"))

    def _rc_get_present_info(self, present_methods):
        import sys
        if (sys.platform in self._SCREEN_ENABLED_PLATFORMS
                and "screen" in present_methods):
            surface_ids = self._get_surface_ids()
            if surface_ids:
                # WA_PaintOnScreen implies WA_NativeWindow; Qt stops
                # painting over the surface area so wgpu owns those
                # pixels.
                self.setAttribute(WA_PaintOnScreen, True)
                return {"method": "screen", **surface_ids}
        # Fall back to whatever rendercanvas would have picked
        # (= bitmap on Qt).
        return super()._rc_get_present_info(present_methods)


class _SilentQtLoop(QtLoop):
    """QtLoop subclass that never schedules tasks.

    We keep a real QtLoop instance on the canvas group -- several
    paths in rendercanvas/qt.py check `isinstance(loop, QtLoop)` and
    take a different (worse) code path if False -- but override
    _rc_add_task so nothing is ever added to the loop's task list.

    Consequences:
      * The keep-alive loop-task (`while True: await sleep(0.1)`
        in rendercanvas/core/loop.py:198) is never started.
      * Per-canvas Scheduler tasks are never started.
      * __start -> __setup_interrupt_hooks never runs, so
        rendercanvas does not install a SIGINT handler.
      * No QTimer chain; paint events still arrive directly from Qt.

    Redraws in our integration go through PygfxView.request_redraw,
    which calls widget._process_events() + widget.force_draw() to
    flush events and synchronously render + present the frame.
    """

    def _rc_add_task(self, async_func, name):
        # Silently drop. We do not use rendercanvas's async machinery
        # in Slicer; MRML events drive redraws via request_redraw.
        return None


# Install the silent loop on the canvas group singleton BEFORE any
# QRenderWidget is constructed. _rc_canvas_group is a class attribute
# on QRenderWidget (rendercanvas/qt.py:386). select_loop() raises
# "Cannot select_loop() when live canvases exist" if canvases from a
# prior import are still registered -- which happens whenever
# _ensure_dependencies pops slicer_wgpu.mrml_bridge from sys.modules
# and triggers a re-import. In that case the loop already in place
# is (or was) our own _SilentQtLoop from the previous import cycle,
# so we leave it alone. Detection is by class name because a
# re-imported module produces a different class object even for the
# "same" _SilentQtLoop.
def _install_silent_loop():
    current = QRenderWidget._rc_canvas_group.get_loop()
    if type(current).__name__ == "_SilentQtLoop":
        return
    try:
        QRenderWidget._rc_canvas_group.select_loop(_SilentQtLoop())
    except RuntimeError:
        # Live canvases exist but current loop isn't ours. Best we
        # can do is neuter _rc_add_task on the existing QtLoop so
        # it stops scheduling tasks going forward. This is belt-
        # and-suspenders; the branch above normally catches re-import.
        current._rc_add_task = lambda *_a, **_k: None


_install_silent_loop()


from .scene_renderer import SceneRenderer
from .displayers import VolumeRenderingDisplayer, FiducialDisplayer


# ------------------------------------------------------------------------
# _SyncFlushFilter -- drain rendercanvas's EventEmitter on Qt events
# ------------------------------------------------------------------------
#
# rendercanvas QRenderWidget's Qt event handlers (mouseMoveEvent,
# mousePressEvent, wheelEvent, keyPressEvent, ...) submit events into
# self._events, a rendercanvas EventEmitter that buffers them. The
# buffer is normally drained by Scheduler._process_events(), which we
# disabled along with the scheduler (see _SilentQtLoop above). Without
# a drain, our pick handlers and the pygfx OrbitController -- both
# registered via renderer.add_event_handler -- never see the events.
#
# This QObject event filter watches for the Qt event types that
# QRenderWidget turns into rendercanvas events, and calls
# widget._process_events() right after Qt's dispatch finishes. Events
# flow in the natural Qt order:
#
#   Qt delivers event -> QRenderWidget's native handler submits to
#   emitter -> this filter drains the emitter -> our Python handlers
#   fire -> a handler may call self.request_redraw() -> Qt paintEvent
#   is scheduled (coalesced) -> pygfx renders.
#
# No scheduler, no background task chain, no signal hooks. All event
# dispatch runs on Qt's main thread in response to Qt events.
def _flush_event_type_ints():
    QE = qt.QEvent
    return frozenset(int(t) for t in (
        QE.MouseMove, QE.MouseButtonPress, QE.MouseButtonRelease,
        QE.MouseButtonDblClick, QE.Wheel,
        QE.KeyPress, QE.KeyRelease,
        QE.Enter, QE.Leave,
        QE.Resize,
    ))


class _SyncFlushFilter(qt.QObject):
    """Qt event filter that drains a QRenderWidget's rendercanvas
    event emitter immediately after Qt dispatches an event to it."""

    _types = None

    def __init__(self, widget):
        super().__init__()
        self._widget_ref = weakref.ref(widget)
        if _SyncFlushFilter._types is None:
            _SyncFlushFilter._types = _flush_event_type_ints()

    def eventFilter(self, _obj, ev):
        widget = self._widget_ref()
        if widget is None:
            return False
        try:
            if int(ev.type()) in self._types:
                widget._process_events()
        except Exception:
            pass
        return False  # never consume; QRenderWidget still gets the event


# ------------------------------------------------------------------------
# PygfxView: widget + pygfx renderer + scene + camera + lights
# ------------------------------------------------------------------------

class PygfxView:
    """A pygfx render widget with scene, camera, orbit controller, and lights."""

    def __init__(self, parent=None):
        self.widget = (
            _ScreenQRenderWidget(parent) if parent is not None
            else _ScreenQRenderWidget()
        )
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
        self.controller = _VtkStyleOrbitController(
            self.camera, register_events=self.renderer)

        self._closed = False
        self._dirty = True
        # Coalesce redraw requests: many request_redraw() calls in
        # quick succession (e.g. a stream of pointer_move events during
        # an orbit drag) all set this flag, but only one
        # QTimer.singleShot(0, self._do_redraw) actually fires and
        # does one render. Without this every mousemove would trigger
        # a full synchronous ray-march and Qt input would stall.
        self._redraw_pending = False

        try:
            self.renderer.add_event_handler(self._on_controller_event, "pointer_move", "wheel")
        except Exception:
            pass

        # Drain rendercanvas's EventEmitter immediately on each Qt
        # mouse/wheel/key/resize event so our Python handlers (pick,
        # controller, hover) fire on the same Qt tick instead of
        # waiting for the next redraw. See _SyncFlushFilter above.
        self._sync_flush_filter = _SyncFlushFilter(self.widget)
        self.widget.installEventFilter(self._sync_flush_filter)

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
        """Request a redraw at the next idle moment, coalescing bursts
        of requests into a single render. Safe to call at input-event
        rate (every mousemove); Qt will collapse many requests into
        one actual ray-march."""
        if self._closed or self._redraw_pending:
            return
        self._dirty = True
        self._redraw_pending = True
        # singleShot(0) runs after the current Qt event finishes, so
        # all synchronous Python work on this tick (event filter
        # dispatch + controller updates) completes first, then one
        # consolidated render happens.
        qt.QTimer.singleShot(0, self._do_redraw)

    def force_redraw(self):
        """Synchronous version of request_redraw -- blocks until the
        render + present completes. Use sparingly (self-tests that need
        a settled frame before snapshotting)."""
        if self._closed:
            return
        try:
            self.widget.force_draw()
        except Exception:
            pass

    def _do_redraw(self):
        self._redraw_pending = False
        if self._closed:
            return
        try:
            self.widget.force_draw()
        except Exception:
            pass

    def close(self):
        self._closed = True
        try:
            self.widget.draw_frame = None
        except Exception:
            pass
        # widget.close() sends a closeEvent. With the rendercanvas
        # scheduler disabled there is no task list to unregister from,
        # but the close is still the right Qt-native teardown call.
        try:
            self.widget.close()
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
        # Shadow (cinematic) state. Off by default so existing callers
        # aren't affected; call enable_shadows(light_direction) to opt in.
        self._shadows_enabled = False
        self._light_direction = (0.0, 0.0, 0.0)
        self._light_intensity = 1.0
        # Fill light: unshadowed, off by default. Set via set_fill_light().
        self._fill_light_direction = (0.0, 0.0, 0.0)
        self._fill_light_intensity = 0.0
        self._shadow_resolution = 256
        # Suppress structure-change callbacks while we populate the
        # displayer list: each Displayer's __init__ scans the scene and
        # would call back into _rebuild_renderer before _displayers is
        # assigned.
        self._initializing = True
        self._displayers = []
        # Interactive-quality state. During a VolumePropertyNode
        # Start/EndInteractionEvent bracket (i.e. while the user is
        # dragging the Shift slider or poking the TF editor) we halve
        # the renderer's pixel ratio and double each ImageField's
        # sample step so the ray march has roughly 8x less work per
        # frame. VTK's DM handles the same trade-off via
        # DesiredUpdateRate on its mapper; doing it explicitly here
        # keeps control predictable.
        self._interaction_depth = 0
        self._saved_pixel_ratio = None
        self._saved_sample_steps: dict[int, float] = {}
        self._interaction_pixel_ratio = 0.5
        self._interaction_step_mult = 2.0
        self._displayers.append(VolumeRenderingDisplayer(
            on_structure_changed=self._on_structure_changed,
            on_field_modified=self._on_field_modified,
            on_interaction_start=self._on_interaction_start,
            on_interaction_end=self._on_interaction_end,
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
        # Shadows depend on TF + transform + scene bounds, all of which may
        # just have changed. Rebuild the transmittance volume too -- but
        # only when we're not mid-drag, where the user cares about
        # responsiveness more than shadow accuracy.
        if (self._shadows_enabled
                and self._interaction_depth == 0):
            self._refresh_shadow_volume()
        self.view.request_redraw()

    def _on_interaction_start(self):
        """Drop quality for the duration of a drag. Nesting: multiple
        overlapping Start events (e.g. two VolumePropertyNodes both
        interactive at once) push the depth counter; only the first
        actually changes settings, and only the last End restores them."""
        self._interaction_depth += 1
        if self._interaction_depth > 1 or self._renderer is None:
            return
        self._apply_interaction_settings(
            pixel_ratio=self._interaction_pixel_ratio,
            step_mult=self._interaction_step_mult,
        )

    def _on_interaction_end(self):
        self._interaction_depth = max(0, self._interaction_depth - 1)
        if self._interaction_depth > 0 or self._renderer is None:
            return
        self._apply_interaction_settings(
            pixel_ratio=self._saved_pixel_ratio,
            step_mult=None,  # restore from saved baseline
        )
        self._saved_pixel_ratio = None
        # Rebuild shadows once the drag ends, since _on_field_modified
        # was skipping them during the drag.
        if self._shadows_enabled:
            self._refresh_shadow_volume()
            self.view.request_redraw()

    def _apply_interaction_settings(self, pixel_ratio, step_mult):
        """Push the pixel ratio and per-field sample_step_mm through to
        the material buffer + GPU in a single pass. step_mult=None means
        restore each field's saved baseline; a float means
        new_step = saved * step_mult (and stash saved if not yet)."""
        try:
            # pygfx's WgpuRenderer.pixel_ratio is a settable property;
            # None means "follow the canvas's own ratio".
            if step_mult is not None and self._saved_pixel_ratio is None:
                self._saved_pixel_ratio = self.view.renderer.pixel_ratio
            self.view.renderer.pixel_ratio = pixel_ratio
        except Exception as e:
            print(f"SceneRendererManager pixel_ratio change: {e}")

        for f in self._renderer.fields():
            if not hasattr(f, "sample_step_mm"):
                continue
            if step_mult is not None:
                if id(f) not in self._saved_sample_steps:
                    self._saved_sample_steps[id(f)] = f.sample_step_mm
                f.sample_step_mm = self._saved_sample_steps[id(f)] * step_mult
            else:
                base = self._saved_sample_steps.pop(id(f), None)
                if base is not None:
                    f.sample_step_mm = base

        # recompute_scene_bounds() writes the new min(sample_step) into
        # the material uniform buffer; update_full() pushes it to the
        # GPU. Without the explicit upload the sample_step change would
        # sit in CPU memory and the shader would keep using the stale
        # value (same latent gap as _on_field_modified's ordering).
        self._renderer.recompute_scene_bounds()
        self._renderer.material.uniform_buffer.update_full()
        self.view.request_redraw()

    def _rebuild_renderer(self):
        from .fields.image import ImageField
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

        # If shadows are enabled and at least one ImageField is present,
        # build a ShadowVolume and hand it to the renderer. The ShadowVolume
        # is created fresh per rebuild so its compute pipeline matches the
        # current field configuration.
        shadow_volume = None
        image_fields = [f for f in fields if isinstance(f, ImageField)]
        if self._shadows_enabled and image_fields:
            try:
                from .shadows import ShadowVolume
                from pygfx.renderers.wgpu.engine.shared import get_shared
                dev = get_shared().device
                sv = ShadowVolume(dev, resolution=self._shadow_resolution)
                sv.build_pipeline_for_image_fields(image_fields)
                shadow_volume = sv
            except Exception as e:
                print(f"SceneRendererManager: shadow pipeline build failed: {e}")

        try:
            self._renderer = SceneRenderer.build_for_fields(
                fields, shadow_volume=shadow_volume)
        except Exception as e:
            print(f"SceneRendererManager: build_for_fields failed: {e}")
            return

        # Push current light state into the scene material uniforms.
        # Zero key-direction keeps the fragment shader's per-pixel
        # headlight fallback; zero fill-intensity skips the fill term.
        try:
            self._renderer.material.light_direction = self._light_direction
            self._renderer.material.light_intensity = self._light_intensity
            self._renderer.material.fill_light_direction = self._fill_light_direction
            self._renderer.material.fill_light_intensity = self._fill_light_intensity
        except Exception as e:
            print(f"SceneRendererManager: light uniform set failed: {e}")

        # Populate the shadow texture immediately.
        if shadow_volume is not None:
            self._refresh_shadow_volume()

        self.view.add(self._renderer)
        self.view.request_redraw()

    # ----- Shadows -----

    def enable_shadows(self, light_direction, resolution: int = 256,
                       light_intensity: float = 1.0,
                       fill_light_direction=None,
                       fill_light_intensity: float = 0.0):
        """Turn on soft volumetric shadows cast by ImageFields.

        ``light_direction`` is a world-space vector pointing FROM the
        surface TO the light (the same vector you'd use in an NdotL
        term). A zero vector disables the directional light and restores
        the per-pixel headlight fallback.

        ``light_intensity`` scales the key-light (shadowed) direct term.
        ``fill_light_direction`` / ``fill_light_intensity`` configure an
        optional unshadowed fill light (direction pointing from surface
        toward the fill source). Pass ``fill_light_direction=None`` or
        ``fill_light_intensity=0`` to disable the fill.
        """
        self._shadows_enabled = True
        self._light_direction = tuple(float(x) for x in light_direction)
        self._light_intensity = float(light_intensity)
        if fill_light_direction is None:
            self._fill_light_direction = (0.0, 0.0, 0.0)
            self._fill_light_intensity = 0.0
        else:
            self._fill_light_direction = tuple(float(x) for x in fill_light_direction)
            self._fill_light_intensity = float(fill_light_intensity)
        self._shadow_resolution = int(resolution)
        self._rebuild_renderer()

    def disable_shadows(self):
        """Return to plain headlight rendering with no shadow compute."""
        self._shadows_enabled = False
        self._light_direction = (0.0, 0.0, 0.0)
        self._light_intensity = 1.0
        self._fill_light_direction = (0.0, 0.0, 0.0)
        self._fill_light_intensity = 0.0
        self._rebuild_renderer()

    def set_light_direction(self, light_direction):
        """Update the directional key light without toggling shadows.
        Uniform-only; triggers a shadow rebuild if shadows are on."""
        self._light_direction = tuple(float(x) for x in light_direction)
        if self._renderer is not None:
            self._renderer.material.light_direction = self._light_direction
            if self._shadows_enabled:
                self._refresh_shadow_volume()
            self.view.request_redraw()

    def set_key_light_intensity(self, intensity: float):
        """Scale the key (shadowed) light's direct contribution. Uniform-only."""
        self._light_intensity = float(intensity)
        if self._renderer is not None:
            self._renderer.material.light_intensity = self._light_intensity
            self.view.request_redraw()

    def set_fill_light(self, direction, intensity: float = 0.5):
        """Set the unshadowed fill light. ``direction`` is surface->light.
        Pass ``direction=None`` or ``intensity=0`` to disable. Uniform-only;
        fill light does not participate in the shadow compute pass."""
        if direction is None or intensity <= 0:
            self._fill_light_direction = (0.0, 0.0, 0.0)
            self._fill_light_intensity = 0.0
        else:
            self._fill_light_direction = tuple(float(x) for x in direction)
            self._fill_light_intensity = float(intensity)
        if self._renderer is not None:
            self._renderer.material.fill_light_direction = self._fill_light_direction
            self._renderer.material.fill_light_intensity = self._fill_light_intensity
            self.view.request_redraw()

    def _refresh_shadow_volume(self):
        """Re-dispatch the shadow compute pass using the renderer's
        current scene bounds and ImageField uniforms. Cheap -- no GPU
        resource allocation."""
        from .fields.image import ImageField
        if self._renderer is None:
            return
        sv = getattr(self._renderer, "_shadow_volume", None)
        if sv is None:
            return
        image_fields = [f for f in self._renderer.fields()
                        if isinstance(f, ImageField)]
        if not image_fields:
            return
        bmin = self._renderer.material.scene_bounds_min
        bmax = self._renderer.material.scene_bounds_max
        sv.build(bmin=bmin, bmax=bmax,
                 light_dir=self._light_direction,
                 image_fields=image_fields)

    # ----- Picking (pointer event handlers) -----

    @staticmethod
    def _ev_field(ev, key, default=None):
        """Read a pointer-event field that may be either a dict entry
        (older rendercanvas / pygfx) or an attribute (pygfx 0.16+ which
        delivers pygfx.objects.PointerEvent objects)."""
        getter = getattr(ev, "get", None)
        if callable(getter):
            return getter(key, default)
        return getattr(ev, key, default)

    def _ndc_from_event(self, ev):
        """Convert a rendercanvas pointer event (logical pixels) to NDC."""
        sz = self.view.widget.get_logical_size()
        if sz[0] <= 0 or sz[1] <= 0:
            return None
        x = self._ev_field(ev, "x", 0.0)
        y = self._ev_field(ev, "y", 0.0)
        ndc_x = (float(x) / sz[0]) * 2.0 - 1.0
        ndc_y = 1.0 - (float(y) / sz[1]) * 2.0
        return ndc_x, ndc_y

    def _on_pointer_down(self, ev):
        # Only left button does pick-and-drag
        if int(self._ev_field(ev, "button", 0) or 0) != 1:
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
        self._last_pushed_signature = None
        super().__init__(view, mrml_scene)

        try:
            # pointer_move: live-sync during drag (matches VTK's direction,
            # which fires camera-Modified continuously). pointer_up:
            # final commit. key_down: keyboard-driven camera nudges.
            view.renderer.add_event_handler(
                self._on_pygfx_event,
                "pointer_move", "pointer_up", "key_down")
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
        # Skip the push if the pygfx camera state is identical to what
        # we pushed last time. pointer_move fires even on idle hover
        # (no active controller drag) and pushing then would cause a
        # needless cam_node Modified fan-out to VTK, linked views, and
        # the Slicer UI.
        sig = self._pygfx_camera_signature()
        if sig == self._last_pushed_signature:
            return
        self._last_pushed_signature = sig
        self._push_pygfx_to_mrml(cam_node)

    def _pygfx_camera_signature(self):
        pc = self.view.camera
        pos = tuple(float(x) for x in pc.local.position)
        up = tuple(float(x) for x in pc.local.up)
        tgt = getattr(self.view.controller, "target", None)
        tgt = tuple(float(x) for x in tgt) if tgt is not None else None
        return (pos, up, tgt, float(pc.fov), float(pc.aspect))

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
        # No SIGINT restore needed: the rendercanvas loop is disabled
        # at module import time (see the _rc_canvas_group.select_loop(None)
        # call at the top of this file), so __setup_interrupt_hooks
        # is never reached and Python's default SIGINT handler stays
        # in place.
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
