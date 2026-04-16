"""FiducialField: a fixed-capacity array of solid spheres in world space,
rendered procedurally (no geometry buffer, no texture). Inspired by
STEP's FiducialField but adapted for the SceneRenderer's per-sample
compositing loop.

Shading is plastic-Phong: a tinted base from the per-sphere color, a
diffuse term from the per-pixel headlight, and a strong specular
highlight tuned to read like the head of a pushpin against the
volume render.

Each sphere has its own colour. The host application owns the
mapping from "fiducial index" to its source MRML control point; this
field is just a flat list of (centre, radius, color) tuples.

Picking: ray vs. sphere test in Python on pointer_down. Subsequent
drag updates the picked sphere's centre by reprojecting onto the
plane perpendicular to the view through the original hit point.
"""

from __future__ import annotations

import numpy as np

from .base import Field, PickHit


# Capacity is fixed at WGSL-generation time so the shader can declare a
# uniform-array of the right size. 256 is enough for typical markup
# scenes; bump if you need more.
MAX_SPHERES_PER_FIDUCIAL_FIELD = 256


class FiducialField(Field):
    """Procedural sphere field. Owns up to MAX_SPHERES_PER_FIDUCIAL_FIELD
    spheres, each with its own world-space centre, radius, and RGBA tint.
    """

    field_kind = "fid"
    supports_picking = True

    def __init__(
        self,
        centers: np.ndarray | None = None,
        radii: np.ndarray | None = None,
        colors: np.ndarray | None = None,
        *,
        light_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        shininess: float = 80.0,
        k_ambient: float = 0.20,
        k_diffuse: float = 0.85,
        k_specular: float = 0.50,
        visible: bool = True,
    ):
        super().__init__()

        # Backing storage for the uniform arrays.
        self._spheres = np.zeros((MAX_SPHERES_PER_FIDUCIAL_FIELD, 4), dtype=np.float32)
        self._colors = np.zeros((MAX_SPHERES_PER_FIDUCIAL_FIELD, 4), dtype=np.float32)
        self._n = 0

        if centers is not None:
            self.set_spheres(centers, radii, colors)

        self.light_color = tuple(float(x) for x in light_color)
        self.shininess = float(shininess)
        self.k_ambient = float(k_ambient)
        self.k_diffuse = float(k_diffuse)
        self.k_specular = float(k_specular)
        self.visible = bool(visible)

    # -------- Mutation --------

    def set_spheres(
        self,
        centers: np.ndarray,
        radii: np.ndarray | float,
        colors: np.ndarray | None = None,
    ) -> None:
        """Replace the entire sphere set. Centers as (N,3), radii as (N,)
        or scalar, colors as (N,4) RGBA in [0,1] (defaults to white)."""
        c = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
        n = c.shape[0]
        if n > MAX_SPHERES_PER_FIDUCIAL_FIELD:
            # Silent truncation -- log via touch() so callers can detect
            # via mtime change. Could also raise; for now match STEP's
            # forgiving behaviour.
            n = MAX_SPHERES_PER_FIDUCIAL_FIELD
            c = c[:n]
        if np.isscalar(radii):
            r = np.full((n,), float(radii), dtype=np.float32)
        else:
            r = np.asarray(radii, dtype=np.float32).reshape(-1)[:n]
        if colors is None:
            cols = np.tile(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), (n, 1))
        else:
            cols = np.asarray(colors, dtype=np.float32).reshape(-1, 4)[:n]

        self._spheres[:n, :3] = c
        self._spheres[:n, 3] = r
        self._colors[:n] = cols
        self._n = n
        self.touch()

    def update_sphere(self, idx: int, *, center=None, radius=None, color=None) -> None:
        if not 0 <= idx < self._n:
            return
        if center is not None:
            self._spheres[idx, :3] = np.asarray(center, dtype=np.float32)
        if radius is not None:
            self._spheres[idx, 3] = float(radius)
        if color is not None:
            self._colors[idx] = np.asarray(color, dtype=np.float32).reshape(4)
        self.touch()

    def get_center(self, idx: int) -> np.ndarray:
        return self._spheres[idx, :3].copy()

    def get_radius(self, idx: int) -> float:
        return float(self._spheres[idx, 3])

    @property
    def n_spheres(self) -> int:
        return self._n

    # -------- Field protocol: WGSL --------

    def uniform_type(self, slot_idx: int) -> dict:
        p = f"fid{slot_idx}"
        # pygfx sorts uniforms by alignment (big->small) automatically, so
        # no manual padding is needed here. See the note in
        # scene_renderer.make_material_class about why "__pad*" fields
        # would silently desynchronise Python and WGSL offsets.
        return {
            f"{p}_n_spheres":   "f4",   # interpreted as int via i32 cast
            f"{p}_visible":     "f4",
            f"{p}_shininess":   "f4",
            f"{p}_k_ambient":   "f4",
            f"{p}_k_diffuse":   "f4",
            f"{p}_k_specular":  "f4",
            f"{p}_light_color": "4xf4",
            f"{p}_spheres":     f"{MAX_SPHERES_PER_FIDUCIAL_FIELD}*4xf4",
            f"{p}_colors":      f"{MAX_SPHERES_PER_FIDUCIAL_FIELD}*4xf4",
        }

    def get_bindings(self, slot_idx: int) -> list:
        return []

    def sampling_wgsl(self, slot_idx: int) -> str:
        i = slot_idx
        return f"""
fn sample_field_fid{i}(wp: vec3<f32>, ray_dir: vec3<f32>, ray_origin: vec3<f32>) -> FieldSample {{
    var out: FieldSample;
    out.hit = false;
    out.color_srgb = vec3<f32>(0.0);
    out.opacity = 0.0;
    if (u_material.fid{i}_visible < 0.5) {{ return out; }}

    // Sphere centers are stored in the receiver's reference frame; a
    // TransformField attached to this field warps the per-sample world
    // position into that frame before the SDF test, so the spheres
    // appear displaced by the grid.
    let wp_r = transform_point_fid{i}(wp);

    let n = i32(u_material.fid{i}_n_spheres);
    var best_depth = -1.0;       // depth of penetration (radius - distance)
    var best_center = vec3<f32>(0.0);
    var best_radius = 0.0;
    var best_color = vec4<f32>(0.0);
    var found = false;

    // Pick the sphere we're MOST inside of (largest penetration). Cheap
    // for our cap of {MAX_SPHERES_PER_FIDUCIAL_FIELD}; revisit if it grows.
    for (var k = 0; k < n; k = k + 1) {{
        let sp = u_material.fid{i}_spheres[k];
        let r = sp.w;
        if (r <= 0.0) {{ continue; }}
        let to_wp = wp_r - sp.xyz;
        let d = length(to_wp);
        let depth = r - d;       // > 0 -> inside
        if (depth > best_depth) {{
            best_depth = depth;
            best_center = sp.xyz;
            best_radius = r;
            best_color = u_material.fid{i}_colors[k];
            found = true;
        }}
    }}

    if (!found || best_depth <= 0.0) {{ return out; }}

    let to_wp = wp_r - best_center;
    var n_hat = to_wp / max(length(to_wp), 1e-6);
    if (dot(n_hat, -ray_dir) < 0.0) {{ n_hat = -n_hat; }}

    // Plastic Phong: ambient * tint + diffuse * tint + specular * white
    let view_dir = normalize(ray_origin - wp);
    let to_light = view_dir;
    let ldotn = max(dot(to_light, n_hat), 0.0);
    let refl = normalize(2.0 * ldotn * n_hat - to_light);
    let rdotv = max(dot(refl, view_dir), 0.0);

    let base = best_color.rgb;
    let highlight = mix(base, u_material.fid{i}_light_color.rgb, 0.85);
    let lit =
        base * u_material.fid{i}_k_ambient
      + base * (u_material.fid{i}_k_diffuse * ldotn)
      + highlight * (u_material.fid{i}_k_specular * pow(rdotv, u_material.fid{i}_shininess));

    out.hit = true;
    out.color_srgb = clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0));
    // Solid pushpin head; alpha = the per-sphere opacity (best_color.a).
    out.opacity = clamp(best_color.a, 0.0, 1.0);
    return out;
}}
"""

    def tf_wgsl(self, slot_idx: int) -> str:
        i = slot_idx
        return f"""
fn tf_field_fid{i}(s: FieldSample) -> vec4<f32> {{
    return vec4<f32>(s.color_srgb, s.opacity);
}}
"""

    # -------- CPU-side state --------

    def fill_uniforms(self, ub, slot_idx: int) -> None:
        p = f"fid{slot_idx}"
        ub.data[f"{p}_n_spheres"] = np.float32(self._n)
        ub.data[f"{p}_visible"] = np.float32(1.0 if self.visible else 0.0)
        ub.data[f"{p}_shininess"] = np.float32(self.shininess)
        ub.data[f"{p}_k_ambient"] = np.float32(self.k_ambient)
        ub.data[f"{p}_k_diffuse"] = np.float32(self.k_diffuse)
        ub.data[f"{p}_k_specular"] = np.float32(self.k_specular)
        ub.data[f"{p}_light_color"] = np.array(
            [*self.light_color, 1.0], dtype=np.float32)
        ub.data[f"{p}_spheres"] = self._spheres
        ub.data[f"{p}_colors"] = self._colors

    def aabb(self):
        if self._n == 0:
            return None
        c = self._spheres[:self._n, :3]
        r = self._spheres[:self._n, 3:4]
        lo = (c - r).min(axis=0)
        hi = (c + r).max(axis=0)
        return lo.astype(np.float64), hi.astype(np.float64)

    # -------- Picking & drag --------

    def pick(self, ray_origin, ray_dir, camera, viewport_size):
        """Ray vs all spheres. Returns the nearest in front of the camera."""
        if not self.visible or self._n == 0:
            return None
        c = self._spheres[:self._n, :3].astype(np.float64)
        r = self._spheres[:self._n, 3].astype(np.float64)

        oc = c - ray_origin              # (N,3) vector to centres
        # Closest-point t along the ray for each sphere centre
        tc = oc @ ray_dir                # (N,)
        # Distance from each centre to the ray
        proj = ray_origin + tc[:, None] * ray_dir
        d = np.linalg.norm(c - proj, axis=1)

        mask = (d <= r) & (tc > 0.0)     # ray actually crosses + in front
        if not np.any(mask):
            return None

        # Entry parameter for each candidate sphere
        thc = np.sqrt(np.maximum(r[mask]**2 - d[mask]**2, 0.0))
        t_enter = tc[mask] - thc
        idx_in_subset = int(np.argmin(t_enter))
        actual_idx = int(np.flatnonzero(mask)[idx_in_subset])
        t_hit = float(t_enter[idx_in_subset])
        world_pos = ray_origin + t_hit * ray_dir

        return PickHit(
            field=self,
            item_index=actual_idx,
            world_pos=world_pos,
            t=t_hit,
            extra={
                "initial_center": c[actual_idx].copy(),
                "initial_t": t_hit,
            },
        )

    def drag_update(self, hit: PickHit, ray_origin, ray_dir, camera, viewport_size):
        """Move the picked sphere so it stays at the same camera-relative
        depth, sliding along the new view ray. This matches Slicer's
        "grab and drag" feel for markup spheres in the 3D view.
        """
        # Place the centre at the same t along the new ray as the
        # original hit point, then offset by the original (centre - hit)
        # vector so the cursor stays "stuck" to the same point on the
        # sphere's surface.
        original_center = hit.extra["initial_center"]
        original_t = hit.extra["initial_t"]
        # Vector from the original hit-point to the original centre
        original_hit_world = hit.world_pos
        offset_local = original_center - original_hit_world

        new_hit_world = ray_origin + original_t * ray_dir
        new_center = new_hit_world + offset_local
        if np.allclose(new_center, self._spheres[hit.item_index, :3]):
            return False
        self._spheres[hit.item_index, :3] = new_center.astype(np.float32)
        self.touch()
        return True
