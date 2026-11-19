"""
Procedural low-poly pirate heroine (stylized ARPG style) + skeleton.

Run:  blender -b -P tools/blender/hero_model.py
Env:  HERO_RENDER=0 skips the preview renders (fast geometry iteration).

Outputs:
  models/hero.blend
  models/hero.glb
  docs/models/hero_{front,side,back,34}.png

The script starts from an empty scene every run (idempotent).
Coordinates: Blender Z-up, character faces -Y, feet at Z=0, meters.
Character's left = +X, right = -X (.L / .R).
"""
import bpy
import bmesh
import math
import os
from mathutils import Vector, Matrix

ROOT = "/home/user/game"
MODELS_DIR = os.path.join(ROOT, "models")
DOCS_DIR = os.path.join(ROOT, "docs", "models")
DO_RENDER = os.environ.get("HERO_RENDER", "1") == "1"

TWO_PI = 2.0 * math.pi
ARM_ANGLE = math.radians(40.0)   # руки опущены на 40 градусов от горизонтали

# ---------------------------------------------------------------- scene reset
bpy.ops.wm.read_factory_settings(use_empty=True)
SCENE = bpy.context.scene
COLL = SCENE.collection

# ---------------------------------------------------------------- materials
MATERIAL_SPEC = {
    #  name        base color              roughness metallic
    "Skin":    ((0.90, 0.68, 0.52),        0.75, 0.0),
    "Lips":    ((0.62, 0.20, 0.20),        0.70, 0.0),
    "Hair":    ((0.66, 0.20, 0.05),        0.80, 0.0),
    "Hat":     ((0.07, 0.06, 0.055),       0.70, 0.0),
    "Gold":    ((0.95, 0.72, 0.25),        0.60, 0.6),
    "Feather": ((0.70, 0.06, 0.08),        0.85, 0.0),
    "Coat":    ((0.24, 0.025, 0.055),      0.75, 0.0),
    "Shirt":   ((0.92, 0.91, 0.88),        0.85, 0.0),
    "Leather": ((0.22, 0.12, 0.06),        0.75, 0.0),
    "Sash":    ((0.52, 0.06, 0.08),        0.80, 0.0),
    "Pants":   ((0.10, 0.09, 0.11),        0.80, 0.0),
    "Boots":   ((0.33, 0.18, 0.08),        0.70, 0.0),
    "Gloves":  ((0.20, 0.11, 0.06),        0.80, 0.0),
    "Eyes":    ((0.14, 0.07, 0.03),        0.60, 0.0),
}


def make_material(name, rgb, rough, metal):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    mat.diffuse_color = (rgb[0], rgb[1], rgb[2], 1.0)
    return mat


MATS = {n: make_material(n, *spec) for n, spec in MATERIAL_SPEC.items()}


# ---------------------------------------------------------------- geometry helpers
def ring_frame(t, ref):
    r = ref - t * ref.dot(t)
    if r.length < 1e-5:
        alt = Vector((0, 1, 0)) if abs(t.y) < 0.9 else Vector((1, 0, 0))
        r = alt - t * alt.dot(t)
    u = r.normalized()
    v = t.cross(u).normalized()
    return u, v


def sweep(bm, pts, rx, ry, mat=0, sides=8, ref=(1.0, 0.0, 0.0),
          closed=False, caps=True):
    """Loft an elliptical cross-section along a polyline path (frames follow the
    path tangent).  rx: semi-axis along the ref-derived 'u' direction, ry: along 'v'.
    Returns the list of rings."""
    n = len(pts)
    rxs = list(rx) if isinstance(rx, (list, tuple)) else [rx] * n
    rys = list(ry) if isinstance(ry, (list, tuple)) else [ry] * n
    ref = Vector(ref)
    rings = []
    for i, p in enumerate(pts):
        if closed:
            a, b = pts[(i - 1) % n], pts[(i + 1) % n]
        else:
            a, b = pts[max(i - 1, 0)], pts[min(i + 1, n - 1)]
        t = (b - a).normalized()
        u, v = ring_frame(t, ref)
        ring = []
        for k in range(sides):
            th = TWO_PI * k / sides
            co = p + u * (math.cos(th) * rxs[i]) + v * (math.sin(th) * rys[i])
            ring.append(bm.verts.new(co))
        rings.append(ring)
    seg = n if closed else n - 1
    for i in range(seg):
        ra, rb = rings[i], rings[(i + 1) % n]
        for k in range(sides):
            k2 = (k + 1) % sides
            bm.faces.new((ra[k], ra[k2], rb[k2], rb[k])).material_index = mat
    if caps and not closed:
        bm.faces.new(rings[0][::-1]).material_index = mat
        bm.faces.new(rings[-1]).material_index = mat
    return rings


def zloft(bm, specs, sides=16, mat=0, arc=None, wrap=False, caps=False):
    """Stack of horizontal elliptical rings.  specs: (z, cx, cy, rx, ry).
    arc: optional function z -> (a0, a1) in radians for open (partial) rings.
    Rings are NOT frame dependent, so the path may reverse (folded cuffs).
    Faces wind outward when moving upward."""
    rings = []
    for (z, cx, cy, rx, ry) in specs:
        ring = []
        if arc:
            a0, a1 = arc(z)
            for k in range(sides + 1):
                th = a0 + (a1 - a0) * k / sides
                ring.append(bm.verts.new((cx + rx * math.cos(th), cy + ry * math.sin(th), z)))
        else:
            for k in range(sides):
                th = TWO_PI * k / sides
                ring.append(bm.verts.new((cx + rx * math.cos(th), cy + ry * math.sin(th), z)))
        rings.append(ring)
    nseg = len(rings) if wrap else len(rings) - 1
    for i in range(nseg):
        ra, rb = rings[i], rings[(i + 1) % len(rings)]
        for k in range(sides):
            k2 = k + 1 if arc else (k + 1) % sides
            bm.faces.new((ra[k], ra[k2], rb[k2], rb[k])).material_index = mat
    if caps and not arc and not wrap:
        bm.faces.new(rings[0][::-1]).material_index = mat
        bm.faces.new(rings[-1]).material_index = mat
    return rings


def axial_loft(bm, origin, axis, ref, specs, sides=10, mat=0, wrap=False, caps=True):
    """Loft along a STRAIGHT axis with one fixed frame.  specs: (t, rx, ry[, du, dv])
    where rx is along u (ref projected), ry along v, du/dv are centre offsets."""
    axis = Vector(axis).normalized()
    u, v = ring_frame(axis, Vector(ref))
    origin = Vector(origin)
    rings = []
    for sp in specs:
        t, rx, ry = sp[0], sp[1], sp[2]
        du = sp[3] if len(sp) > 3 else 0.0
        dv = sp[4] if len(sp) > 4 else 0.0
        c = origin + axis * t + u * du + v * dv
        ring = []
        for k in range(sides):
            th = TWO_PI * k / sides
            ring.append(bm.verts.new(c + u * (math.cos(th) * rx) + v * (math.sin(th) * ry)))
        rings.append(ring)
    nseg = len(rings) if wrap else len(rings) - 1
    for i in range(nseg):
        ra, rb = rings[i], rings[(i + 1) % len(rings)]
        for k in range(sides):
            k2 = (k + 1) % sides
            bm.faces.new((ra[k], ra[k2], rb[k2], rb[k])).material_index = mat
    if caps and not wrap:
        bm.faces.new(rings[0][::-1]).material_index = mat
        bm.faces.new(rings[-1]).material_index = mat
    return rings


def hermite_table(tab):
    """tab: ascending list of (z, v1, v2, ...); returns f(z) -> list of values
    (smooth cubic Hermite, finite-difference tangents)."""
    n = len(tab)
    zs = [r[0] for r in tab]
    m = len(tab[0]) - 1

    def tang(i, j):
        if i == 0:
            return (tab[1][j + 1] - tab[0][j + 1]) / (zs[1] - zs[0])
        if i == n - 1:
            return (tab[-1][j + 1] - tab[-2][j + 1]) / (zs[-1] - zs[-2])
        return (tab[i + 1][j + 1] - tab[i - 1][j + 1]) / (zs[i + 1] - zs[i - 1])

    def f(z):
        z = min(max(z, zs[0]), zs[-1])
        i = 0
        while i < n - 2 and z > zs[i + 1]:
            i += 1
        h = zs[i + 1] - zs[i]
        t = (z - zs[i]) / h
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        return [h00 * tab[i][j + 1] + h10 * h * tang(i, j)
                + h01 * tab[i + 1][j + 1] + h11 * h * tang(i + 1, j) for j in range(m)]
    return f


def levels(z0, z1, step):
    n = max(1, int(round(abs(z1 - z0) / step)))
    return [z0 + (z1 - z0) * i / n for i in range(n + 1)]


def specs_from(tab, zs, cx=0.0):
    """table columns after z: rx, ry, cy  ->  zloft specs"""
    f = hermite_table(tab)
    out = []
    for z in zs:
        v = f(z)
        out.append((z, cx, v[2] if len(v) > 2 else 0.0, v[0], v[1]))
    return out


def catmull(pts, sub=3):
    """Catmull-Rom resample of a Vector polyline (sub points per span)."""
    out = []
    n = len(pts)
    for i in range(n - 1):
        p0, p1, p2, p3 = pts[max(i - 1, 0)], pts[i], pts[i + 1], pts[min(i + 2, n - 1)]
        for j in range(sub):
            t = j / sub
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1].copy())
    return out


def smoothstep(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def ellipse_pts(center, rx, ry, n=24):
    c = Vector(center)
    return [c + Vector((rx * math.cos(TWO_PI * i / n), ry * math.sin(TWO_PI * i / n), 0.0))
            for i in range(n)]


def circle_pts(center, normal, radius, n=12):
    c = Vector(center)
    nrm = Vector(normal).normalized()
    a = nrm.orthogonal().normalized()
    b = nrm.cross(a).normalized()
    return [c + (a * math.cos(TWO_PI * i / n) + b * math.sin(TWO_PI * i / n)) * radius
            for i in range(n)]


def ellipsoid_bm(center, radii, segs=8, rings=5, taper=None, rot=None):
    """UV sphere scaled to an ellipsoid; optional taper on unit-sphere coords,
    optional rotation (3x3 Matrix) about the centre."""
    tmp = bmesh.new()
    bmesh.ops.create_uvsphere(tmp, u_segments=segs, v_segments=rings,
                              radius=1.0, matrix=Matrix.Identity(4))
    for v in tmp.verts:
        u = v.co.copy()
        if taper is not None:
            u = taper(u)
        off = Vector((u.x * radii[0], u.y * radii[1], u.z * radii[2]))
        if rot is not None:
            off = rot @ off
        v.co = Vector(center) + off
    return tmp


def blob(bm, center, radii, mat=0, segs=8, rings=5, rot=None):
    append(bm, ellipsoid_bm(center, radii, segs, rings, rot=rot), mat)


def box(bm, center, size, mat=0, rot_z=0.0):
    before = set(bm.faces)
    m = (Matrix.Translation(center) @ Matrix.Rotation(rot_z, 4, 'Z')
         @ Matrix.Diagonal((size[0], size[1], size[2], 1.0)))
    bmesh.ops.create_cube(bm, size=1.0, matrix=m)
    for f in bm.faces:
        if f not in before:
            f.material_index = mat


def xform(bm, mtx):
    for v in bm.verts:
        v.co = mtx @ v.co


def subsurf(src, levels_=1):
    """Apply a Subdivision Surface modifier through a temporary object and
    return the resulting mesh as a new bmesh (equivalent to modifier apply)."""
    return _apply_mod(src, 'SUBSURF', levels=levels_, render_levels=levels_)


def solidify(src, thickness, offset=0.0):
    return _apply_mod(src, 'SOLIDIFY', thickness=thickness, offset=offset,
                      use_even_offset=True, use_rim=True)


def _apply_mod(src, mtype, **kw):
    me = bpy.data.meshes.new("_tmp_mod")
    src.to_mesh(me)
    ob = bpy.data.objects.new("_tmp_mod", me)
    COLL.objects.link(ob)
    mod = ob.modifiers.new("M", mtype)
    for k, v in kw.items():
        setattr(mod, k, v)
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = ob.evaluated_get(dg)
    emesh = ev.to_mesh()
    out = bmesh.new()
    out.from_mesh(emesh)
    ev.to_mesh_clear()
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.meshes.remove(me)
    src.free()
    return out


def append(dst, src, mat):
    src.verts.ensure_lookup_table()
    src.verts.index_update()
    vm = [dst.verts.new(v.co) for v in src.verts]
    for f in src.faces:
        try:
            nf = dst.faces.new([vm[v.index] for v in f.verts])
            nf.material_index = mat if mat is not None else f.material_index
        except ValueError:
            pass
    src.free()


def quad_out(bm, verts, out_dir, mat=0):
    """Create a face and flip it so its normal points along out_dir."""
    try:
        f = bm.faces.new(verts)
    except ValueError:
        return None
    f.normal_update()
    if f.normal.dot(Vector(out_dir)) < 0:
        f.normal_flip()
    f.material_index = mat
    return f


def finalize(name, bm, mat_names, recalc=True, weld=True):
    if weld:
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-5)
    if recalc:
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for mn in mat_names:
        me.materials.append(MATS[mn])
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    COLL.objects.link(ob)
    return ob


# ---------------------------------------------------------------- body plan
SHOULDER_X = 0.152
SHOULDER_Z = 1.30
ELBOW_FLEX = {1: math.radians(10.0), -1: math.radians(24.0)}   # forearm bent toward -Y
WRIST_FLEX = {1: math.radians(18.0), -1: math.radians(32.0)}
LEG_X = 0.092


def arm_joints(s):
    """s=+1 -> character's left (+X), s=-1 -> right (-X).
    Returns S (shoulder), E (elbow), W (wrist), T (hand tip), d2 (forearm dir), d3 (hand dir)."""
    c, sn = math.cos(ARM_ANGLE), math.sin(ARM_ANGLE)
    d1 = Vector((s * c, 0.0, -sn))
    fwd = Vector((0.0, -1.0, 0.0))
    f2, f3 = ELBOW_FLEX[s], WRIST_FLEX[s]
    d2 = (d1 * math.cos(f2) + fwd * math.sin(f2)).normalized()
    d3 = (d1 * math.cos(f3) + fwd * math.sin(f3)).normalized()
    S = Vector((s * SHOULDER_X, 0.0, SHOULDER_Z))
    E = S + d1 * 0.28
    W = E + d2 * 0.25
    T = W + d3 * 0.15
    return S, E, W, T, d2, d3


def arm_path(s, n_up=4, n_fore=4):
    """Dense polyline from S to W through E with parameter t (0..1)."""
    S, E, W, T, d2, d3 = arm_joints(s)
    pts, ts = [], []
    L1, L2 = (E - S).length, (W - E).length
    tot = L1 + L2
    for i in range(n_up):
        a = i / n_up
        pts.append(S.lerp(E, a))
        ts.append(a * L1 / tot)
    for i in range(n_fore + 1):
        a = i / n_fore
        pts.append(E.lerp(W, a))
        ts.append((L1 + a * L2) / tot)
    return pts, ts


def limb_radii(ts, table):
    f = hermite_table(table)
    return [f(t)[0] for t in ts]


# ---------------------------------------------------------------- BODY (skin)
HEAD_C = (0.0, 0.0, 1.53)
HEAD_R = (0.083, 0.098, 0.112)

TORSO_TAB = [  # z, rx, ry, cy
    (0.80, 0.135, 0.090, 0.000),
    (0.90, 0.150, 0.098, 0.000),
    (1.07, 0.104, 0.074, 0.000),
    (1.16, 0.114, 0.080, -0.002),
    (1.23, 0.134, 0.088, -0.007),
    (1.29, 0.146, 0.082, -0.002),
    (1.34, 0.090, 0.062, 0.000),
    (1.375, 0.052, 0.052, 0.000),
]
SKIN_ARM_TAB = [(0.0, 0.043), (0.10, 0.045), (0.30, 0.040), (0.527, 0.031),
                (0.75, 0.027), (1.0, 0.021)]
SLEEVE_TAB = [(0.0, 0.057), (0.10, 0.059), (0.30, 0.053), (0.527, 0.046),
              (0.75, 0.042), (0.95, 0.040)]
SKIN_LEG_TAB = [(0.86, 0.068, 0.070), (0.74, 0.060, 0.062), (0.62, 0.046, 0.048),
                (0.48, 0.040, 0.042), (0.36, 0.040, 0.042), (0.22, 0.034, 0.036),
                (0.09, 0.030, 0.032)]


def head_deform(u):
    """Unit-sphere point -> displaced head point (face shaping)."""
    rx, ry, rz = HEAD_R
    cx, cy, cz = HEAD_C
    x, y, z = u.x * rx, u.y * ry, u.z * rz
    if u.z < 0.0:
        k = -u.z
        x *= 1.0 - 0.28 * k ** 1.6
        y *= 1.0 - 0.12 * k
        if u.y < 0:
            k2 = max(0.0, k - 0.25) / 0.75
            y -= 0.024 * k2 ** 1.3 * (-u.y) * (1.0 - min(1.0, abs(u.x) * 1.2))
    p = Vector((cx + x, cy + y, cz + z))
    c = Vector(HEAD_C)
    dirv = (p - c).normalized()
    disp = 0.0
    for sx in (1, -1):
        for (gx, gy, gz, sig, amp) in (
                (0.056, -0.066, 1.515, 0.024, 0.0075),    # cheekbones
                (0.036, -0.088, 1.586, 0.020, 0.0045),    # brow ridge
                (0.034, -0.090, 1.549, 0.013, -0.0025),   # eye sockets
                (0.040, -0.060, 1.448, 0.022, -0.0040)):  # hollow under cheek
            d2 = (p.x - sx * gx) ** 2 + (p.y - gy) ** 2 + (p.z - gz) ** 2
            disp += amp * math.exp(-d2 / (2 * sig * sig))
    return p + dirv * disp


def build_head_into(bm):
    tmp = bmesh.new()
    bmesh.ops.create_uvsphere(tmp, u_segments=24, v_segments=15, radius=1.0,
                              matrix=Matrix.Identity(4))
    for v in tmp.verts:
        v.co = head_deform(v.co.copy())
    append(bm, tmp, 0)


def eye_rot(s, deg=9.0):
    """Rotation about Y that lowers the OUTER eye corner (outer = sign s)."""
    return Matrix.Rotation(math.radians(deg) * s, 3, 'Y')


def build_body():
    bm = bmesh.new()
    build_head_into(bm)
    # --- nose (small): bridge, tip, tucked nostrils
    blob(bm, (0.0, -0.0905, 1.520), (0.0070, 0.0105, 0.021), 0, 8, 5,
         rot=Matrix.Rotation(math.radians(-14), 3, 'X'))
    blob(bm, (0.0, -0.1005, 1.4965), (0.0098, 0.0085, 0.0085), 0, 8, 5)
    for sx in (1, -1):
        # eyelids (skin): upper lid heavy, lower lid thin
        R = eye_rot(sx)
        blob(bm, (sx * 0.0345, -0.0815, 1.5620), (0.0235, 0.0130, 0.0070), 0, 8, 5, rot=R)
        blob(bm, (sx * 0.0345, -0.0830, 1.5320), (0.0215, 0.0105, 0.0035), 0, 8, 4, rot=R)
        # ears
        blob(bm, (sx * 0.0795, 0.006, 1.522), (0.0075, 0.012, 0.020), 0, 6, 4)
    # chin
    # lips (second material slot)
    smirk = Matrix.Rotation(math.radians(7.0), 3, 'Y')      # left corner of the mouth lifted
    blob(bm, (0.0, -0.0690, 1.4585), (0.0205, 0.0095, 0.0058), 1, 8, 5, rot=smirk)
    blob(bm, (0.0, -0.0680, 1.4460), (0.0190, 0.0100, 0.0078), 1, 8, 5, rot=smirk)
    # neck
    zloft(bm, [(1.35, 0, 0.003, 0.046, 0.046), (1.40, 0, 0.004, 0.041, 0.041),
               (1.46, 0, 0.006, 0.040, 0.040)], sides=10, caps=True)
    # torso
    zs = levels(0.80, 1.375, 0.05)
    zloft(bm, specs_from(TORSO_TAB, zs), sides=20, caps=True)
    # arms
    for s in (1, -1):
        pts, ts = arm_path(s)
        rr = limb_radii(ts, SKIN_ARM_TAB)
        sweep(bm, pts, rr, rr, mat=0, sides=10)
        # legs
        lz = levels(0.09, 0.86, 0.11)
        f = hermite_table(sorted(SKIN_LEG_TAB, key=lambda r: r[0]))
        sp = [(z, s * LEG_X, 0.004 if z < 0.2 else 0.0, f(z)[0], f(z)[1]) for z in lz]
        zloft(bm, sp, sides=8, caps=True)
    return finalize("Body", bm, ["Skin", "Lips"])


# ---------------------------------------------------------------- EYES (+ brows live in Hair)
def build_eyes():
    bm = bmesh.new()
    for sx in (1, -1):
        R = eye_rot(sx)
        c0 = Vector((sx * 0.0345, -0.0790, 1.5470))
        blob(bm, c0, (0.0215, 0.0115, 0.0185), 1, 8, 5, rot=R)
        blob(bm, c0 + R @ Vector((sx * 0.0008, -0.0078, 0.0010)), (0.0140, 0.0062, 0.0150), 0, 8, 5, rot=R)
        # dark upper lash line
        blob(bm, c0 + R @ Vector((0.0, -0.0075, 0.0150)), (0.0250, 0.0095, 0.0030), 0, 8, 4, rot=R)
    return finalize("Eyes", bm, ["Eyes", "Shirt"])


def brows_into(bm):
    rx_, ry_, rz_ = HEAD_R
    for sx in (1, -1):
        xs = [0.010, 0.026, 0.042, 0.056]
        zs = [1.5760, 1.5835, 1.5840, 1.5750]
        pts = []
        for x, z in zip(xs, zs):
            q = 1.0 - (x / rx_) ** 2 - ((z - HEAD_C[2]) / rz_) ** 2
            y = -ry_ * math.sqrt(max(q, 0.05)) - 0.0075
            pts.append(Vector((sx * x, y, z)))
        sweep(bm, pts, [0.0042, 0.0058, 0.0058, 0.0030], [0.0038, 0.0045, 0.0045, 0.0028],
              mat=0, sides=6, ref=(0.0, 0.0, 1.0))


# ---------------------------------------------------------------- HAIR
def build_hair():
    import random
    rnd = random.Random(11)
    bm = bmesh.new()
    # --- cap over the skull; the face area is cut out
    cap = ellipsoid_bm((0.0, 0.030, 1.572), (0.106, 0.116, 0.104), segs=18, rings=11)
    dele = []
    for f in cap.faces:
        c = f.calc_center_median()
        if c.y < -0.030 and c.z < 1.648:
            dele.append(f)
        elif c.z < 1.50 and c.y < 0.06:
            dele.append(f)
    bmesh.ops.delete(cap, geom=dele, context='FACES')
    append(bm, cap, 0)
    # --- big back mass (closed, rippled), hangs to the shoulder blades
    MA, MB, MZ, MH = 0.150, 0.088, 1.400, 0.210
    tmp = bmesh.new()
    bmesh.ops.create_uvsphere(tmp, u_segments=16, v_segments=10, radius=1.0,
                              matrix=Matrix.Identity(4))
    for v in tmp.verts:
        u = v.co.copy()
        th = math.atan2(u.y, u.x)
        rip = 1.0 + 0.07 * math.sin(5 * th + 11.0 * u.z) * (1 - u.z * u.z)
        low = 1.0 + 0.10 * max(0.0, -u.z)
        v.co = Vector((u.x * MA * rip * low, 0.075 + u.y * MB * rip * low, MZ + u.z * MH))
    append(bm, tmp, 0)
    # --- thick wavy curls hanging down the back and sides (big readable strands)
    NS = 11
    wid = [0.024, 0.036, 0.040, 0.040, 0.038, 0.036, 0.032, 0.025, 0.013]
    thk = [0.016, 0.020, 0.022, 0.022, 0.021, 0.020, 0.018, 0.015, 0.009]
    for i in range(NS):
        th = math.radians(-20 + 220 * i / (NS - 1))
        z_end = 1.205 + 0.07 * rnd.random() + (0.02 if i % 2 else 0.0)
        pts = []
        ntp = len(wid) - 1
        for j in range(ntp + 1):
            t = j / ntp
            z = 1.545 + (z_end - 1.545) * t
            uz = (z - MZ) / MH
            prof = math.sqrt(max(0.0, 1.0 - uz * uz)) * (1.0 + 0.10 * max(0.0, -uz))
            wv = 0.011 * math.sin(2 * math.pi * 1.5 * t + i * 1.7)
            out = 1.0 + (0.08 if i % 2 else 0.03) + 0.10 * t * t
            x = MA * prof * math.cos(th) * out - math.sin(th) * wv
            y = 0.075 + MB * prof * math.sin(th) * out + math.cos(th) * wv
            pts.append(Vector((x, y, z)))
        tan = Vector((-math.sin(th), math.cos(th), 0.0))
        sweep(bm, pts, wid, thk, mat=0, sides=6, ref=tan)
        e = pts[-1]
        blob(bm, (e.x, e.y, e.z - 0.006), (0.028, 0.026, 0.028), 0, 6, 4)
    for sx in (1, -1):
        blob(bm, (sx * 0.092, 0.030, 1.545), (0.030, 0.045, 0.048), 0, 6, 4)
    # --- wavy locks framing the face (outside of the cheeks, never over the eyes)
    for sx in (1, -1):
        for (x0, y0, zt, zb, amp, ph) in ((0.100, -0.030, 1.600, 1.300, 0.012, 0.0),
                                          (0.108, -0.006, 1.570, 1.265, 0.014, 2.2)):
            pts = []
            n = 10
            for j in range(n + 1):
                t = j / n
                z = zt + (zb - zt) * t
                x = x0 + amp * math.sin(2 * math.pi * 1.4 * t + ph) + 0.006 * t
                y = y0 + 0.012 * t + 0.006 * math.sin(2 * math.pi * 2.0 * t)
                pts.append(Vector((sx * x, y, z)))
            wid = [0.016, 0.025, 0.029, 0.028, 0.027, 0.026, 0.024, 0.021, 0.017, 0.011, 0.004]
            thk = [0.012, 0.016, 0.018, 0.018, 0.018, 0.017, 0.016, 0.014, 0.012, 0.008, 0.004]
            sweep(bm, pts, wid, thk, mat=0, sides=6, ref=(1.0, 0.0, 0.0))
            e = pts[-1]
            blob(bm, (e.x, e.y, e.z - 0.004), (0.020, 0.019, 0.020), 0, 6, 4)
    brows_into(bm)
    return finalize("Hair", bm, ["Hair"], recalc=False)


# ---------------------------------------------------------------- BANDANA (under the hat)
def build_bandana():
    bm = bmesh.new()
    specs = [(1.592, 0.0, 0.030, 0.1085, 0.1165), (1.626, 0.0, 0.036, 0.0975, 0.1005),
             (1.626, 0.0, 0.036, 0.0925, 0.0955), (1.592, 0.0, 0.030, 0.1035, 0.1115)]
    zloft(bm, specs, sides=24, mat=0, wrap=True)
    # knot at the back, two tails
    blob(bm, (0.0, 0.158, 1.600), (0.020, 0.016, 0.018), 0, 8, 5)
    for sx, ang in ((1, 12), (-1, -12)):
        pts = [Vector((sx * 0.012, 0.168, 1.585)), Vector((sx * 0.040, 0.182, 1.545)),
               Vector((sx * 0.052, 0.185, 1.500)), Vector((sx * 0.062, 0.186, 1.455))]
        sweep(bm, pts, [0.014, 0.018, 0.020, 0.015], [0.005] * 4, mat=0, sides=6,
              ref=(1.0, 0.0, 0.0))
    return finalize("Bandana", bm, ["Sash"])


# ---------------------------------------------------------------- HAT (tricorn)
def build_hat():
    bm = bmesh.new()
    cy = 0.020
    z0 = 1.606
    # crown
    crown = [(z0, 0.0, cy, 0.128, 0.118), (1.650, 0.0, cy, 0.131, 0.121),
             (1.700, 0.0, cy, 0.128, 0.118), (1.735, 0.0, cy, 0.118, 0.109),
             (1.752, 0.0, cy, 0.098, 0.090), (1.756, 0.0, cy, 0.060, 0.055)]
    zloft(bm, crown, sides=20, mat=0, caps=True)
    # gold band at the crown base
    zloft(bm, [(1.608, 0.0, cy, 0.1330, 0.1235), (1.636, 0.0, cy, 0.1345, 0.1250),
               (1.636, 0.0, cy, 0.1270, 0.1175), (1.608, 0.0, cy, 0.1270, 0.1175)],
          sides=20, mat=1, wrap=True)
    # brim: three pointed lobes, the sides between them folded up against the crown
    NA = 40
    corners = [math.radians(a_) for a_ in (-90.0, 30.0, 150.0)]
    ring_f = [0.0, 0.33, 0.68, 1.0]
    brim = bmesh.new()
    rings = []
    outer_pts = []
    for fi, f in enumerate(ring_f):
        ring = []
        for k in range(NA):
            phi = TWO_PI * k / NA
            d = min(abs((phi - c + math.pi) % TWO_PI - math.pi) for c in corners)
            w = smoothstep(0.08, 0.72, d / math.radians(60.0))
            R_out = 0.315 - (0.315 - 0.150) * w
            R_in = 0.128
            r = R_in + (R_out - R_in) * f
            lift = 0.078 * w * f - 0.014 * (1.0 - w) * f
            p = Vector((r * math.cos(phi), cy + r * math.sin(phi) * 0.93, z0 + 0.006 + lift))
            ring.append(brim.verts.new(p))
            if fi == len(ring_f) - 1:
                outer_pts.append(p.copy())
        rings.append(ring)
    for i in range(len(rings) - 1):
        for k in range(NA):
            k2 = (k + 1) % NA
            brim.faces.new((rings[i][k], rings[i + 1][k], rings[i + 1][k2], rings[i][k2]))
    brim = solidify(brim, 0.016, 0.0)
    append(bm, brim, 0)
    # gold piping along the brim edge
    sweep(bm, outer_pts, 0.0068, 0.0068, mat=1, sides=4, ref=(0, 0, 1), closed=True)
    # feather: big flat plumes curling up and back on the character's left side
    def plume(pts, wid, ref, thick=0.006):
        sub = 3
        dp = catmull(pts, sub)
        dw = []
        for i in range(len(wid) - 1):
            for j in range(sub):
                dw.append(wid[i] + (wid[i + 1] - wid[i]) * j / sub)
        dw.append(wid[-1])
        sweep(bm, dp, dw, [thick] * len(dp), mat=2, sides=6, ref=ref)
    fp = [Vector((0.108, 0.040, 1.640)), Vector((0.168, 0.026, 1.725)),
          Vector((0.212, 0.032, 1.815)), Vector((0.214, 0.075, 1.890)),
          Vector((0.182, 0.140, 1.925)), Vector((0.132, 0.200, 1.910)),
          Vector((0.085, 0.240, 1.860))]
    plume(fp, [0.014, 0.036, 0.052, 0.058, 0.052, 0.036, 0.010], (1.0, -0.7, 0.0))
    fp2 = [Vector((0.112, 0.020, 1.648)), Vector((0.180, 0.000, 1.712)),
           Vector((0.236, 0.010, 1.775)), Vector((0.262, 0.045, 1.825)),
           Vector((0.262, 0.095, 1.850)), Vector((0.236, 0.140, 1.842))]
    plume(fp2, [0.012, 0.030, 0.042, 0.044, 0.034, 0.008], (1.0, -0.2, 0.0))
    sweep(bm, catmull(fp[:6], 2), 0.0035, 0.0035, mat=1, sides=4, ref=(0, 1, 0))
    # tilt the hat: lower on the character's right, front slightly up
    pivot = Vector((0.0, cy, 1.606))
    M = (Matrix.Translation(pivot) @ Matrix.Rotation(math.radians(-11.0), 4, 'Y')
         @ Matrix.Rotation(math.radians(-7.0), 4, 'X') @ Matrix.Translation(-pivot))
    xform(bm, M)
    return finalize("Hat", bm, ["Hat", "Gold", "Feather"], recalc=False)


# ---------------------------------------------------------------- COAT
COAT_TAB = [  # z, rx, ry, cy, gap(deg)
    (0.66, 0.240, 0.165, 0.006, 19.0),
    (0.76, 0.208, 0.146, 0.005, 16.0),
    (0.86, 0.190, 0.130, 0.004, 13.0),
    (0.96, 0.170, 0.116, 0.002, 11.0),
    (1.06, 0.128, 0.090, 0.000, 9.0),
    (1.14, 0.137, 0.097, -0.004, 16.0),
    (1.22, 0.156, 0.108, -0.010, 27.0),
    (1.29, 0.166, 0.100, -0.004, 37.0),
    (1.34, 0.122, 0.084, 0.000, 44.0),
    (1.385, 0.078, 0.066, 0.004, 52.0),
    (1.425, 0.086, 0.072, 0.008, 58.0),
]
COAT_F = hermite_table(COAT_TAB)
COAT_SIDES = 28
LAPEL_W = [(1.10, 0.0), (1.15, 0.020), (1.22, 0.038), (1.29, 0.045), (1.335, 0.040),
           (1.385, 0.030), (1.425, 0.020)]


def coat_arc(z):
    g = math.radians(COAT_F(z)[3])
    return (-math.pi / 2 + g, 1.5 * math.pi - g)


def coat_point(z, theta, extra=0.0):
    v = COAT_F(z)
    rx, ry, cy = v[0] + extra, v[1] + extra, v[2]
    return Vector((rx * math.cos(theta), cy + ry * math.sin(theta), z))


def coat_normal(z, theta):
    v = COAT_F(z)
    n = Vector((math.cos(theta) / v[0], math.sin(theta) / v[1], 0.0))
    return n.normalized()


def tube_path(bm, pts, r, mat=1, sides=5):
    if len(pts) >= 2:
        sweep(bm, pts, r, r, mat=mat, sides=sides, ref=(0, 0, 1))


def build_coat():
    bm = bmesh.new()
    zs = levels(0.66, 1.425, 0.055)
    specs = specs_from([(t[0], t[1], t[2], t[3]) for t in COAT_TAB], zs)
    rings = zloft(bm, specs, sides=COAT_SIDES, mat=0, arc=coat_arc)
    nz = len(zs)
    C = COAT_SIDES // 2
    # back vent: remove the two centre columns below z=0.87
    vent_top = None
    for i in range(nz - 1):
        if zs[i + 1] <= 0.87:
            vent_top = i + 1
    kill = []
    vert_set = {}
    for i in range(nz):
        for k, v in enumerate(rings[i]):
            vert_set[v] = (i, k)
    for f in list(bm.faces):
        iis = [vert_set[v][0] for v in f.verts]
        kks = [vert_set[v][1] for v in f.verts]
        if max(iis) <= vent_top and min(kks) >= C - 1 and max(kks) <= C + 1:
            kill.append(f)
    bmesh.ops.delete(bm, geom=kill, context='FACES_ONLY')

    # ---- lapels (flat flaps folded outward from the front edges)
    lw = hermite_table(LAPEL_W)
    lap_edges = {}
    for side, kidx in ((0, 0), (1, COAT_SIDES)):
        Ls = []
        Es = []
        for i in range(nz):
            z = zs[i]
            if z < 1.095:
                continue
            e = rings[i][kidx]
            a0, a1 = coat_arc(z)
            th = a0 if side == 0 else a1
            tan = Vector((-math.sin(th) * COAT_F(z)[0], math.cos(th) * COAT_F(z)[1], 0.0)).normalized()
            if side == 1:
                tan = -tan
            N = coat_normal(z, th)
            w = lw(z)[0]
            L = e.co + tan * w + N * 0.007
            Ls.append(bm.verts.new(L))
            Es.append(e)
        for j in range(len(Es) - 1):
            n = coat_normal(zs[nz - len(Es) + j], coat_arc(zs[nz - len(Es) + j])[side])
            quad_out(bm, (Es[j], Es[j + 1], Ls[j + 1], Ls[j]), n, 0)
        lap_edges[side] = [l.co.copy() for l in Ls]
    # ---- gold piping: front edges, lapel edges, hem, collar rim, vent
    for side, kidx in ((0, 0), (1, COAT_SIDES)):
        pts = [Vector(rings[i][kidx].co) + coat_normal(zs[i], coat_arc(zs[i])[side]) * 0.003
               for i in range(nz)]
        tube_path(bm, pts, 0.0058)
        lp = lap_edges[side]
        tube_path(bm, [p + Vector((0, 0, 0)) for p in lp], 0.0052)
    hem0 = [Vector(v.co) for v in rings[0]]
    tube_path(bm, hem0[:C - 1 + 1], 0.0075)
    tube_path(bm, hem0[C + 1:], 0.0075)
    tube_path(bm, [Vector(v.co) for v in rings[-1]], 0.0055)
    for kk in (C - 1, C + 1):
        tube_path(bm, [Vector(rings[i][kk].co) for i in range(vent_top + 1)], 0.0050)
    # ---- sleeves
    for s in (1, -1):
        pts, ts = arm_path(s)
        rr = limb_radii(ts, SLEEVE_TAB)
        # stop the sleeve a little before the wrist (the glove gauntlet overlaps it)
        n_keep = len(pts) - 1
        sweep(bm, pts[:n_keep], rr[:n_keep], rr[:n_keep], mat=0, sides=12, caps=False)
    # ---- buttons: 3 per front panel, on the skirt
    for side in (0, 1):
        for z in (0.735, 0.800, 0.865):
            a0, a1 = coat_arc(z)
            th = (a0 + math.radians(8.0)) if side == 0 else (a1 - math.radians(8.0))
            p = coat_point(z, th, 0.006)
            box(bm, p, (0.016, 0.016, 0.016), mat=1, rot_z=th + math.pi / 2)
    return finalize("Coat", bm, ["Coat", "Gold"], recalc=False)


# ---------------------------------------------------------------- SHIRT
SHIRT_TAB = [  # z, rx, ry, cy, gap
    (0.92, 0.140, 0.092, 0.0, 0.0),
    (1.06, 0.116, 0.082, 0.0, 0.0),
    (1.16, 0.125, 0.088, -0.003, 0.0),
    (1.22, 0.144, 0.098, -0.009, 6.0),
    (1.28, 0.152, 0.092, -0.004, 14.0),
    (1.335, 0.104, 0.074, 0.000, 22.0),
    (1.380, 0.064, 0.060, 0.003, 30.0),
    (1.430, 0.080, 0.070, 0.006, 44.0),
]
SHIRT_F = hermite_table(SHIRT_TAB)


def build_shirt():
    bm = bmesh.new()

    def arc(z):
        g = math.radians(max(0.0, SHIRT_F(z)[3]))
        return (-math.pi / 2 + g, 1.5 * math.pi - g)
    zs = levels(0.92, 1.43, 0.045)
    specs = specs_from([(t[0], t[1], t[2], t[3]) for t in SHIRT_TAB], zs)
    zloft(bm, specs, sides=24, mat=0, arc=arc)
    return finalize("Shirt", bm, ["Shirt"], recalc=False)


# ---------------------------------------------------------------- BELT and SASH
def build_belt():
    bm = bmesh.new()
    z0, z1 = 1.058, 1.098
    c0, c1 = COAT_F(z0), COAT_F(z1)
    rx, ry = max(c0[0], c1[0]) + 0.008, max(c0[1], c1[1]) + 0.008
    cy = c0[2]
    t = 0.011
    specs = [(z0, 0, cy, rx + t, ry + t), (z1, 0, cy, rx + t, ry + t),
             (z1, 0, cy, rx, ry), (z0, 0, cy, rx, ry)]
    zloft(bm, specs, sides=26, mat=0, wrap=True)
    yf = cy - (ry + t)
    zc = 0.5 * (z0 + z1)
    # big gold buckle: frame + dark inset + prong
    box(bm, (0.0, yf - 0.006, zc), (0.080, 0.018, 0.068), mat=1)
    box(bm, (0.0, yf - 0.0145, zc), (0.050, 0.010, 0.038), mat=0)
    box(bm, (0.0, yf - 0.0215, zc), (0.060, 0.008, 0.010), mat=1)
    return finalize("Belt", bm, ["Leather", "Gold"], recalc=False)


def build_sash():
    bm = bmesh.new()
    zs = [0.980, 1.008, 1.036, 1.064]
    wid = [0.0, 0.004, 0.006, 0.002]
    specs, specs_in = [], []
    for z, w in zip(zs, wid):
        c = COAT_F(z)
        specs.append((z, 0, c[2], c[0] + 0.004 + w, c[1] + 0.004 + w))
        specs_in.append((z, 0, c[2], c[0] + 0.001, c[1] + 0.001))
    zloft(bm, specs + specs_in[::-1], sides=24, mat=0, wrap=True)
    # knot on the left hip (front) + two flat tails
    th = math.radians(-62.0)
    z_k = 1.020
    c = COAT_F(z_k)
    rr = 1.02
    kp = Vector((c[0] * rr * math.cos(th), c[2] + c[1] * rr * math.sin(th), z_k)) + Vector((0.006, 0, 0))
    blob(bm, kp, (0.030, 0.026, 0.028), 0, 8, 5)
    tan = Vector((-math.sin(th) * c[0], math.cos(th) * c[1], 0)).normalized()
    out = Vector((math.cos(th) / c[0], math.sin(th) / c[1], 0)).normalized()
    for (dz, side, off) in ((0.34, 0.018, 0.020), (0.27, -0.022, 0.030)):
        pts = []
        for j in range(6):
            t = j / 5
            pts.append(kp + out * (0.020 + off * t + 0.012 * math.sin(5.0 * t)) - Vector((0, 0, 0.015 + dz * t))
                       + tan * (side * t))
        sweep(bm, pts, [0.020, 0.028, 0.032, 0.035, 0.036, 0.034],
              [0.0065] * 6, mat=0, sides=6, ref=tan)
    return finalize("Sash", bm, ["Sash"], recalc=False)


# ---------------------------------------------------------------- PANTS, BOOTS, GLOVES
PANTS_HIP = [(0.78, 0.140, 0.100, 0.0), (0.88, 0.158, 0.108, 0.0),
             (0.96, 0.146, 0.100, 0.0), (1.05, 0.118, 0.088, 0.0)]
PANTS_LEG = [(0.86, 0.078, 0.080), (0.78, 0.075, 0.077), (0.70, 0.066, 0.069),
             (0.60, 0.054, 0.057), (0.50, 0.049, 0.051), (0.44, 0.048, 0.050)]


def build_pants():
    bm = bmesh.new()
    zloft(bm, specs_from(PANTS_HIP, levels(0.78, 1.05, 0.045)), sides=22, mat=0)
    f = hermite_table(sorted(PANTS_LEG, key=lambda r: r[0]))
    for s in (1, -1):
        sp = [(z, s * LEG_X, 0.005 if z < 0.6 else 0.0, f(z)[0], f(z)[1]) for z in levels(0.44, 0.86, 0.07)]
        zloft(bm, sp, sides=12, mat=0)
    return finalize("Pants", bm, ["Pants"], recalc=False)


def build_boots():
    bm = bmesh.new()
    for s in (1, -1):
        x0 = s * LEG_X
        # shaft with a turned-down cuff (folded sheet: outside up, inside down)
        sp = [(0.042, x0, 0.012, 0.045, 0.052), (0.10, x0, 0.007, 0.040, 0.047),
              (0.18, x0, 0.004, 0.043, 0.049), (0.26, x0, 0.004, 0.050, 0.054),
              (0.34, x0, 0.005, 0.055, 0.058), (0.42, x0, 0.006, 0.056, 0.059),
              (0.490, x0, 0.006, 0.057, 0.061),
              (0.508, x0, 0.006, 0.066, 0.070), (0.525, x0, 0.006, 0.082, 0.086),
              (0.568, x0, 0.006, 0.088, 0.092), (0.585, x0, 0.006, 0.083, 0.087),
              (0.586, x0, 0.006, 0.070, 0.074), (0.545, x0, 0.006, 0.062, 0.066),
              (0.500, x0, 0.006, 0.060, 0.064)]
        zloft(bm, sp, sides=16, mat=0, caps=False)
        # one-piece foot: instep + toe, overlapping the lower shaft (no gap)
        O = Vector((x0, 0.058, 0.0))
        foot = [(0.000, 0.030, 0.030, 0.0, 0.043), (0.018, 0.041, 0.037, 0.0, 0.047),
                (0.060, 0.047, 0.041, 0.0, 0.050), (0.118, 0.050, 0.037, 0.0, 0.047),
                (0.168, 0.047, 0.031, 0.0, 0.041), (0.212, 0.038, 0.025, 0.0, 0.037),
                (0.236, 0.022, 0.018, 0.0, 0.032)]
        axial_loft(bm, O, (0, -1, 0), (1, 0, 0), foot, sides=12, mat=0)
        # sole slab
        sole = [(0.000, 0.032, 0.0075, 0, 0.0075), (0.020, 0.043, 0.0075, 0, 0.0075),
                (0.070, 0.050, 0.0075, 0, 0.0075), (0.125, 0.053, 0.0075, 0, 0.0075),
                (0.175, 0.049, 0.0075, 0, 0.0075), (0.218, 0.040, 0.0075, 0, 0.0075),
                (0.240, 0.024, 0.0070, 0, 0.0070)]
        axial_loft(bm, O, (0, -1, 0), (1, 0, 0), sole, sides=10, mat=1)
        # heel block
        zloft(bm, [(0.0, x0, 0.044, 0.036, 0.034), (0.030, x0, 0.044, 0.034, 0.032),
                   (0.046, x0, 0.044, 0.032, 0.030)], sides=12, mat=1, caps=True)
    return finalize("Boots", bm, ["Boots", "Leather"], recalc=False)


def hand_frame(s):
    """Local basis of the hand: d (towards fingers), a (finger row, ~+Y), n (palm normal)."""
    S, E, W, T, d2, d3 = arm_joints(s)
    d = d3.copy()
    a = Vector((0, 1, 0))
    a = (a - d * a.dot(d)).normalized()
    n = d.cross(a).normalized()
    inward = Vector((-s, 0, -0.3))
    if n.dot(inward) < 0:
        n = -n
    return W, d, a, n


def grip_center(s):
    W, d, a, n = hand_frame(s)
    return W + d * 0.066 + n * 0.042


def build_gloves():
    bm = bmesh.new()
    for s in (1, -1):
        S, E, W, T, d2, d3 = arm_joints(s)
        # gauntlet cuff (krag): flared, covers the end of the sleeve
        cuff = [(-0.078, 0.046, 0.046), (-0.074, 0.056, 0.056), (-0.066, 0.058, 0.058),
                (-0.040, 0.050, 0.048), (-0.010, 0.040, 0.034), (0.018, 0.036, 0.026)]
        axial_loft(bm, W, d2, (0, 1, 0), cuff, sides=14, mat=0, caps=False)
        # gold cuff band on the rim
        axial_loft(bm, W, d2, (0, 1, 0),
                   [(-0.081, 0.0605, 0.0605), (-0.058, 0.0625, 0.0625),
                    (-0.058, 0.0560, 0.0560), (-0.081, 0.0540, 0.0540)],
                   sides=14, mat=1, wrap=True, caps=False)
        Wp, d, a, n = hand_frame(s)
        # palm / back of hand
        palm = [(-0.012, 0.030, 0.021), (0.020, 0.035, 0.021), (0.060, 0.039, 0.019),
                (0.082, 0.037, 0.016)]
        axial_loft(bm, Wp, d, a, palm, sides=10, mat=0)
        P = Wp + d * 0.040
        K0 = Wp + d * 0.080
        fing_len = [0.064, 0.070, 0.064, 0.052]          # index .. pinky
        offs = [-0.027, -0.009, 0.009, 0.026]
        fr = [0.0092, 0.0095, 0.0090, 0.0080]
        for fi in range(4):
            base = K0 + a * offs[fi] + n * 0.002
            if s == -1:
                # right hand: loose C-curl around the sword grip (grip axis = a)
                pts = []
                for ang in (-60.0, 10.0, 80.0, 150.0):
                    r_ = math.radians(ang)
                    pts.append(Wp + a * offs[fi] + d * (0.066 + 0.031 * math.cos(r_))
                               + n * (0.042 + 0.031 * math.sin(r_)))
            else:
                L1, L2 = fing_len[fi] * 0.55, fing_len[fi] * 0.45
                c1 = math.radians(18)
                c2 = math.radians(55)
                spread = a * (offs[fi] * 0.25)
                d1_ = (d * math.cos(c1) + n * math.sin(c1) + spread * 0.3).normalized()
                d2_ = (d * math.cos(c2) + n * math.sin(c2) + spread * 0.4).normalized()
                pts = [base, base + d1_ * L1, base + d1_ * L1 + d2_ * L2]
            m = len(pts)
            rads = [fr[fi] * (1.0 - 0.18 * i / (m - 1)) for i in range(m)]
            sweep(bm, pts, rads, rads, mat=0, sides=6, ref=tuple(a))
            # knuckle pad
            blob(bm, base, (fr[fi] * 1.05,) * 3, 0, 6, 4)
        # thumb
        tb = Wp + d * 0.018 + a * (-0.030) + n * 0.004
        if s == -1:
            tp = [tb, Wp + d * 0.036 + a * (-0.034) + n * 0.028,
                  Wp + d * 0.044 + a * (-0.034) + n * 0.0565]
        else:
            tp = [tb, tb + d * 0.026 - a * 0.014 + n * 0.006,
                  tb + d * 0.046 - a * 0.020 + n * 0.014]
        sweep(bm, tp, [0.0105, 0.0098, 0.0088], [0.0105, 0.0098, 0.0088], mat=0, sides=6, ref=tuple(a))
        blob(bm, tb, (0.011,) * 3, 0, 6, 4)
    return finalize("Gloves", bm, ["Gloves", "Gold"], recalc=False)


# ---------------------------------------------------------------- ARMATURE
BONE_DEFS = []  # (name, head, tail, parent, deform)


def define_bones():
    global BONE_DEFS
    d = []
    d.append(("Hips", (0, 0, 0.90), (0, 0, 1.04), None, True))
    d.append(("Spine", (0, 0, 1.04), (0, 0, 1.17), "Hips", True))
    d.append(("Chest", (0, 0, 1.17), (0, 0, 1.31), "Spine", True))
    d.append(("Neck", (0, 0, 1.31), (0, 0, 1.40), "Chest", True))
    d.append(("Head", (0, 0, 1.40), (0, 0, 1.66), "Neck", True))
    for s, side in ((1, "L"), (-1, "R")):
        S, E, W, T, d2, d3 = arm_joints(s)
        d.append((f"Shoulder.{side}", (s * 0.05, 0, 1.29), tuple(S), "Chest", True))
        d.append((f"UpperArm.{side}", tuple(S), tuple(E), f"Shoulder.{side}", True))
        d.append((f"LowerArm.{side}", tuple(E), tuple(W), f"UpperArm.{side}", True))
        d.append((f"Hand.{side}", tuple(W), tuple(T), f"LowerArm.{side}", True))
        if side == "R":
            sock_h = grip_center(-1)
            d.append(("WeaponSocket.R", tuple(sock_h), tuple(sock_h + d3 * 0.05),
                      "Hand.R", False))
        d.append((f"UpperLeg.{side}", (s * 0.09, 0, 0.88), (s * LEG_X, 0, 0.48), "Hips", True))
        d.append((f"LowerLeg.{side}", (s * LEG_X, 0, 0.48), (s * LEG_X, 0, 0.09),
                  f"UpperLeg.{side}", True))
        d.append((f"Foot.{side}", (s * LEG_X, 0, 0.09), (s * LEG_X, -0.14, 0.035),
                  f"LowerLeg.{side}", True))
        d.append((f"Toes.{side}", (s * LEG_X, -0.14, 0.035), (s * LEG_X, -0.185, 0.03),
                  f"Foot.{side}", True))
    BONE_DEFS = d


def make_armature():
    define_bones()
    arm_data = bpy.data.armatures.new("Armature")
    arm = bpy.data.objects.new("Armature", arm_data)
    COLL.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_data.edit_bones
    for name, head, tail, parent, deform in BONE_DEFS:
        b = eb.new(name)
        b.head = Vector(head)
        b.tail = Vector(tail)
        if parent:
            b.parent = eb[parent]
        b.use_deform = deform
    bpy.ops.object.mode_set(mode='OBJECT')
    arm.show_in_front = True
    arm_data.display_type = 'STICK'
    return arm


def bind_head_only(ob, arm):
    """Rigid binding: every vertex weight 1.0 to bone Head."""
    vg = ob.vertex_groups.new(name="Head")
    vg.add([v.index for v in ob.data.vertices], 1.0, 'REPLACE')
    mod = ob.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm
    ob.parent = arm


def fix_unweighted(ob, arm, default="Head"):
    """Vertices that automatic weights left empty (separate islands) get bound
    rigidly: Head for the head parts, otherwise the nearest deform bone."""
    bones = [b for b in arm.data.bones if b.use_deform]
    fixed = 0
    for v in ob.data.vertices:
        if sum(g.weight for g in v.groups) < 1e-6:
            if ob.name in ("Body", "Hair") and v.co.z > 1.40:
                name = default
            else:
                best, name = 1e9, default
                for b in bones:
                    h, t = b.head_local, b.tail_local
                    ab = t - h
                    k = max(0.0, min(1.0, (v.co - h).dot(ab) / max(ab.length_squared, 1e-9)))
                    dist = (v.co - (h + ab * k)).length
                    if dist < best:
                        best, name = dist, b.name
            vg = ob.vertex_groups.get(name) or ob.vertex_groups.new(name=name)
            vg.add([v.index], 1.0, 'REPLACE')
            fixed += 1
    print(f"fix_unweighted {ob.name}: {fixed} verts bound to nearest bone / Head")


def ramp_group(ob, vweights, bone_name):
    """vweights: {vertex index: w}.  Blend each vertex towards a single bone:
    w=1 -> 100% bone_name, 0<w<1 keeps (1-w) of the existing weights."""
    vg_new = ob.vertex_groups.get(bone_name) or ob.vertex_groups.new(name=bone_name)
    for vi, w in vweights.items():
        v = ob.data.vertices[vi]
        cur = [(g.group, g.weight) for g in v.groups if ob.vertex_groups[g.group].name != bone_name]
        own = sum(g.weight for g in v.groups if ob.vertex_groups[g.group].name == bone_name)
        tot = sum(x for _, x in cur) + own
        if tot < 1e-6:
            cur, own, tot = [], 0.0, 1.0
        for gi, x in cur:
            ob.vertex_groups[gi].add([vi], (x / tot) * (1.0 - w), 'REPLACE')
        vg_new.add([vi], (own / tot) * (1.0 - w) + w, 'REPLACE')


def force_group(ob, vidx, bone_name):
    """Make the given vertices 100% weighted to a single bone."""
    for gi in range(len(ob.vertex_groups)):
        ob.vertex_groups[gi].remove(vidx)
    vg = ob.vertex_groups.get(bone_name) or ob.vertex_groups.new(name=bone_name)
    vg.add(vidx, 1.0, 'REPLACE')


def limit_influences(ob, n=4):
    """glTF allows at most 4 joint influences per vertex: keep the 4 strongest
    weights and renormalize, so the export does not silently drop weights."""
    changed = 0
    for v in ob.data.vertices:
        pairs = [(g.group, min(1.0, max(0.0, g.weight))) for g in v.groups]  # plain copies
        if not pairs:
            continue
        pairs.sort(key=lambda t: -t[1])
        keep, drop = pairs[:n], pairs[n:]
        for gi, _ in drop:
            ob.vertex_groups[gi].remove([v.index])
        total = sum(w for _, w in keep)
        if total <= 0.0:
            continue
        for gi, w in keep:
            # float32 rounding can leave 1.0000001 after normalization; clamp it
            ob.vertex_groups[gi].add([v.index], min(1.0, w / total), 'REPLACE')
        if drop or any(abs(w - 1.0) > 1e-7 for _, w in keep) and len(keep) > 1:
            changed += 1
    print(f"limit_influences {ob.name}: {changed} verts rewritten (max {n} bones, clamped)")


def bind_auto(objs, arm):
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for ob in objs:
        ob.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    bpy.ops.object.select_all(action='DESELECT')


# ---------------------------------------------------------------- RENDER
def look_rot(loc, target):
    return (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()


def setup_scene_render():
    SCENE.render.engine = 'CYCLES'
    SCENE.cycles.device = 'CPU'
    SCENE.cycles.samples = 32
    SCENE.cycles.use_denoising = False
    SCENE.render.resolution_x = 640
    SCENE.render.resolution_y = 800
    SCENE.render.resolution_percentage = 100
    SCENE.render.film_transparent = False
    SCENE.view_settings.view_transform = 'Standard'
    SCENE.render.image_settings.file_format = 'PNG'

    world = bpy.data.worlds.new("Backdrop")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.20, 0.23, 0.29, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    SCENE.world = world

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50.0
    cam_data.sensor_fit = 'VERTICAL'
    cam_data.sensor_height = 24.0
    cam = bpy.data.objects.new("Camera", cam_data)
    COLL.objects.link(cam)
    SCENE.camera = cam

    def area(name, loc, energy, color, size):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.energy = energy
        ld.color = color
        ld.size = size
        lo = bpy.data.objects.new(name, ld)
        COLL.objects.link(lo)
        lo.location = Vector(loc)
        lo.rotation_euler = look_rot(loc, (0, 0, 0.95))

    # warm key (front-left), cold rim (back-right), soft fill (front-right)
    area("Key", (-2.4, -3.0, 2.6), 170.0, (1.0, 0.84, 0.66), 2.0)
    area("Rim", (2.6, 2.8, 2.4), 160.0, (0.62, 0.78, 1.0), 1.5)
    area("Fill", (3.0, -2.6, 1.0), 45.0, (0.92, 0.94, 1.0), 3.0)
    return cam


def render_views(cam):
    dist, height = 4.4, 0.90
    body_target = (0.0, 0.0, 0.88)
    views = {
        "hero_front": ((0.0, -dist, height), body_target),
        "hero_side": ((dist, 0.0, height), body_target),
        "hero_back": ((0.0, dist, height), body_target),
        "hero_34": ((-dist * 0.707, -dist * 0.707, height + 0.1), body_target),
        # close-up of the face, slightly from the character's left
        "hero_face": ((0.22, -0.80, 1.58), (0.0, -0.04, 1.52)),
    }
    os.makedirs(DOCS_DIR, exist_ok=True)
    dbg = os.environ.get("HERO_CAM")          # debug: "x,y,z,tx,ty,tz,out.png"
    if dbg:
        v = dbg.split(",")
        loc, target = tuple(map(float, v[0:3])), tuple(map(float, v[3:6]))
        cam.location = Vector(loc)
        cam.rotation_euler = look_rot(loc, target)
        SCENE.render.filepath = v[6]
        bpy.ops.render.render(write_still=True)
        return
    for name, (loc, target) in views.items():
        cam.location = Vector(loc)
        cam.rotation_euler = look_rot(loc, target)
        SCENE.render.filepath = os.path.join(DOCS_DIR, name + ".png")
        bpy.ops.render.render(write_still=True)
        print("RENDERED", SCENE.render.filepath)


# ---------------------------------------------------------------- CHECKS
def count_tris(ob):
    me = ob.data
    me.calc_loop_triangles()
    return len(me.loop_triangles)


def mesh_islands(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    seen, islands = set(), []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack, comp = [v], []
        seen.add(v.index)
        while stack:
            c = stack.pop()
            comp.append(c.index)
            for e in c.link_edges:
                o = e.other_vert(c)
                if o.index not in seen:
                    seen.add(o.index)
                    stack.append(o)
        islands.append(comp)
    bm.free()
    return islands


def check_geometry(meshes):
    """Island orientation (closed shells must have positive volume), island
    connectivity (overlap / touching) and global bounds."""
    from mathutils.bvhtree import BVHTree
    print("\n=== GEOMETRY CHECK ===")
    zmin, zmax = 1e9, -1e9
    for ob in meshes:
        me = ob.data
        islands = mesh_islands(me)
        vis = [set(i) for i in islands]
        closed_bad, closed_n, open_n, open_in = 0, 0, 0, 0
        trees = []
        for comp, vs in zip(islands, vis):
            polys = [p for p in me.polygons if p.vertices[0] in vs]
            vol = 0.0
            for p in polys:
                vv = [me.vertices[i].co for i in p.vertices]
                for k in range(1, len(vv) - 1):
                    vol += vv[0].dot(vv[k].cross(vv[k + 1])) / 6.0
            edges = {}
            for p in polys:
                for ek in p.edge_keys:
                    edges[ek] = edges.get(ek, 0) + 1
            is_closed = all(c == 2 for c in edges.values())
            if is_closed:
                closed_n += 1
                if vol < 0:
                    closed_bad += 1
            else:
                open_n += 1
                cen = sum((me.vertices[i].co for i in comp), Vector()) / len(comp)
                good = 0
                for p in polys:
                    if p.normal.dot(p.center - Vector((0, cen.y, p.center.z))) > 0 or \
                       p.normal.dot(p.center - cen) > 0:
                        good += 1
                open_in += good / max(1, len(polys))
            tris = [list(p.vertices) for p in polys]
            idx = sorted(vs)
            remap = {v: k for k, v in enumerate(idx)}
            tri = []
            for p in polys:
                pv = [remap[i] for i in p.vertices]
                for k in range(1, len(pv) - 1):
                    tri.append((pv[0], pv[k], pv[k + 1]))
            trees.append(BVHTree.FromPolygons([me.vertices[i].co.copy() for i in idx], tri, epsilon=0.003))
        # union-find over overlapping / touching (<=3 mm) islands
        par = list(range(len(islands)))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for i in range(len(islands)):
            for j in range(i + 1, len(islands)):
                if find(i) == find(j):
                    continue
                if trees[i].overlap(trees[j]):
                    par[find(i)] = find(j)
        groups = len({find(i) for i in range(len(islands))})
        for v in me.vertices:
            zmin, zmax = min(zmin, v.co.z), max(zmax, v.co.z)
        flag = "  <-- FLIPPED" if closed_bad else ""
        print(f"{ob.name:8s} islands={len(islands):3d} closed={closed_n:3d} (inside-out: {closed_bad}) "
              f"open={open_n:3d} outward_frac={open_in / max(1, open_n):.2f} "
              f"connected_groups={groups}{flag}")
    print(f"BOUNDS z: min={zmin:.4f}  max={zmax:.4f}")


def deformation_test(arm, meshes):
    """Rotate each bone 0.35 rad: every mesh must move, edges must not tear,
    and parts that touch at rest (different islands/objects, <4 mm) must not
    drift apart by more than 2 cm."""
    from mathutils.kdtree import KDTree
    print("\n=== DEFORMATION TEST (each bone rotated 0.35 rad) ===")
    moved_by = {ob.name: set() for ob in meshes}
    rest = {ob.name: [v.co.copy() for v in ob.data.vertices] for ob in meshes}
    edges = {ob.name: [(e.vertices[0], e.vertices[1]) for e in ob.data.edges] for ob in meshes}
    # touching pairs across objects at rest
    allv = [(ob.name, i, co) for ob in meshes for i, co in enumerate(rest[ob.name])]
    kd = KDTree(len(allv))
    for k, (_, _, co) in enumerate(allv):
        kd.insert(co, k)
    kd.balance()
    pairs = []
    for k, (on, i, co) in enumerate(allv):
        for (_, k2, d) in kd.find_range(co, 0.004):
            if k2 > k and allv[k2][0] != on:
                pairs.append((k, k2))
    if len(pairs) > 6000:
        pairs = pairs[::len(pairs) // 6000 + 1]
    pair_rest = [(allv[a][2] - allv[b][2]).length for a, b in pairs]
    index = {(on, i): k for k, (on, i, _) in enumerate(allv)}
    worst_edge, worst_pair = (1.0, "", ""), (0.0, "", "")
    for pb in arm.pose.bones:
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = (0.35, 0.0, 0.0)
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        pos = {}
        for ob in meshes:
            ev = ob.evaluated_get(dg)
            em = ev.to_mesh()
            cur = [v.co.copy() for v in em.vertices]
            ev.to_mesh_clear()
            pos[ob.name] = cur
            n = sum(1 for c, r in zip(cur, rest[ob.name]) if (c - r).length > 1e-4)
            if n > 0:
                moved_by[ob.name].add(pb.name)
            for (a, b) in edges[ob.name]:
                l0 = (rest[ob.name][a] - rest[ob.name][b]).length
                if l0 > 1e-5:
                    r = (cur[a] - cur[b]).length / l0
                    if r > worst_edge[0]:
                        worst_edge = (r, ob.name, pb.name)
        flat = [pos[on][i] for (on, i, _) in allv]
        for (a, b), d0 in zip(pairs, pair_rest):
            d = (flat[a] - flat[b]).length
            if d - d0 > worst_pair[0]:
                worst_pair = (d - d0, f"{allv[a][0]}/{allv[b][0]}", pb.name)
        pb.rotation_euler = (0.0, 0.0, 0.0)
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    for ob in meshes:
        bl = sorted(moved_by[ob.name])
        status = "OK" if bl else "NO DEFORMATION"
        print(f"{ob.name:8s} {status} moved_by={len(bl)} bones")
    print(f"worst edge stretch: x{worst_edge[0]:.2f} ({worst_edge[1]}, bone {worst_edge[2]})")
    print(f"worst drift of touching parts ({len(pairs)} pairs): {worst_pair[0] * 100:.2f} cm "
          f"({worst_pair[1]}, bone {worst_pair[2]})")


def check_and_report(arm, meshes):
    total_tris = 0
    print("\n=== OBJECTS ===")
    for ob in meshes:
        t = count_tris(ob)
        total_tris += t
        mats = [s.material.name for s in ob.material_slots if s.material]
        print(f"{ob.name:8s} verts={len(ob.data.vertices):5d} tris={t:5d} "
              f"mats={mats} parent={ob.parent.name if ob.parent else None}")
    print(f"TOTAL TRIANGLES: {total_tris}")

    bones = [b.name for b in arm.data.bones]
    print(f"\n=== BONES ({len(bones)}) ===")
    print(", ".join(bones))

    used = sorted({s.material.name for ob in meshes for s in ob.material_slots if s.material})
    print(f"\n=== MATERIALS ({len(used)}) ===")
    print(", ".join(used))

    print("\n=== BINDING CHECK ===")
    for ob in meshes:
        unbound = 0
        bone_names = set()
        for v in ob.data.vertices:
            tot = 0.0
            for g in v.groups:
                if g.weight > 1e-6:
                    tot += g.weight
                    bone_names.add(ob.vertex_groups[g.group].name)
            if tot < 1e-6:
                unbound += 1
        has_arm = any(m.type == 'ARMATURE' and m.object == arm for m in ob.modifiers)
        print(f"{ob.name:8s} armature_mod={has_arm} unbound_verts={unbound} "
              f"influenced_by={len(bone_names)} bones")

    check_geometry(meshes)
    deformation_test(arm, meshes)
    return total_tris


# ---------------------------------------------------------------- MAIN
def main():
    # no Blender backup files (.blend1) on repeated runs
    bpy.context.preferences.filepaths.save_version = 0
    body = build_body()
    hair = build_hair()
    hat = build_hat()
    coat = build_coat()
    shirt = build_shirt()
    belt = build_belt()
    sash = build_sash()
    pants = build_pants()
    boots = build_boots()
    gloves = build_gloves()
    eyes = build_eyes()
    bandana = build_bandana()

    arm = make_armature()

    bind_auto([body, coat, shirt, pants, boots, gloves, belt, sash], arm)
    # head part of the skin mesh (face, lids, nose, lips, ears) moves rigidly with Head
    ramp_group(body, {v.index: smoothstep(1.36, 1.43, v.co.z) for v in body.data.vertices
                      if v.co.z > 1.36}, "Head")
    force_group(body, [v.index for v in body.data.vertices if v.co.z > 1.43], "Head")
    # fingers / hand rigidly on Hand.L/R; the gauntlet cuff blends with the forearm
    for s_, side in ((1, "L"), (-1, "R")):
        W_, d_, a_, n_ = hand_frame(s_)
        ramp = {v.index: smoothstep(-0.03, 0.06, (v.co - W_).dot(d_)) for v in gloves.data.vertices
                if (v.co.x * s_ > 0) and (v.co - W_).dot(d_) > -0.03}
        ramp_group(gloves, ramp, f"Hand.{side}")
        force_group(gloves, [i for i, w in ramp.items() if (gloves.data.vertices[i].co - W_).dot(d_) > 0.06],
                    f"Hand.{side}")
    for ob in (body, coat, shirt, pants, boots, gloves, belt, sash):
        fix_unweighted(ob, arm)
        limit_influences(ob, 4)
    for ob in (hair, hat, eyes, bandana):
        bind_head_only(ob, arm)

    meshes = [body, hair, hat, coat, shirt, belt, sash, pants, boots, gloves, eyes, bandana]

    # leave armature at rest pose
    for pb in arm.pose.bones:
        pb.rotation_mode = 'XYZ'

    check_and_report(arm, meshes)

    os.makedirs(MODELS_DIR, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(MODELS_DIR, "hero.blend"))
    print("SAVED", os.path.join(MODELS_DIR, "hero.blend"))

    bpy.ops.export_scene.gltf(
        filepath=os.path.join(MODELS_DIR, "hero.glb"),
        export_format='GLB',
        export_apply=False,
        export_yup=True,
        export_skins=True,
        export_animations=False,
    )
    print("EXPORTED", os.path.join(MODELS_DIR, "hero.glb"))

    if DO_RENDER:
        cam = setup_scene_render()
        render_views(cam)


main()
