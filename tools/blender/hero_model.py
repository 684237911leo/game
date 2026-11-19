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
    "Hair":    ((0.62, 0.22, 0.07),        0.80, 0.0),
    "Hat":     ((0.07, 0.06, 0.055),       0.70, 0.0),
    "Gold":    ((0.95, 0.72, 0.25),        0.60, 0.6),
    "Feather": ((0.70, 0.06, 0.08),        0.85, 0.0),
    "Coat":    ((0.30, 0.03, 0.07),        0.75, 0.0),
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
    """Loft an elliptical cross-section along a polyline path.
    rx: semi-axis along the ref-derived 'u' direction, ry: along 'v'."""
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
        for ring in (rings[0], rings[-1]):
            bm.faces.new(ring).material_index = mat


def densify(pts, rx, ry, steps):
    P, X, Y = [], [], []
    for i in range(len(pts) - 1):
        for s in range(steps):
            t = s / steps
            P.append(pts[i].lerp(pts[i + 1], t))
            X.append(rx[i] + (rx[i + 1] - rx[i]) * t)
            Y.append(ry[i] + (ry[i + 1] - ry[i]) * t)
    P.append(pts[-1])
    X.append(rx[-1])
    Y.append(ry[-1])
    return P, X, Y


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


def ellipsoid_bm(center, radii, segs=8, rings=5, taper=None):
    """UV sphere scaled to an ellipsoid; optional taper on unit-sphere coords."""
    tmp = bmesh.new()
    bmesh.ops.create_uvsphere(tmp, u_segments=segs, v_segments=rings,
                              radius=1.0, matrix=Matrix.Identity(4))
    for v in tmp.verts:
        u = v.co.copy()
        if taper is not None:
            u = taper(u)
        v.co = Vector((center[0] + u.x * radii[0],
                       center[1] + u.y * radii[1],
                       center[2] + u.z * radii[2]))
    return tmp


def subsurf(src, levels=1):
    """Apply a Subdivision Surface modifier through a temporary object and
    return the resulting mesh as a new bmesh (equivalent to modifier apply)."""
    me = bpy.data.meshes.new("_tmp_sub")
    src.to_mesh(me)
    ob = bpy.data.objects.new("_tmp_sub", me)
    COLL.objects.link(ob)
    mod = ob.modifiers.new("Subdivision", 'SUBSURF')
    mod.levels = levels
    mod.render_levels = levels
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = ob.evaluated_get(dg)
    emesh = ev.to_mesh()
    out = bmesh.new()
    out.from_mesh(emesh)
    ev.to_mesh_clear()
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.meshes.remove(me)
    return out


def append(dst, src, mat):
    src.verts.ensure_lookup_table()
    src.verts.index_update()
    vm = [dst.verts.new(v.co) for v in src.verts]
    for f in src.faces:
        try:
            dst.faces.new([vm[v.index] for v in f.verts]).material_index = mat
        except ValueError:
            pass
    src.free()


def finalize(name, bm, mat_names):
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


def arm_joints(s):
    """s=+1 -> character's left (+X), s=-1 -> right (-X)."""
    d = Vector((s * math.cos(ARM_ANGLE), 0.0, -math.sin(ARM_ANGLE)))
    S = Vector((s * 0.17, 0.0, 1.31))
    E = S + d * 0.29
    W = E + d * 0.26
    T = W + d * 0.14
    return S, E, W, T, d


# ---------------------------------------------------------------- BODY (skin)
def build_body():
    bm = bmesh.new()

    def head_taper(u):
        if u.z < 0.0:
            k = min(1.0, -u.z)
            return Vector((u.x * (1.0 - 0.28 * k * k), u.y * (1.0 - 0.10 * k), u.z))
        return u

    head = ellipsoid_bm((0.0, 0.0, 1.53), (0.088, 0.10, 0.115),
                        segs=12, rings=8, taper=head_taper)
    append(bm, subsurf(head, 1), 0)
    # small stylized lips (second material slot)
    append(bm, ellipsoid_bm((0.0, -0.085, 1.462), (0.020, 0.011, 0.007),
                            segs=6, rings=4), 1)
    # neck
    sweep(bm, [Vector((0, 0, 1.36)), Vector((0, 0, 1.45))],
          [0.050, 0.046], [0.046, 0.044], mat=0, sides=8)
    # torso
    tp = [Vector((0, 0, z)) for z in (0.80, 0.92, 1.02, 1.14, 1.25, 1.32, 1.34)]
    sweep(bm, tp, [0.12, 0.165, 0.125, 0.15, 0.17, 0.12, 0.07],
          [0.10, 0.115, 0.09, 0.105, 0.10, 0.08, 0.07], mat=0, sides=12)
    for s in (1, -1):
        S, E, W, T, _ = arm_joints(s)
        P, X, Y = densify([S, E, W, T], [0.042, 0.036, 0.028, 0.026],
                          [0.042, 0.036, 0.028, 0.014], 2)
        sweep(bm, P, X, Y, mat=0, sides=10)
        # legs
        pts = [Vector((s * 0.090, 0, 0.88)), Vector((s * 0.092, 0, 0.68)),
               Vector((s * 0.095, 0, 0.48)), Vector((s * 0.095, 0.008, 0.28)),
               Vector((s * 0.095, 0, 0.09))]
        sweep(bm, pts, [0.085, 0.070, 0.048, 0.050, 0.034],
              [0.085, 0.072, 0.048, 0.052, 0.034], mat=0, sides=10)
    return finalize("Body", bm, ["Skin", "Lips"])


# ---------------------------------------------------------------- HAIR
def build_hair():
    bm = bmesh.new()
    cx, cy, cz = 0.0, 0.035, 1.58
    shell = ellipsoid_bm((cx, cy, cz), (0.135, 0.128, 0.120), segs=12, rings=8)
    append(bm, subsurf(shell, 1), 0)
    # volume bumps (curl clusters)
    # (volume comes from the shell and the ringlets; no bumps above the brim)
    # hanging ringlets: (angle deg, base z, length, coil radius, turns)
    ringlets = [
        (-22, 1.57, 0.30, 0.018, 1.6),    # front side curl (+X), beside face
        (-158, 1.57, 0.30, 0.018, 1.6),   # front side curl (-X), beside face
        (-62, 1.55, 0.22, 0.022, 2.5),
        (-118, 1.55, 0.22, 0.022, 2.5),
        (20, 1.55, 0.22, 0.022, 2.5),
        (160, 1.55, 0.22, 0.022, 2.5),
        (55, 1.60, 0.20, 0.022, 2.5),
        (125, 1.60, 0.20, 0.022, 2.5),
        (90, 1.52, 0.18, 0.022, 2.5),
        (35, 1.52, 0.18, 0.020, 2.5),
        (145, 1.52, 0.18, 0.020, 2.5),
        (0, 1.50, 0.16, 0.020, 2.5),
        (180, 1.50, 0.16, 0.020, 2.5),
        (110, 1.64, 0.14, 0.020, 2.0),
    ]
    ax, ay = 0.135, 0.128
    for ang_deg, z0, length, coil, turns in ringlets:
        th = math.radians(ang_deg)
        out = Vector((math.cos(th), math.sin(th), 0.0))
        tan = Vector((-math.sin(th), math.cos(th), 0.0))
        down = Vector((0.0, 0.0, -1.0))
        base = Vector((cx + ax * math.cos(th), cy + ay * math.sin(th), z0))
        nseg = int(turns * 7)
        pts = []
        for j in range(nseg + 1):
            t = j / nseg
            a = TWO_PI * turns * t
            p = (base + down * (length * t)
                 + out * (coil * (math.cos(a) - 1.0))
                 + tan * (coil * math.sin(a)))
            pts.append(p)
        sweep(bm, pts, 0.0145, 0.0145, mat=0, sides=8)
    return finalize("Hair", bm, ["Hair"])


# ---------------------------------------------------------------- HAT (tricorn)
def tricorn_brim(bm, center, R=0.25, edge_pts=8, mat=0):
    corners = [Vector((R * math.cos(a), R * math.sin(a), 0.0))
               for a in (-math.pi / 2, -math.pi / 2 + 2 * math.pi / 3,
                         -math.pi / 2 + 4 * math.pi / 3)]
    outline, flags = [], []
    for k in range(3):
        A, B = corners[k], corners[(k + 1) % 3]
        for j in range(edge_pts):
            t = j / edge_pts
            p = A.lerp(B, t)
            p = p - p.normalized() * 0.03 * math.sin(math.pi * t)  # bowed edges
            outline.append(p)
            flags.append(1.0 - 2.0 * min(t, 1.0 - t))          # 1 at corners
    K = len(outline)
    c_v = bm.verts.new(center)
    rings = []
    for f in (0.34, 0.68, 1.0):
        ring = []
        for p, fl in zip(outline, flags):
            co = Vector((center.x + p.x * f, center.y + p.y * f,
                         center.z + 0.075 * fl * f * f))
            ring.append(bm.verts.new(co))
        rings.append(ring)
    r0 = rings[0]
    for k in range(K):
        bm.faces.new((c_v, r0[k], r0[(k + 1) % K])).material_index = mat
    for j in range(len(rings) - 1):
        ra, rb = rings[j], rings[j + 1]
        for k in range(K):
            k2 = (k + 1) % K
            bm.faces.new((ra[k], ra[k2], rb[k2], rb[k])).material_index = mat


def build_hat():
    bm = bmesh.new()
    # crown (sinks into the hair volume at its base)
    crown = [Vector((0.0, 0.02, z)) for z in (1.60, 1.66, 1.72, 1.74)]
    sweep(bm, crown, [0.134, 0.130, 0.125, 0.118],
          [0.124, 0.120, 0.115, 0.108], mat=0, sides=12)
    # gold band
    sweep(bm, ellipse_pts((0.0, 0.02, 1.61), 0.136, 0.126, 24),
          0.007, 0.007, mat=1, sides=6, closed=True)
    # brim
    tricorn_brim(bm, Vector((0.0, 0.02, 1.615)), R=0.25, edge_pts=8, mat=0)
    # feather
    fp = [Vector((0.10, 0.02, 1.70)), Vector((0.19, 0.00, 1.80)),
          Vector((0.27, -0.02, 1.84)), Vector((0.30, -0.06, 1.79))]
    sweep(bm, fp, [0.03, 0.04, 0.035, 0.02], [0.004] * 4,
          mat=2, sides=4, ref=(0.0, 1.0, 0.0))
    return finalize("Hat", bm, ["Hat", "Gold", "Feather"])


# ---------------------------------------------------------------- COAT
def build_coat():
    bm = bmesh.new()
    zs = [0.56, 0.70, 0.86, 0.96, 1.04, 1.14, 1.25, 1.31, 1.37]
    rxs = [0.27, 0.23, 0.19, 0.20, 0.15, 0.18, 0.195, 0.15, 0.095]
    rys = [0.19, 0.17, 0.14, 0.15, 0.115, 0.13, 0.12, 0.095, 0.085]
    pts = [Vector((0, 0, z)) for z in zs]
    sweep(bm, pts, rxs, rys, mat=0, sides=16, caps=False)
    # front opening: remove front faces so the shirt/belt/trousers show
    kill = []
    for f in bm.faces:
        c = f.calc_center_median()
        if c.y < -0.06 and abs(c.x) < 0.07 and 0.60 < c.z < 1.36:
            kill.append(f)
    bmesh.ops.delete(bm, geom=kill, context='FACES')
    # sleeves
    for s in (1, -1):
        S, E, W, T, d = arm_joints(s)
        cuff_c = W - d * 0.02
        P, X, Y = densify([S, E, cuff_c], [0.064, 0.054, 0.046],
                          [0.064, 0.054, 0.046], 2)
        sweep(bm, P, X, Y, mat=0, sides=10, caps=False)
        # gold cuff
        sweep(bm, circle_pts(cuff_c, d, 0.054, 16), 0.007, 0.007,
              mat=1, sides=6, closed=True)
    # gold hem and collar
    sweep(bm, ellipse_pts((0, 0, 0.565), 0.276, 0.196, 32),
          0.008, 0.008, mat=1, sides=6, closed=True)
    sweep(bm, ellipse_pts((0, 0, 1.37), 0.100, 0.090, 24),
          0.007, 0.007, mat=1, sides=6, closed=True)
    return finalize("Coat", bm, ["Coat", "Gold"])


# ---------------------------------------------------------------- SHIRT
def build_shirt():
    bm = bmesh.new()
    zs = [0.94, 1.04, 1.14, 1.25, 1.31, 1.36]
    rxs = [0.14, 0.125, 0.145, 0.16, 0.11, 0.09]
    rys = [0.10, 0.09, 0.10, 0.095, 0.08, 0.075]
    pts = [Vector((0, 0, z)) for z in zs]
    sweep(bm, pts, rxs, rys, mat=0, sides=12, caps=False)
    return finalize("Shirt", bm, ["Shirt"])


# ---------------------------------------------------------------- BELT and SASH
def build_belt():
    bm = bmesh.new()
    sweep(bm, ellipse_pts((0, 0, 1.05), 0.160, 0.124, 28),
          0.030, 0.012, mat=0, sides=8, ref=(0, 0, 1), closed=True)
    # gold buckle on the front
    before = set(bm.faces)
    bmesh.ops.create_cube(
        bm, size=1.0,
        matrix=Matrix.Translation((0.0, -0.136, 1.05)) @ Matrix.Diagonal((0.045, 0.014, 0.040, 1.0)))
    for f in bm.faces:
        if f not in before:
            f.material_index = 1
    return finalize("Belt", bm, ["Leather", "Gold"])


def build_sash():
    bm = bmesh.new()
    sweep(bm, ellipse_pts((0, 0, 0.96), 0.212, 0.160, 28),
          0.030, 0.020, mat=0, sides=8, ref=(0, 0, 1), closed=True)
    tail = [Vector((0.205, -0.07, 0.95)), Vector((0.235, -0.09, 0.85)),
            Vector((0.245, -0.10, 0.74)), Vector((0.250, -0.105, 0.66))]
    sweep(bm, tail, [0.030, 0.030, 0.028, 0.020], [0.012] * 4, mat=0, sides=6)
    return finalize("Sash", bm, ["Sash"])


# ---------------------------------------------------------------- PANTS, BOOTS, GLOVES
def build_pants():
    bm = bmesh.new()
    hip = [Vector((0, 0, z)) for z in (0.80, 0.92, 1.00)]
    sweep(bm, hip, [0.13, 0.165, 0.14], [0.10, 0.115, 0.10], mat=0, sides=12, caps=False)
    for s in (1, -1):
        pts = [Vector((s * 0.090, 0, 0.86)), Vector((s * 0.092, 0, 0.68)),
               Vector((s * 0.095, 0, 0.50))]
        sweep(bm, pts, [0.097, 0.084, 0.064], [0.097, 0.084, 0.064],
              mat=0, sides=10, caps=False)
    return finalize("Pants", bm, ["Pants"])


def build_boots():
    bm = bmesh.new()
    for s in (1, -1):
        shaft = [Vector((s * 0.095, 0, 0.10)), Vector((s * 0.095, 0, 0.28)),
                 Vector((s * 0.095, 0, 0.44)), Vector((s * 0.095, 0, 0.505)),
                 Vector((s * 0.095, 0, 0.525))]
        sweep(bm, shaft, [0.048, 0.058, 0.066, 0.070, 0.076],
              [0.052, 0.064, 0.070, 0.074, 0.080], mat=0, sides=12)
        foot = [Vector((s * 0.095, 0.03, 0.034)), Vector((s * 0.095, -0.06, 0.034)),
                Vector((s * 0.095, -0.15, 0.026))]
        sweep(bm, foot, [0.034, 0.032, 0.026], [0.040, 0.042, 0.034],
              mat=0, sides=10, ref=(0, 0, 1))
    return finalize("Boots", bm, ["Boots"])


def build_gloves():
    bm = bmesh.new()
    for s in (1, -1):
        S, E, W, T, d = arm_joints(s)
        sweep(bm, [W - d * 0.03, W + d * 0.07], [0.040, 0.037], [0.040, 0.034],
              mat=0, sides=10)
    return finalize("Gloves", bm, ["Gloves"])


# ---------------------------------------------------------------- EYES
def build_eyes():
    bm = bmesh.new()
    for x in (0.032, -0.032):
        append(bm, ellipsoid_bm((x, -0.094, 1.548), (0.017, 0.011, 0.019),
                                segs=8, rings=5), 0)
    return finalize("Eyes", bm, ["Eyes"])


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
        S, E, W, T, dd = arm_joints(s)
        d.append((f"Shoulder.{side}", (s * 0.05, 0, 1.29), tuple(S), "Chest", True))
        d.append((f"UpperArm.{side}", tuple(S), tuple(E), f"Shoulder.{side}", True))
        d.append((f"LowerArm.{side}", tuple(E), tuple(W), f"UpperArm.{side}", True))
        d.append((f"Hand.{side}", tuple(W), tuple(T), f"LowerArm.{side}", True))
        if side == "R":
            sock_h = W + dd * 0.07
            d.append(("WeaponSocket.R", tuple(sock_h), tuple(sock_h + dd * 0.05),
                      "Hand.R", False))
        hipx = s * 0.09
        d.append((f"UpperLeg.{side}", (hipx, 0, 0.88), (s * 0.095, 0, 0.48), "Hips", True))
        d.append((f"LowerLeg.{side}", (s * 0.095, 0, 0.48), (s * 0.095, 0, 0.09),
                  f"UpperLeg.{side}", True))
        d.append((f"Foot.{side}", (s * 0.095, 0, 0.09), (s * 0.095, -0.14, 0.035),
                  f"LowerLeg.{side}", True))
        d.append((f"Toes.{side}", (s * 0.095, -0.14, 0.035), (s * 0.095, -0.18, 0.03),
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


def fix_unweighted(ob, arm):
    """Islands disconnected from the body (lips) get no heat weights: bind them to Head."""
    head = ob.vertex_groups.get("Head") or ob.vertex_groups.new(name="Head")
    fixed = 0
    for v in ob.data.vertices:
        if sum(g.weight for g in v.groups) < 1e-6:
            head.add([v.index], 1.0, 'REPLACE')
            fixed += 1
    print(f"fix_unweighted {ob.name}: {fixed} verts bound to Head")


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
    views = {
        "hero_front": (0.0, -dist, height),
        "hero_side": (dist, 0.0, height),
        "hero_back": (0.0, dist, height),
        "hero_34": (-dist * 0.707, -dist * 0.707, height + 0.1),
    }
    target = (0.0, 0.0, 0.88)
    os.makedirs(DOCS_DIR, exist_ok=True)
    for name, loc in views.items():
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

    # deformation test: rotate each bone, see which objects move
    print("\n=== DEFORMATION TEST (each bone rotated 0.35 rad) ===")
    moved_by = {ob.name: set() for ob in meshes}
    rest = {ob.name: [v.co.copy() for v in ob.data.vertices] for ob in meshes}
    for pb in arm.pose.bones:
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = (0.35, 0.0, 0.0)
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        for ob in meshes:
            ev = ob.evaluated_get(dg)
            em = ev.to_mesh()
            n = 0
            for v, r in zip(em.vertices, rest[ob.name]):
                if (v.co - r).length > 1e-4:
                    n += 1
            ev.to_mesh_clear()
            if n > 0:
                moved_by[ob.name].add(pb.name)
        pb.rotation_euler = (0.0, 0.0, 0.0)
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    for ob in meshes:
        bl = sorted(moved_by[ob.name])
        status = "OK" if bl else "NO DEFORMATION"
        print(f"{ob.name:8s} {status} moved_by={len(bl)} bones")
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

    arm = make_armature()

    bind_auto([body, coat, shirt, pants, boots, gloves, belt, sash], arm)
    fix_unweighted(body, arm)
    for ob in (body, coat, shirt, pants, boots, gloves, belt, sash):
        limit_influences(ob, 4)
    for ob in (hair, hat, eyes):
        bind_head_only(ob, arm)

    meshes = [body, hair, hat, coat, shirt, belt, sash, pants, boots, gloves, eyes]

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
