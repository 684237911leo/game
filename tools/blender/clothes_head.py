"""Одежда головы героини: треуголка (Hat, Hat_Trim), перо (Feather),
бандана (Bandana), серьги (Earring). Все объекты жёстко на кости Head, вес 1.0.

Запуск (нужен numpy для glTF-импорта, см. CLOTHES_BRIEF.md):
  PYTHONPATH=<scratchpad>/pylib blender -b --factory-startup -P tools/blender/clothes_head.py -- build
  PYTHONPATH=<scratchpad>/pylib blender -b --factory-startup -P tools/blender/clothes_head.py -- check

build: строит геометрию по телу, выталкивает ее из волос/головы, пишет
       source_art/clothes_head.blend (только объекты одежды), печатает отчет.
check: подгружает тело и одежду, тест деформации (поворот Head на 0.35 рад
       по X/Y/Z) и рендер превью в docs/models/clothes_head_*.png.
"""
import bpy
import bmesh
import math
import random
import sys

from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

BODY_GLB = "/home/user/game/models/adapted_body.glb"
SRC_BLEND = "/home/user/game/source_art/clothes_head.blend"
DOCS = "/home/user/game/docs/models"
BODY_NAMES = ["Body", "Hair", "Scalp", "Eyes", "Eyelashes"]
HEAD_NAMES = ["Body", "Hair", "Scalp"]
PARTS = ["Hat", "Hat_Trim", "Feather", "Bandana", "Earring"]
BONE = "Head"

TILT = Matrix.Rotation(math.radians(-10.0), 3, "Y")  # наклон шляпы набок
PIVOT = Vector((0.0, 0.0, 1.66))
MARGIN = 0.006  # зазор одежды до волос/кожи, м


def tilt(p):
    return TILT @ (p - PIVOT) + PIVOT


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ell(th, rx, ry, cy=0.0):
    return Vector((rx * math.cos(th), ry * math.sin(th) + cy, 0.0))


# ---------------------------------------------------------------- сцена / материалы

def load_body():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=BODY_GLB)


def make_mat(name, rgb, metallic=0.0, roughness=0.6):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    mat.diffuse_color = (*rgb, 1.0)
    return mat


def make_materials():
    make_mat("Hat", (0.075, 0.055, 0.045), roughness=0.75)
    make_mat("Gold", (0.95, 0.72, 0.28), metallic=1.0, roughness=0.35)
    make_mat("Feather", (0.78, 0.05, 0.05), roughness=0.5)
    make_mat("Bandana", (0.62, 0.04, 0.04), roughness=0.85)


def mk(name, V, F, mat):
    me = bpy.data.meshes.new(name + "_mesh")
    me.from_pydata([tuple(v) for v in V], [], [tuple(f) for f in F])
    me.validate()
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    me.materials.append(bpy.data.materials[mat])
    for p in me.polygons:
        p.use_smooth = False
    vg = ob.vertex_groups.new(name=BONE)
    vg.add(list(range(len(V))), 1.0, "REPLACE")
    return ob


def tri_count(me):
    return sum(len(p.vertices) - 2 for p in me.polygons)


# ---------------------------------------------------------------- тулья + поля

CROWN_ROWS = [  # (z, rx, ry) — низ тульи (сядет на волосы), околыш, тулья чуть сужается, купол
    (1.616, 0.098, 0.108),
    (1.636, 0.106, 0.116),
    (1.652, 0.110, 0.119),  # околыш
    (1.668, 0.107, 0.116),
    (1.720, 0.104, 0.113),
    (1.785, 0.098, 0.106),
]
NC = 32
BRIM_Z = 1.68
BRIM_T = 0.015  # толщина полей (эквивалент Solidify 1.5 см)
NB = 36
EDGE_TH = [math.radians(a) for a in (-30, 90, 210)]  # середины сторон
CORNER_TH = [math.radians(a) for a in (-90, 30, 150)]  # три угла


def crown_geo():
    V, F = [], []
    for z, rx, ry in CROWN_ROWS:
        for i in range(NC):
            th = 2 * math.pi * i / NC
            V.append(Vector((rx * math.cos(th), ry * math.sin(th), z)))
    rows = len(CROWN_ROWS)
    for r in range(rows - 1):
        for i in range(NC):
            a = r * NC + i
            b = r * NC + (i + 1) % NC
            c = (r + 1) * NC + (i + 1) % NC
            d = (r + 1) * NC + i
            F.append((a, b, c, d))
    top = len(V)
    V.append(Vector((0.0, 0.0, 1.80)))
    last = (rows - 1) * NC
    for i in range(NC):
        F.append((last + i, last + (i + 1) % NC, top))
    return V, F


def brim_R(th):
    off = min(abs(wrap(th - e)) for e in EDGE_TH)
    return 0.175 / max(math.cos(off), 1e-3) ** 0.6  # углы ~0.27, середины 0.175


def corner_factor(th):
    return 0.5 * (1 + math.cos(3 * (th - CORNER_TH[0])))


def brim_H(th):
    return 0.035 + 0.065 * corner_factor(th)  # углы загнуты сильнее


def brim_point(th, s, dz):
    inner = ell(th, 0.106, 0.116)
    R = brim_R(th)
    outer = Vector((R * math.cos(th), R * math.sin(th), 0.0))
    p = inner + (outer - inner) * s
    p.z = BRIM_Z + brim_H(th) * (s ** 1.7) - dz
    return p


def brim_geo():
    V, F = [], []
    rows = [0.0, 0.5, 1.0]
    for dz in (0.0, BRIM_T):  # верхний и нижний лист
        for s in rows:
            for i in range(NB):
                V.append(brim_point(2 * math.pi * i / NB, s, dz))
    top0, bot0 = 0, 3 * NB
    for base, flip in ((top0, True), (bot0, False)):
        for r in range(2):
            for i in range(NB):
                a = base + r * NB + i
                b = base + r * NB + (i + 1) % NB
                c = base + (r + 1) * NB + (i + 1) % NB
                d = base + (r + 1) * NB + i
                F.append((a, d, c, b) if flip else (a, b, c, d))
    for i in range(NB):  # внешний край
        j = (i + 1) % NB
        ta, tb = top0 + 2 * NB + i, top0 + 2 * NB + j
        ba, bb = bot0 + 2 * NB + i, bot0 + 2 * NB + j
        F.append((ta, ba, bb, tb))
    for i in range(NB):  # внутренний край (под тульей)
        j = (i + 1) % NB
        ta, tb = top0 + i, top0 + j
        ba, bb = bot0 + i, bot0 + j
        F.append((ta, tb, bb, ba))
    return V, F


def trim_geo():
    """Золотой кант по внешнему краю полей: квадратное сечение 1.5 см."""
    h = 0.0075
    V, F = [], []
    for i in range(NB):
        th = 2 * math.pi * i / NB
        R = brim_R(th)
        c = Vector((R * math.cos(th), R * math.sin(th), 0.0))
        c.z = BRIM_Z + brim_H(th) - BRIM_T / 2
        ro = Vector((math.cos(th), math.sin(th), 0.0))
        zz = Vector((0.0, 0.0, 1.0))
        for sr, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            V.append(c + ro * (sr * h) + zz * (sz * h))
    for i in range(NB):
        j = (i + 1) % NB
        for k in range(4):
            k2 = (k + 1) % 4
            F.append((i * 4 + k, i * 4 + k2, j * 4 + k2, j * 4 + k))
    return V, F


# ---------------------------------------------------------------- перо

def bez(p, t):
    p0, p1, p2, p3 = p
    u = 1 - t
    return u ** 3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t ** 3 * p3


def feather_geo():
    ctrl = [
        Vector((0.11, 0.0, 1.76)),
        Vector((0.16, -0.012, 1.90)),
        Vector((0.25, -0.03, 1.95)),
        Vector((0.33, -0.012, 1.86)),
    ]
    V, F = [], []
    # стержень: шестигранная трубка, сужается к кончику
    NS, NR = 10, 6
    pts = [bez(ctrl, j / NS) for j in range(NS + 1)]
    rings = []
    for j in range(NS + 1):
        T = (pts[min(j + 1, NS)] - pts[max(j - 1, 0)]).normalized()
        N = T.cross(Vector((0, 0, 1)))
        if N.length < 1e-4:
            N = T.cross(Vector((1, 0, 0)))
        N.normalize()
        Bv = T.cross(N).normalized()
        r = 0.0042 * (1 - 0.65 * j / NS)
        ring = []
        for k in range(NR):
            a = 2 * math.pi * k / NR
            ring.append(len(V))
            V.append(pts[j] + (N * math.cos(a) + Bv * math.sin(a)) * r)
        rings.append(ring)
    for j in range(NS):
        for k in range(NR):
            k2 = (k + 1) % NR
            F.append((rings[j][k], rings[j][k2], rings[j + 1][k2], rings[j + 1][k]))
    # перья: 10 вытянутых лепестков, каждый — 3 ряда (край, ребро, край)
    rng = random.Random(7)
    # 12 длинных лепестков веером + 6 коротких внутренних у основания (пышность)
    petals = []
    for k in range(12):
        t = min(0.98, max(0.35, 0.35 + 0.056 * k + rng.uniform(-0.015, 0.015)))
        d = Vector((0.45 + 0.45 * rng.random(), rng.uniform(-0.9, 0.9), rng.uniform(-0.45, 0.35))).normalized()
        petals.append((t, d, rng.uniform(0.15, 0.23), rng.uniform(0.015, 0.023)))
    for k in range(6):
        t = 0.22 + 0.06 * k + rng.uniform(-0.01, 0.01)
        d = Vector((0.25 + 0.4 * rng.random(), rng.uniform(-0.8, 0.8), rng.uniform(0.1, 0.6))).normalized()
        petals.append((t, d, rng.uniform(0.09, 0.13), rng.uniform(0.017, 0.024)))
    NU = 5
    for t, d, L, W in petals:
        B = bez(ctrl, t)
        sv = d.cross(Vector((0, 0, 1))).normalized()
        nrm = sv.cross(d).normalized()
        base = len(V)
        rowsL, rowsC, rowsR = [], [], []
        for j in range(NU + 1):
            u = j / NU
            S = B + d * (L * u) + Vector((0, 0, -0.035 * u * u))
            w = W * math.sin(math.pi * min(1.0, u) ** 0.75)
            rowsL.append(len(V)); V.append(S - sv * w)
            rowsC.append(len(V)); V.append(S + nrm * (0.35 * w))
            rowsR.append(len(V)); V.append(S + sv * w)
        for j in range(NU):
            F.append((rowsL[j], rowsC[j], rowsC[j + 1], rowsL[j + 1]))
            F.append((rowsC[j], rowsR[j], rowsR[j + 1], rowsC[j + 1]))
    return V, F


# ---------------------------------------------------------------- бандана, серьги

def bandana_geo():
    V, F = [], []
    NBd = 32
    rings = [(1.574, 0.095, 0.108, -0.004), (1.607, 0.097, 0.110, -0.004)]
    idx = []
    for z, rx, ry, cy in rings:
        row = []
        for i in range(NBd):
            th = 2 * math.pi * i / NBd
            p = ell(th, rx, ry, cy)
            p.z = z
            row.append(len(V))
            V.append(p)
        idx.append(row)
    for i in range(NBd):
        j = (i + 1) % NBd
        F.append((idx[0][i], idx[0][j], idx[1][j], idx[1][i]))
    # узелок на затылке (икосфера, 80 треугольников)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.018)
    base = len(V)
    center = Vector((0.0, 0.112, 1.592))
    for v in bm.verts:
        V.append(v.co + center)
    for f in bm.faces:
        F.append(tuple(base + v.index for v in f.verts))
    bm.free()
    # два коротких хвоста сзади
    for side in (-1, 1):
        rows = []
        for j in range(5):
            x = side * (0.006 + 0.007 * j)
            y = 0.116 + 0.012 * j
            z = 1.586 - 0.026 * j
            rows.append((len(V), len(V) + 1))
            V.append(Vector((x - 0.008, y, z)))
            V.append(Vector((x + 0.008, y, z)))
        for j in range(4):
            a, b = rows[j], rows[j + 1]
            F.append((a[0], a[1], b[1], b[0]))
    return V, F


def earring_geo():
    V, F = [], []
    R, r = 0.012, 0.0022
    NMj, NMn = 10, 4
    for side in (-1, 1):
        C = Vector((side * 0.125, 0.005, 1.455))
        X = Vector((1, 0, 0))
        base = len(V)
        for i in range(NMj):
            a = 2 * math.pi * i / NMj
            dirA = Vector((0.0, math.cos(a), math.sin(a)))
            for k in range(NMn):
                b = 2 * math.pi * k / NMn
                V.append(C + dirA * (R + r * math.cos(b)) + X * (r * math.sin(b)))
        for i in range(NMj):
            i2 = (i + 1) % NMj
            for k in range(NMn):
                k2 = (k + 1) % NMn
                F.append((base + i * NMn + k, base + i * NMn + k2,
                          base + i2 * NMn + k2, base + i2 * NMn + k))
    return V, F


# ---------------------------------------------------------------- вытеснение из головы

def bvh_from(objs_or_geo):
    V, F = objs_or_geo
    return BVHTree.FromPolygons(V, F)


def world_geo(ob):
    M = ob.matrix_world
    return [M @ v.co for v in ob.data.vertices], [tuple(p.vertices) for p in ob.data.polygons]


def merged_geo(names):
    V, F = [], []
    for n in names:
        v, f = world_geo(bpy.data.objects[n])
        off = len(V)
        V.extend(v)
        F.extend(tuple(i + off for i in fp) for fp in f)
    return V, F


HEAD_C = Vector((0.0, 0.0, 1.50))  # центр черепа для радиального выталкивания


def push_vec(v, bvh, mode, margin, maxd):
    """mode: 'sign' — нормали препятствия наружу (тело);
    'radial' — наружу от центра головы (волосы, их нормали не гарантированы);
    'd' — от ближайшей точки (другие предметы одежды)."""
    loc, n, _, dist = bvh.find_nearest(v, maxd)
    if loc is None:
        return None
    d = v - loc
    if mode == "sign":
        s = d.dot(n)
        if s >= margin:
            return None
        return n * (margin - s)
    if dist >= margin:
        return None
    if mode == "radial":
        rv = v - HEAD_C
        rv.z *= 0.7
        dirv = rv.normalized() if rv.length > 1e-7 else Vector((0, 0, 1))
    else:
        dirv = d.normalized() if dist > 1e-7 else n
    return dirv * (margin - dist)


def relax(ob, obstacles, margin=MARGIN, iters=80, maxd=0.07):
    """Выталкивает вершины ob наружу, пока зазор до препятствий < margin.
    obstacles: список (bvh, mode)."""
    me = ob.data
    V = [v.co.copy() for v in me.vertices]
    nb = [set() for _ in V]
    for e in me.edges:
        a, b = e.vertices
        nb[a].add(b)
        nb[b].add(a)
    for _ in range(iters):
        moved = []
        for i, v in enumerate(V):
            best = None
            for bvh, mode in obstacles:
                p = push_vec(v, bvh, mode, margin, maxd)
                if p is not None and (best is None or p.length > best.length):
                    best = p
            if best is not None:
                moved.append((i, v + best))
        if not moved:
            break
        for i, p in moved:
            V[i] = p
        smooth = []
        for i, _ in moved:
            if nb[i]:
                avg = sum((V[j] for j in nb[i]), Vector()) / len(nb[i])
                smooth.append((i, V[i] * 0.6 + avg * 0.4))
        for i, p in smooth:
            V[i] = p
    for i, v in enumerate(V):
        me.vertices[i].co = v
    me.update()


# ---------------------------------------------------------------- сборка

def build():
    load_body()
    make_materials()
    bvh_body = bvh_from(merged_geo(["Body"]))
    bvh_hs = bvh_from(merged_geo(HEAD_NAMES))
    obst = [(bvh_body, "sign"), (bvh_hs, "radial")]

    # треуголка = тулья + поля (+ золотой кант отдельным объектом)
    cv, cf = crown_geo()
    bv, bf = brim_geo()
    off = len(cv)
    hatV = [tilt(p) for p in cv + bv]
    hatF = cf + [tuple(i + off for i in f) for f in bf]
    hat = mk("Hat", hatV, hatF, "Hat")
    tv, tf = trim_geo()
    mk("Hat_Trim", [tilt(p) for p in tv], tf, "Gold")
    relax(hat, obst)
    bvh_hat = bvh_from(world_geo(hat))

    fv, ff = feather_geo()
    mk("Feather", [tilt(p) for p in fv], ff, "Feather")
    relax(bpy.data.objects["Feather"], obst + [(bvh_hat, "d")])

    bdv, bdf = bandana_geo()
    mk("Bandana", bdv, bdf, "Bandana")
    relax(bpy.data.objects["Bandana"], obst + [(bvh_hat, "d")])

    ev, ef = earring_geo()
    mk("Earring", ev, ef, "Gold")
    relax(bpy.data.objects["Earring"], obst)

    report(bvh_hs)
    cleanup_and_save()


def report(bvh_hs):
    # перекрытия треугольников с телом/волосами и минимальный зазор
    all_V, all_F = merged_geo(BODY_NAMES)
    bvh_all = bvh_from((all_V, all_F))
    for n in PARTS:
        ob = bpy.data.objects[n]
        V, F = world_geo(ob)
        bvh_p = bvh_from((V, F))
        pairs = bvh_p.overlap(bvh_all)
        mind = min(
            (bvh_all.find_nearest(v, 0.2)[3] or 0.2) for v in V
        )
        tris = tri_count(ob.data)
        print("HEADREPORT %-10s tris=%4d overlap_pairs_vs_body=%d min_dist_to_body=%.4f"
              % (n, tris, len(pairs), mind))
    total = sum(tri_count(bpy.data.objects[n].data) for n in PARTS)
    print("HEADREPORT TOTAL tris=%d" % total)


def cleanup_and_save():
    for ob in list(bpy.data.objects):
        if ob.name not in PARTS:
            bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.images,
                 bpy.data.textures, bpy.data.cameras, bpy.data.lights):
        for d in list(coll):
            if d.users == 0:
                coll.remove(d)
    keep = {"Hat", "Gold", "Feather", "Bandana"}
    for m in list(bpy.data.materials):
        if m.name not in keep:
            bpy.data.materials.remove(m)
    bpy.ops.wm.save_as_mainfile(filepath=SRC_BLEND)
    print("HEADREPORT saved", SRC_BLEND)


# ---------------------------------------------------------------- проверка, деформация, превью

def evaluated_geo(ob, dg):
    oe = ob.evaluated_get(dg)
    me = oe.to_mesh()
    M = oe.matrix_world
    V = [M @ v.co for v in me.vertices]
    F = [tuple(p.vertices) for p in me.polygons]
    oe.to_mesh_clear()
    return V, F


def check():
    load_body()
    make_materials()
    with bpy.data.libraries.load(SRC_BLEND, link=False) as (df, dt):
        dt.objects = [n for n in df.objects if n in PARTS]
    scene = bpy.context.scene
    parts = []
    for ob in dt.objects:
        scene.collection.objects.link(ob)
        parts.append(ob)
    arm = bpy.data.objects["Armature"]
    for ob in parts:
        mod = ob.modifiers.new("Armature", "ARMATURE")
        mod.object = arm

    pb = arm.pose.bones[BONE]
    pb.rotation_mode = "XYZ"
    worst = {}
    for axis_i, axis in enumerate("XYZ"):
        for sign in (1, -1):
            e = [0.0, 0.0, 0.0]
            e[axis_i] = sign * 0.35
            pb.rotation_euler = e
            bpy.context.view_layer.update()
            dg = bpy.context.evaluated_depsgraph_get()
            body_V, body_F = [], []
            for n in HEAD_NAMES:
                v, f = evaluated_geo(bpy.data.objects[n], dg)
                off = len(body_V)
                body_V.extend(v)
                body_F.extend(tuple(i + off for i in fp) for fp in f)
            bvh_b = bvh_from((body_V, body_F))
            for ob in parts:
                V, F = evaluated_geo(ob, dg)
                bvh_p = bvh_from((V, F))
                pairs = len(bvh_p.overlap(bvh_b))
                worst[ob.name] = max(worst.get(ob.name, 0), pairs)
                if pairs:
                    print("DEFORM %s%+.2f %-10s overlap_pairs=%d" % (axis, sign * 0.35, ob.name, pairs))
            hair_v, _ = evaluated_geo(bpy.data.objects["Hair"], dg)
            hat_v, _ = evaluated_geo(bpy.data.objects["Hat"], dg)
            print("DEFORM pose %s%+.2f hair_zmax=%.3f hat_zmin=%.3f hat_xmin=%.3f hat_xmax=%.3f"
                  % (axis, sign * 0.35, max(p.z for p in hair_v),
                     min(p.z for p in hat_v), min(p.x for p in hat_v), max(p.x for p in hat_v)))
    pb.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    print("DEFORM worst overlap pairs over 6 poses:", worst)

    # превью
    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.45, 0.46, 0.5, 1.0)
    bg.inputs[1].default_value = 1.0
    scene.world = world
    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = 3.0
    so = bpy.data.objects.new("Sun", sun)
    scene.collection.objects.link(so)
    so.rotation_euler = (math.radians(55), 0.0, math.radians(35))

    def cam(name, loc, target, lens=55):
        cd = bpy.data.cameras.new(name)
        cd.lens = lens
        co = bpy.data.objects.new(name, cd)
        scene.collection.objects.link(co)
        co.location = Vector(loc)
        d = Vector(target) - Vector(loc)
        co.rotation_mode = "QUATERNION"
        co.rotation_quaternion = d.to_track_quat("-Z", "Y")
        return co

    sc = scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 16
    sc.cycles.use_denoising = False
    sc.render.image_settings.file_format = "PNG"
    views = [
        ("front", (0.0, -1.45, 1.52), (0.0, 0.0, 1.50), 55, 800, 900),
        ("side", (1.45, 0.0, 1.52), (0.0, 0.0, 1.50), 55, 800, 900),
        ("back", (0.0, 1.45, 1.52), (0.0, 0.0, 1.50), 55, 800, 900),
        ("close", (0.42, -0.62, 1.60), (0.0, -0.01, 1.58), 80, 900, 900),
    ]
    for name, loc, tgt, lens, w, h in views:
        sc.camera = cam("cam_" + name, loc, tgt, lens)
        sc.render.resolution_x = w
        sc.render.resolution_y = h
        sc.render.resolution_percentage = 100
        out = "%s/clothes_head_%s.png" % (DOCS, name)
        sc.render.filepath = out
        bpy.ops.render.render(write_still=True)
        print("RENDER", out)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["build"]
    mode = argv[0]
    if mode == "build":
        build()
    elif mode == "check":
        check()
    else:
        raise SystemExit("unknown mode " + mode)


main()
