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

TILT = Matrix.Rotation(math.radians(-9.0), 3, "Y")  # наклон шляпы набок (левая сторона вверх)
PIVOT = Vector((0.0, 0.0, 1.57))
MARGIN = 0.006  # зазор одежды до волос/кожи, м


def tilt(p):
    return TILT @ (p - PIVOT) + PIVOT


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ell(th, rx, ry):
    return 1.0 / math.sqrt((math.cos(th) / rx) ** 2 + (math.sin(th) / ry) ** 2)


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
    make_mat("Feather", (0.50, 0.015, 0.09), roughness=0.5)
    make_mat("Feather_Edge", (0.17, 0.004, 0.035), roughness=0.55)
    make_mat("Bandana", (0.62, 0.04, 0.04), roughness=0.85)


def mk(name, V, F, mats, fmat=None, smooth=False):
    me = bpy.data.meshes.new(name + "_mesh")
    me.from_pydata([tuple(v) for v in V], [], [tuple(f) for f in F])
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    for m in mats:
        me.materials.append(bpy.data.materials[m])
    for i, p in enumerate(me.polygons):
        p.use_smooth = smooth
        if fmat:
            p.material_index = fmat[i]
    vg = ob.vertex_groups.new(name=BONE)
    vg.add(list(range(len(V))), 1.0, "REPLACE")
    return ob


def tri_count(me):
    return sum(len(p.vertices) - 2 for p in me.polygons)


# ---------------------------------------------------------------- радиальные замеры головы

def far_hit(bvh, th, z, maxd=0.4):
    """Наружная граница волос/головы вдоль горизонтального луча из оси головы."""
    d = Vector((math.cos(th), math.sin(th), 0.0))
    o = Vector((0.0, 0.0, z))
    pos, last = o, None
    for _ in range(14):
        loc, n, i, dist = bvh.ray_cast(pos, d, maxd)
        if loc is None:
            break
        last = loc
        pos = loc + d * 0.0005
    if last is None:
        return None
    return (last - o).dot(d)


def smooth_ring(r, win=2, passes=2):
    n = len(r)
    r = [max(r[(i + k) % n] for k in range(-win, win + 1)) for i in range(n)]
    for _ in range(passes):
        r = [(r[i - 1] + 2 * r[i] + r[(i + 1) % n]) / 4 for i in range(n)]
    return r


def ring_radius(bvh, z, n, margin, fallback=None, k=1.0):
    r = []
    for i in range(n):
        th = 2 * math.pi * i / n
        h = far_hit(bvh, th, z)
        f = (fallback(th) * k) if fallback else 0.0
        r.append(max((h + margin) if h is not None else 0.0, f))
    return smooth_ring(r)


# ---------------------------------------------------------------- тулья + поля

CROWN_ROWS = [  # (z, k для запаса формы) — купол с лёгким сужением к верху
    (1.560, 1.00), (1.583, 0.99), (1.606, 0.97), (1.628, 0.93),
    (1.645, 0.86), (1.658, 0.72), (1.668, 0.50), (1.675, 0.25),
]
NC = 32
BRIM_Z = 1.575
BRIM_T = 0.015  # толщина полей 1.5 см
CORNERS = [math.radians(a) for a in (-90, 18, 162)]  # спереди, слева, справа
BRIM_S = [0.0, 0.45, 1.0]


def cf(th):
    return max(math.exp(-(wrap(th - c) / 0.52) ** 2) for c in CORNERS)


def crown_geo(bvh_head):
    fb = lambda th: ell(th, 0.108, 0.120)
    rows = [ring_radius(bvh_head, z, NC, 0.011, fb, k) for z, k in CROWN_ROWS]
    V, F = [], []
    for (z, _), r in zip(CROWN_ROWS, rows):
        for i in range(NC):
            th = 2 * math.pi * i / NC
            V.append(Vector((r[i] * math.cos(th), r[i] * math.sin(th), z)))
    for r in range(len(CROWN_ROWS) - 1):
        for i in range(NC):
            j = (i + 1) % NC
            F.append((r * NC + j, r * NC + i, (r + 1) * NC + i, (r + 1) * NC + j))
    top = len(V)
    V.append(Vector((0.0, 0.0, 1.679)))
    last = (len(CROWN_ROWS) - 1) * NC
    for i in range(NC):
        F.append((last + i, last + (i + 1) % NC, top))
    return V, F, rows


def brim_point(th, r0, s, dz):
    c = cf(th)
    R = r0 + 0.118 + 0.028 * c
    r = r0 + (R - r0) * s
    z = BRIM_Z + (0.012 + 0.072 * c) * (s ** 1.8) - dz
    return Vector((r * math.cos(th), r * math.sin(th), z))


def brim_geo(r0s):
    V, F = [], []
    ns = len(BRIM_S)
    for dz in (0.0, BRIM_T):
        for s in BRIM_S:
            for i in range(NC):
                V.append(brim_point(2 * math.pi * i / NC, r0s[i], s, dz))
    top0, bot0 = 0, ns * NC
    for base, up in ((top0, True), (bot0, False)):
        for r in range(ns - 1):
            for i in range(NC):
                j = (i + 1) % NC
                a, b = base + r * NC + i, base + r * NC + j
                c, d = base + (r + 1) * NC + j, base + (r + 1) * NC + i
                F.append((a, b, c, d) if not up else (b, a, d, c))
    # (порядок обхода: верх смотрит вверх, низ вниз; ориентацию уточняет recalc ниже)
    for i in range(NC):
        j = (i + 1) % NC
        ta, tb = top0 + (ns - 1) * NC + i, top0 + (ns - 1) * NC + j
        ba, bb = bot0 + (ns - 1) * NC + i, bot0 + (ns - 1) * NC + j
        F.append((ta, tb, bb, ba))
        F.append((top0 + j, top0 + i, bot0 + i, bot0 + j))
    return V, F


def trim_geo(r0s):
    """Золотой кант по внешнему краю полей."""
    hr, hz = 0.0045, 0.0095
    V, F = [], []
    for i in range(NC):
        th = 2 * math.pi * i / NC
        c = brim_point(th, r0s[i], 1.0, BRIM_T / 2)
        ro = Vector((math.cos(th), math.sin(th), 0.0))
        zz = Vector((0.0, 0.0, 1.0))
        for sr, sz in ((1, 0), (-1, 1), (-1, -1)):
            V.append(c + ro * (sr * hr) + zz * (sz * hz))
    for i in range(NC):
        j = (i + 1) % NC
        for k in range(3):
            k2 = (k + 1) % 3
            F.append((i * 3 + k, j * 3 + k, j * 3 + k2, i * 3 + k2))
    return V, F


def cockade_geo(center, radial):
    """Золотая кокарда-диск на тулье (нормаль — радиально наружу)."""
    n = 10
    ax = radial.normalized()
    u = Vector((0, 0, 1)).cross(ax).normalized()
    v = ax.cross(u).normalized()
    V, F = [], []
    for side, off in ((0, 0.0), (1, 0.006)):
        for i in range(n):
            a = 2 * math.pi * i / n
            V.append(center + ax * off + (u * math.cos(a) + v * math.sin(a)) * 0.019)
    cb = len(V)
    V.append(center)
    ct = len(V)
    V.append(center + ax * 0.011)
    for i in range(n):
        j = (i + 1) % n
        F.append((i, j, n + j, n + i))
        F.append((n + i, n + j, ct))
        F.append((j, i, cb))
    return V, F


# ---------------------------------------------------------------- перо

def bez(p, t):
    p0, p1, p2, p3 = p
    u = 1 - t
    return u ** 3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t ** 3 * p3


def feather_geo(root):
    ctrl = [
        root,
        root + Vector((0.050, 0.030, 0.150)),
        root + Vector((0.100, 0.180, 0.230)),
        root + Vector((0.115, 0.330, 0.120)),
    ]
    V, F, FM = [], [], []
    NS, NR = 12, 5
    pts = [bez(ctrl, j / NS) for j in range(NS + 1)]
    tans = [(pts[min(j + 1, NS)] - pts[max(j - 1, 0)]).normalized() for j in range(NS + 1)]
    rings = []
    for j in range(NS + 1):
        T = tans[j]
        S = (Vector((1, 0, 0)) - T * T.x).normalized()
        Nv = T.cross(S).normalized()
        r = 0.0035 * (1 - 0.6 * j / NS)
        ring = []
        for k in range(NR):
            a = 2 * math.pi * k / NR
            ring.append(len(V))
            V.append(pts[j] + (S * math.cos(a) + Nv * math.sin(a)) * r)
        rings.append(ring)
    for j in range(NS):
        for k in range(NR):
            k2 = (k + 1) % NR
            F.append((rings[j][k], rings[j][k2], rings[j + 1][k2], rings[j + 1][k]))
            FM.append(1)
    # три слоя перьев-бородок по бокам стержня
    layers = [  # (смещение по нормали, завиток, длина, угол к стержню, провис)
        (0.000, 0.00, 1.00, math.radians(52), 0.020),
        (0.007, 0.55, 0.84, math.radians(42), 0.030),
        (-0.007, -0.55, 0.68, math.radians(62), 0.040),
    ]
    NBARB = 22
    rows_u = [0.0, 0.25, 0.5, 0.75]
    wprof = [0.45, 0.95, 0.80, 0.42]
    rng = random.Random(11)
    for li, (noff, curl, lf, ang, droop) in enumerate(layers):
        for side in (-1, 1):
            for b in range(NBARB):
                t = 0.10 + 0.87 * (b + (0.5 if li == 1 else 0.0) + rng.uniform(-0.12, 0.12)) / NBARB
                t = min(0.985, max(0.06, t))
                P = bez(ctrl, t)
                T = (bez(ctrl, min(1, t + 0.02)) - bez(ctrl, max(0, t - 0.02))).normalized()
                S = (Vector((1, 0, 0)) - T * T.x).normalized()
                Nv = T.cross(S).normalized()
                prof = max(0.25, math.sin(math.pi * (0.06 + 0.92 * t)) ** 0.7)
                L = 0.078 * lf * prof * rng.uniform(0.92, 1.06)
                d = (S * side * math.cos(ang) + T * math.sin(ang)).normalized()
                W = 0.0098 * (0.7 + 0.5 * prof)
                wv = Nv.cross(d).normalized()
                root0 = P + Nv * noff
                rows = []
                for u, wp in zip(rows_u, wprof):
                    c = root0 + d * (L * u) + Nv * (curl * L * 0.45 * u * u) + Vector((0, 0, -droop * u * u * (L / 0.078)))
                    rows.append((len(V), len(V) + 1))
                    V.append(c - wv * W * wp)
                    V.append(c + wv * W * wp)
                tipc = root0 + d * L + Nv * (curl * L * 0.45) + Vector((0, 0, -droop * (L / 0.078)))
                tip = len(V)
                V.append(tipc)
                for q in range(3):
                    a, bq = rows[q], rows[q + 1]
                    m = 0 if q < 2 else 1
                    flip = side > 0
                    f1, f2 = (a[0], a[1], bq[1]), (a[0], bq[1], bq[0])
                    if flip:
                        f1, f2 = f1[::-1], f2[::-1]
                    F.append(f1); FM.append(m)
                    F.append(f2); FM.append(m)
                last = rows[3]
                F.append((last[0], last[1], tip) if not side > 0 else (last[1], last[0], tip))
                FM.append(1)
    return V, F, FM


# ---------------------------------------------------------------- бандана, серьги

def bandana_geo(bvh_head):
    NBd = 32
    zs = [1.538, 1.559, 1.580]
    th_k = math.radians(128)
    rows_in, rows_out = [], []
    for z in zs:
        rin = ring_radius(bvh_head, z, NBd, 0.0045)
        rows_in.append(rin)
    V, F = [], []
    idx_in, idx_out = [], []
    for z, rin in zip(zs, rows_in):
        row_i, row_o = [], []
        for i in range(NBd):
            th = 2 * math.pi * i / NBd
            zz = z - 0.010 * max(0.0, math.sin(th)) * 0.6  # сзади чуть ниже
            row_i.append(len(V)); V.append(Vector((rin[i] * math.cos(th), rin[i] * math.sin(th), zz)))
        for i in range(NBd):
            th = 2 * math.pi * i / NBd
            zz = z - 0.010 * max(0.0, math.sin(th)) * 0.6
            ro = rin[i] + 0.0035
            row_o.append(len(V)); V.append(Vector((ro * math.cos(th), ro * math.sin(th), zz)))
        idx_in.append(row_i); idx_out.append(row_o)
    nr = len(zs)
    for r in range(nr - 1):
        for i in range(NBd):
            j = (i + 1) % NBd
            F.append((idx_out[r][i], idx_out[r][j], idx_out[r + 1][j], idx_out[r + 1][i]))
            F.append((idx_in[r][j], idx_in[r][i], idx_in[r + 1][i], idx_in[r + 1][j]))
    for i in range(NBd):
        j = (i + 1) % NBd
        F.append((idx_in[0][i], idx_in[0][j], idx_out[0][j], idx_out[0][i]))
        F.append((idx_out[-1][i], idx_out[-1][j], idx_in[-1][j], idx_in[-1][i]))
    # узел сзади-справа
    kr = far_hit(bvh_head, th_k, 1.560) + 0.012
    center = Vector((kr * math.cos(th_k), kr * math.sin(th_k), 1.552))
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    ax = Vector((math.cos(th_k), math.sin(th_k), 0))
    tg = Vector((-math.sin(th_k), math.cos(th_k), 0))
    base = len(V)
    for v in bm.verts:
        co = v.co
        V.append(center + ax * (co.x * 0.011) + tg * (co.y * 0.020) + Vector((0, 0, co.z * 0.016)))
    for f in bm.faces:
        F.append(tuple(base + v.index for v in f.verts))
    bm.free()
    # два хвоста вдоль волос на спину/плечи
    for sgn, dth, w0, zend in ((1, 1.2, 0.026, 1.325), (-1, -2.6, 0.024, 1.300)):
        NJ = 9
        pr = []
        for j in range(NJ):
            u = j / (NJ - 1)
            z = 1.548 - u * (1.548 - zend)
            th = th_k + math.radians(dth * j) + 0.05 * math.sin(u * 5.0) * sgn
            h = far_hit(bvh_head, th, z)
            if h is None:
                h = 0.11
            r = h + 0.007 + 0.004 * math.sin(u * 7 + sgn)
            w = w0 * (1.0 - 0.25 * u)
            tgv = Vector((-math.sin(th), math.cos(th), 0))
            c = Vector((r * math.cos(th), r * math.sin(th), z))
            nrm = Vector((math.cos(th), math.sin(th), 0))
            pr.append((c, tgv, nrm, w))
        # заострённый (ласточкин хвост) конец
        for layer, off in ((0, 0.0), (1, -0.003)):
            ids = []
            for c, tgv, nrm, w in pr:
                ids.append((len(V), len(V) + 1))
                V.append(c + nrm * off - tgv * w / 2)
                V.append(c + nrm * off + tgv * w / 2)
            c, tgv, nrm, w = pr[-1]
            tip = len(V)
            V.append(c + nrm * off - Vector((0, 0, 0.030)))
            for j in range(NJ - 1):
                a, b = ids[j], ids[j + 1]
                q = (a[0], a[1], b[1], b[0])
                F.append(q if layer == 0 else q[::-1])
            a = ids[-1]
            F.append((a[0], a[1], tip) if layer == 0 else (a[1], a[0], tip))
    return V, F


def earring_geo():
    V, F = [], []
    R, r = 0.0085, 0.0016
    NMj, NMn = 10, 3
    phi = math.radians(50)
    for side in (-1, 1):
        C = Vector((side * 0.063, 0.012, 1.412))
        u = Vector((math.cos(phi), math.sin(phi) * side, 0.0))
        base = len(V)
        for i in range(NMj):
            a = 2 * math.pi * i / NMj
            dirA = u * math.cos(a) + Vector((0, 0, 1)) * math.sin(a)
            for k in range(NMn):
                b = 2 * math.pi * k / NMn
                nrm = u.cross(Vector((0, 0, 1)))
                V.append(C + dirA * (R + r * math.cos(b)) + nrm * (r * math.sin(b)))
        for i in range(NMj):
            i2 = (i + 1) % NMj
            for k in range(NMn):
                k2 = (k + 1) % NMn
                F.append((base + i * NMn + k, base + i * NMn + k2,
                          base + i2 * NMn + k2, base + i2 * NMn + k))
        # шарик под кольцом
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.0045)
        bb = len(V)
        for v in bm.verts:
            V.append(v.co + C - Vector((0, 0, R + 0.006)))
        for f in bm.faces:
            F.append(tuple(bb + v.index for v in f.verts))
        bm.free()
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
    bvh_all = bvh_from(merged_geo(BODY_NAMES))
    obst = [(bvh_body, "sign"), (bvh_hs, "radial")]

    cv, cf_, rows = crown_geo(bvh_all)
    r0s = rows[0]
    bv, bf = brim_geo(r0s)
    off = len(cv)
    hatV = [tilt(p) for p in cv + bv]
    hatF = cf_ + [tuple(i + off for i in f) for f in bf]
    hat = mk("Hat", hatV, hatF, ["Hat"], smooth=True)
    tv, tf = trim_geo(r0s)
    th_c = CORNERS[1]
    ci = int(round(th_c / (2 * math.pi) * NC)) % NC
    cr = rows[2][ci] + 0.001
    cen = Vector((cr * math.cos(th_c), cr * math.sin(th_c), CROWN_ROWS[2][0]))
    radial = Vector((math.cos(th_c), math.sin(th_c), 0.0))
    kv, kf = cockade_geo(cen, radial)
    tf2 = tf + [tuple(i + len(tv) for i in f) for f in kf]
    mk("Hat_Trim", [tilt(p) for p in tv + kv], tf2, ["Gold"])
    relax(hat, obst, margin=0.008)
    bvh_hat = bvh_from(world_geo(hat))

    root = cen + radial * 0.004
    fv, ff, fm = feather_geo(root)
    feather = mk("Feather", [tilt(p) for p in fv], ff, ["Feather", "Feather_Edge"], fm)
    relax(feather, obst + [(bvh_hat, "d")], margin=0.004, iters=30)

    bdv, bdf = bandana_geo(bvh_all)
    band = mk("Bandana", bdv, bdf, ["Bandana"], smooth=True)
    relax(band, obst, margin=0.003, iters=40)

    ev, ef = earring_geo()
    mk("Earring", ev, ef, ["Gold"])
    relax(bpy.data.objects["Earring"], [(bvh_body, "sign")], margin=0.004, iters=30)

    for n in PARTS:  # единые нормали
        bm = bmesh.new(); bm.from_mesh(bpy.data.objects[n].data)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(bpy.data.objects[n].data); bm.free()
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
    keep = {"Hat", "Gold", "Feather", "Feather_Edge", "Bandana"}
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
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["all"]
    mode = argv[0]
    if mode == "all":
        build()
        check()
    elif mode == "build":
        build()
    elif mode == "check":
        check()
    else:
        raise SystemExit("unknown mode " + mode)


main()
