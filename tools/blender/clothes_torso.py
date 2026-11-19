"""Одежда торса героини (v2): Shirt, Coat, Coat_Trim, Buttons, Belt, Buckle, Sash.

Длинный приталенный редингот с разрезами спереди/сзади, лацканы-отвороты, воротник-стойка,
рукава с манжетами, рубашка с воланом и кружевными манжетами, ремень с пряжкой, кушак с узлом.
Subdivision 1 на Coat/Shirt (применён), Shade Smooth + автосглаживание 40 град на всех объектах.

Запуск из корня репозитория:
  PYTHONPATH=<pylib с numpy> blender -b -P tools/blender/clothes_torso.py

Переменные окружения:
  CLO_RENDER=0   не рендерить превью (только геометрия, веса, тест деформации)

Вход (только чтение): models/adapted_body.glb (тело, волосы, арматура 22 кости).
Выход:
  source_art/clothes_torso.blend     только объекты одежды (без тела и арматуры)
  docs/models/clothes_torso_{front,side,back}.png   превью на теле с волосами

Геометрия: кольца-эллипсы по замерам сечений тела (таблица TORSO), рукава - трубки
вдоль костей рук с радиусом, измеренным по вершинам тела. Затем «прижим»: вершины,
которые оказались внутри тела или ближе порога, выносятся на офсет по нормали
ближайшей точки тела (аналог Shrinkwrap с офсетом). Веса переносятся с тела
модификатором Data Transfer (Nearest Face Interpolated), затем нормализуются.

Системы координат: вперёд = -Y, вверх = +Z, левая сторона героини = +X.
"""

import math
import os

import numpy as np
import bpy
import bmesh
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

ROOT = "/home/user/game"
BODY_GLB = os.path.join(ROOT, "models", "adapted_body.glb")
OUT_BLEND = os.path.join(ROOT, "source_art", "clothes_torso.blend")
DOCS = os.path.join(ROOT, "docs", "models")
RENDER = os.environ.get("CLO_RENDER", "1") == "1"

BONES = [
    "Hips", "Spine", "Chest", "Neck", "Head", "Shoulder.L", "UpperArm.L", "LowerArm.L",
    "Hand.L", "Shoulder.R", "UpperArm.R", "LowerArm.R", "Hand.R", "WeaponSocket.R",
    "UpperLeg.L", "LowerLeg.L", "Foot.L", "Toes.L", "UpperLeg.R", "LowerLeg.R",
    "Foot.R", "Toes.R",
]
ANGLE = 0.35
TRI_BUDGET = 7000
OFF_SHIRT = 0.018
OFF_COAT = 0.035
SLEEVE_OFF_SHIRT = 0.016
SLEEVE_OFF_COAT = 0.040
OFF_BELT = 0.042
OFF_SASH = 0.046

# Сечения торса по замеру тела: (z, полуширина по X, центр по Y, полуглубина по Y)
TORSO = [
    (0.88, 0.150, 0.012, 0.097),
    (0.93, 0.136, 0.000, 0.090),
    (0.99, 0.125, -0.024, 0.077),
    (1.05, 0.110, -0.022, 0.076),
    (1.11, 0.117, -0.012, 0.085),
    (1.17, 0.133, -0.007, 0.107),
    (1.23, 0.175, 0.016, 0.094),
    (1.29, 0.178, 0.029, 0.076),
    (1.35, 0.150, 0.032, 0.055),
    (1.40, 0.068, 0.005, 0.072),
]
_ZS = np.array([r[0] for r in TORSO])
_AS = np.array([r[1] for r in TORSO])
_CS = np.array([r[2] for r in TORSO])
_BS = np.array([r[3] for r in TORSO])

SHIRT_ZS = [0.88, 0.93, 0.99, 1.05, 1.11, 1.17, 1.23, 1.29, 1.35, 1.40]
COAT_ZS = [0.66, 0.72, 0.78, 0.84, 0.88, 0.93, 0.99, 1.02, 1.05, 1.11, 1.17,
           1.23, 1.29, 1.35, 1.40]


def log(*a):
    print("[torso]", *a)


def ang(n):
    return [2.0 * math.pi * k / n for k in range(n)]


def torso_at(z):
    return (float(np.interp(z, _ZS, _AS)), float(np.interp(z, _ZS, _CS)),
            float(np.interp(z, _ZS, _BS)))


def garment_params(z, off, flare=0.0):
    """Эллипс одежды на высоте z. flare - расклёш юбки ниже бёдер."""
    a, cy, b = torso_at(z)
    # у шеи одежда плотнее: офсет плавно уменьшается выше 1.28 м
    off = off * (1.0 - 0.5 * min(1.0, max(0.0, (z - 1.28) / 0.12)))
    f = flare * min(1.0, max(0.0, (0.88 - z) / 0.22))
    return a + off + f, cy, b + off + 0.5 * f


def ell_pt(z, t, off, flare=0.0):
    a, cy, b = garment_params(z, off, flare)
    p = Vector((a * math.cos(t), cy + b * math.sin(t), z))
    n = Vector((math.cos(t) / a, math.sin(t) / b, 0.0)).normalized()
    return p, n


def front_pt(x, z, off):
    """Точка на передней стороне эллипса при заданном x, с нормалью наружу."""
    a, cy, b = garment_params(z, off)
    y = cy - b * math.sqrt(max(1.0 - (x / a) ** 2, 0.0))
    n = Vector((x / a ** 2, (y - cy) / b ** 2, 0.0)).normalized()
    return Vector((x, y, z)), n


def w_open(z):
    """Полуширина расстёгнутого переда камзола на высоте z."""
    return max(0.02, 0.02 + 0.2 * (z - 1.0))


# ---------------------------------------------------------------- сцена и тело
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=BODY_GLB)
body = bpy.data.objects["Body"]
arm = bpy.data.objects["Armature"]
stray = bpy.data.objects.get("Icosphere")
if stray is not None:
    # Служебная сфера радиуса 1 м в начале координат закрыла бы героиню в превью;
    # в рабочей сцене её нет (сам файл тела не меняется).
    bpy.data.objects.remove(stray, do_unlink=True)


def reset_pose():
    """Bind-поза (identity). В glb после импорта скелет стоит в позе с кватернионами,
    и вычисленное тело смещается на ~0.45 м относительно сетки; сетка в bind-позе."""
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.location = (0.0, 0.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()


reset_pose()
BV = np.array([v.co[:] for v in body.data.vertices])
BODY_VERTS = [tuple(v.co) for v in body.data.vertices]
BODY_POLYS = [tuple(p.vertices) for p in body.data.polygons]
BVH_BODY = BVHTree.FromPolygons(BODY_VERTS, BODY_POLYS)



# ================================================================ v2: длинный редингот
# Тело: измерения шеи
NECK = (0.052, 0.030, 0.052)   # шея (замер среза тела на z 1.36-1.40; выше - челюсть)

ARM_BVH = {}


def arm_bvh(s):
    """BVH только по граням руки: вершины тела с весом костей руки (плечо/предплечье/кисть) > 0.5."""
    if s not in ARM_BVH:
        side = "L" if s > 0 else "R"
        want = {f"UpperArm.{side}", f"LowerArm.{side}", f"Hand.{side}"}
        gnames = {g.index: g.name for g in body.vertex_groups}
        wsum = []
        for v in body.data.vertices:
            wsum.append(sum(g.weight for g in v.groups if gnames.get(g.group) in want))
        faces = [tuple(p.vertices) for p in body.data.polygons
                 if sum(wsum[i] for i in p.vertices) / len(p.vertices) > 0.5]
        log(f"  BVH руки s={s:+d}: {len(faces)} граней")
        ARM_BVH[s] = BVHTree.FromPolygons(BODY_VERTS, faces)
    return ARM_BVH[s]


def arm_axis(s):
    S = Vector((s * 0.152, 0.0, 1.30))
    if s > 0:
        E, W = Vector((0.366, 0.0, 1.12)), Vector((0.555, -0.043, 0.962))
    else:
        E, W = Vector((-0.366, 0.0, 1.12)), Vector((-0.541, -0.102, 0.973))
    return S, E, W


# ---------------------------------------------------------------- профили пальто
ROWS_COAT = [0.50, 0.62, 0.72, 0.80, 0.88, 0.96, 1.04, 1.12, 1.20, 1.28, 1.34,
             1.38, 1.40, 1.43, 1.47]
M_COAT = 17           # сегментов в дуге (нечётно: между 10 и 11 - задний разрез)
W_Z = [0.50, 0.74, 0.84, 1.08, 1.35, 1.40, 1.43, 1.47]
W_W = [0.020, 0.020, 0.0, 0.0, 0.072, 0.056, 0.046, 0.056]


def w_coat(z):
    """Полуширина разлёта переда: разрез юбки ~4 см, потом закрыто, потом V."""
    return float(np.interp(z, W_Z, W_W))


def coat_params(z):
    if z >= 1.395:
        return {1.40: (0.082, 0.018, 0.078), 1.43: (0.072, 0.026, 0.070), 1.47: (0.078, 0.028, 0.074)}[round(z, 2)]
    if z >= 0.88:
        a, cy, b = torso_at(z)
        off = float(np.interp(z, [0.88, 1.05, 1.18, 1.28, 1.35, 1.40], [0.030, 0.024, 0.034, 0.038, 0.034, 0.024]))
        return a + off, cy, b + off
    a, cy, b = torso_at(0.88)
    u = (0.88 - z) / 0.38
    return a + 0.030 + 0.095 * u ** 1.2, cy - 0.010 * u, b + 0.030 + 0.05 * u


def row_points(z, a, cy, b, w, M):
    t0 = -math.acos(min(1.0, w / a))
    t1 = math.pi + math.acos(min(1.0, w / a))
    return [Vector((a * math.cos(t0 + (t1 - t0) * k / M), cy + b * math.sin(t0 + (t1 - t0) * k / M), z))
            for k in range(M + 1)]


def front_pt(x, z, off=0.0, params=None):
    """Точка на передней стороне эллипса одежды при заданном x, с нормалью наружу."""
    a, cy, b = (params or coat_params)(z)
    x = max(-a * 0.999, min(a * 0.999, x))
    y = cy - b * math.sqrt(max(1.0 - (x / a) ** 2, 0.0))
    n = Vector((x / a ** 2, (y - cy) / b ** 2, 0.0)).normalized()
    return Vector((x, y, z)) + n * off, n


# ---------------------------------------------------------------- примитивы
def tube(bm, rings, closed=True):
    rv = [[bm.verts.new(p) for p in ring] for ring in rings]
    for k in range(len(rv) - 1):
        A, B = rv[k], rv[k + 1]
        n = len(A)
        for i in range(n if closed else n - 1):
            j = (i + 1) % n
            bm.faces.new((A[i], A[j], B[j], B[i]))
    return rv


def merge(bm, d=1e-4):
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=d)


def tidy(bm):
    loose_v = [v for v in bm.verts if not v.link_faces]
    if loose_v:
        bmesh.ops.delete(bm, geom=loose_v, context="VERTS")


def ribbon_edge(bm, P, N, IN, width=0.010, over=0.002, closed=False):
    """Полоска-кант: лежит на ткани, смещена от края внутрь; P - точки на кромке."""
    n = len(P)
    A, B = [], []
    for i in range(n):
        A.append(bm.verts.new(P[i] + N[i] * 0.0025 - IN[i] * over))
        B.append(bm.verts.new(P[i] + N[i] * 0.0025 + IN[i] * (width - over)))
    for i in range(n if closed else n - 1):
        j = (i + 1) % n
        fn = (A[j].co - A[i].co).cross(B[i].co - A[i].co)
        bm.faces.new((A[i], A[j], B[j], B[i]) if fn.dot(N[i]) >= 0 else (A[i], B[i], B[j], A[j]))


def boundary_loops(me):
    """Замкнутые/открытые цепочки граничных рёбер сетки + вектор «внутрь ткани»."""
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    adj, inward = {}, {}
    for e in bm.edges:
        if len(e.link_faces) != 1:
            continue
        a, b = e.verts
        f = e.link_faces[0]
        T = (b.co - a.co).normalized()
        d = f.calc_center_median() - (a.co + b.co) * 0.5
        d = (d - T * d.dot(T)).normalized()
        for v, o in ((a, b), (b, a)):
            adj.setdefault(v.index, []).append(o.index)
            inward[v.index] = inward.get(v.index, Vector()) + d
    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        # идём в одну сторону до конца
        chain = [start]
        seen.add(start)
        cur, prev = start, None
        closed = False
        while True:
            nx = [x for x in adj[cur] if x != prev and x not in seen]
            if not nx:
                closed = start in adj[cur] and len(chain) > 2
                break
            prev, cur = cur, nx[0]
            seen.add(cur)
            chain.append(cur)
        pts = [bm.verts[i].co.copy() for i in chain]
        ins = [inward[i].normalized() for i in chain]
        loops.append((pts, ins, closed))
    bm.free()
    return loops


# ---------------------------------------------------------------- рукав
def sleeve_frames(s):
    S, E, W = arm_axis(s)
    ud, ld = (E - S).normalized(), (W - E).normalized()
    cs = [S + Vector((-s * 0.05, 0.0, 0.012)), S, S.lerp(E, 0.38), S.lerp(E, 0.72),
          E - ud * 0.045, E, E + ld * 0.045, E.lerp(W, 0.55), W - ld * 0.045]
    return cs, ld


def frame_at(cs, i):
    if i == 0:
        T = cs[1] - cs[0]
    elif i == len(cs) - 1:
        T = cs[-1] - cs[-2]
    else:
        T = cs[i + 1] - cs[i - 1]
    T = T.normalized()
    ref = Vector((0, 1, 0)) if abs(T.y) < 0.9 else Vector((1, 0, 0))
    U = T.cross(ref).normalized()
    V = T.cross(U).normalized()
    return T, U, V


def ray_radius(s, c, d, default):
    loc, nrm, _, dist = arm_bvh(s).ray_cast(c, d, 0.25)
    return dist if loc is not None else default


RCAP = [0.060, 0.105, 0.100, 0.090, 0.078, 0.074, 0.068, 0.060, 0.056]


def ring_radii(s, cs, n, offs, ld):
    """Радиусы колец по лучам из оси к поверхности руки (BVH); ось уточняется по центроиду
    попаданий (2 прохода), радиусы сглаживаются. Возвращает (R, новые центры)."""
    cs = [c.copy() for c in cs]
    R = []
    for i in range(len(cs)):
        T, U, V = frame_at(cs, i)
        j0 = max(i, 1)
        for _ in range(2):
            hits = []
            for t in ang(n):
                d = (U * math.cos(t) + V * math.sin(t)).normalized()
                loc, nrm, _, dist = arm_bvh(s).ray_cast(cs[j0], d, 0.2)
                if loc is not None:
                    hits.append(loc)
            if len(hits) >= 0.75 * n:
                cen = sum(hits, Vector()) / len(hits)
                dlt = cen - cs[j0]
                dlt -= T * dlt.dot(T)
                cs[j0] = cs[j0] + dlt
        row = []
        for t in ang(n):
            d = (U * math.cos(t) + V * math.sin(t)).normalized()
            loc, nrm, _, dist = arm_bvh(s).ray_cast(cs[j0], d, 0.2)
            row.append(min(dist, RCAP[min(i, len(RCAP) - 1)]) if loc is not None else 0.06)
        R.append(row)
        if i == 0:
            cs[0] = cs[0]
    cs[0] = cs[0] + (cs[1] - cs[0]) * 0.0
    R[0] = list(R[1])
    R = np.array(R)
    # дилатация (максимум по соседним кольцам и секторам) - рукав накрывает выпуклости руки
    P = np.pad(R, ((1, 1), (0, 0)), mode="edge")
    R = np.maximum(np.maximum(P[:-2], P[1:-1]), P[2:])
    R = np.maximum(np.maximum(np.roll(R, 1, 1), R), np.roll(R, -1, 1))
    R = 0.5 * R + 0.5 * np.maximum(R, 0.0)
    return R + np.array(offs)[:, None], cs


def _rings_from(s, cs0, n, offs, ld, ruffle=False):
    R, cs = ring_radii(s, cs0, n, offs, ld)
    if os.environ.get("CLO_DBG"):
        for i in range(len(cs)):
            log("   R", s, i, tuple(round(c, 3) for c in cs[i]), "min/max", round(R[i].min(), 3), round(R[i].max(), 3))
    rings = []
    for i, c in enumerate(cs):
        T, U, V = frame_at(cs, i)
        rings.append([c + (U * math.cos(t) + V * math.sin(t)) * (R[i][j] + (0.006 if (ruffle and i == len(cs) - 1 and j % 2 == 0) else 0.0))
                      for j, t in enumerate(ang(n))])
    return rings


def sleeve_rings(s, n=10):
    cs, ld = sleeve_frames(s)
    offs = [0.016, 0.026, 0.026, 0.022, 0.026, 0.022, 0.018, 0.014, 0.012]
    return _rings_from(s, cs, n, offs, ld)


def cuff_rings(s, n=10):
    S, E, W = arm_axis(s)
    ld = (W - E).normalized()
    cs = [W - ld * 0.115, W - ld * 0.06, W + ld * 0.004]
    return _rings_from(s, cs, n, [0.030, 0.040, 0.046], ld)


def shirt_cuff_rings(s, n=8):
    S, E, W = arm_axis(s)
    ld = (W - E).normalized()
    cs = [W - ld * 0.07, W - ld * 0.02, W + ld * 0.022]
    return _rings_from(s, cs, n, [0.009, 0.011, 0.020], ld, ruffle=True)


# ---------------------------------------------------------------- лацкан
LAP_Z = [1.385, 1.32, 1.24, 1.16, 1.10]
LAP_X = [0.100, 0.125, 0.140, 0.115, 0.050]
LAP_H = [0.006, 0.012, 0.014, 0.008]


def lapel_xz(s, k, u):
    z = LAP_Z[k]
    xi = max(w_coat(z) + 0.004, 0.004)
    xo = max(LAP_X[k], xi + 0.03)
    return s * (xi + (xo - xi) * u), z


def lapel(bm, s):
    levels = []
    for k in range(len(LAP_Z)):
        front, back = [], []
        for ui, u in enumerate((0.0, 0.35, 0.7, 1.0)):
            x, z = lapel_xz(s, k, u)
            p, n = front_pt(x, z, 0.0)
            front.append(p + n * LAP_H[ui])
            back.append(p + n * 0.002)
        levels.append(front + back[::-1])
    rv = [[bm.verts.new(p) for p in ring] for ring in levels]
    for k in range(len(rv) - 1):
        A, B = rv[k], rv[k + 1]
        for i in range(8):
            j = (i + 1) % 8
            bm.faces.new((A[i], A[j], B[j], B[i]) if s > 0 else (A[i], B[i], B[j], A[j]))
    for ring in (rv[0], rv[-1]):
        bm.faces.new(ring if (s > 0) == (ring is rv[-1]) else ring[::-1])


# ---------------------------------------------------------------- построители
def build_coat():
    bm = bmesh.new()
    rings = []
    for z in ROWS_COAT:
        a, cy, b = coat_params(z)
        rings.append(row_points(z, a, cy, b, w_coat(z), M_COAT))
    rv = tube(bm, rings, closed=False)
    merge(bm)
    # задний разрез: грани между вершинами M//2 и M//2+1 ниже 0.84
    gap = [f for f in bm.faces if f.calc_center_median().z < 0.835 and f.calc_center_median().y > 0.05
           and abs(f.calc_center_median().x) < 0.01]
    bmesh.ops.delete(bm, geom=gap, context="FACES")
    tidy(bm)
    dl = bm.verts.layers.deform.verify()
    for s in (+1, -1):
        rings = sleeve_rings(s)
        rv = tube(bm, rings)
        bm.faces.new(rv[0] if s > 0 else rv[0][::-1])    # крышка у плеча (внутри торса)
        for ring in rv:
            for v in ring:
                v[dl][1] = 1.0                           # группа _slv: рукав (обтекание рукой)
        cv = tube(bm, cuff_rings(s))
        for ring in cv:
            for v in ring:
                v[dl][0] = 1.0                           # группа _cuff: манжета
        lapel(bm, s)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    return bm


def build_shirt():
    bm = bmesh.new()
    zs = [1.00, 1.09, 1.17, 1.24, 1.30, 1.35, 1.39, 1.43]
    rings = []
    for z in zs:
        if z >= 1.43:
            a, cy, b = NECK[0] + 0.012, NECK[1], NECK[2] + 0.012
        else:
            a, cy, b = torso_at(z)
            a, b = a + 0.014, b + 0.014
        w = float(np.interp(z, [1.0, 1.20, 1.30, 1.40, 1.43], [0.0, 0.0, 0.024, 0.048, 0.052]))
        rings.append(row_points(z, a, cy, b, w, 16))
    tube(bm, rings, closed=False)
    merge(bm)
    tidy(bm)
    for s in (+1, -1):
        tube(bm, shirt_cuff_rings(s))
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    return bm


def shirt_frill(bm):
    """Волан вдоль V-выреза рубашки (белая оборка, после subsurf)."""
    def sp(x, z):
        if z >= 1.43:
            a, cy, b = NECK[0] + 0.012, NECK[1], NECK[2] + 0.012
        else:
            a, cy, b = torso_at(z)
            a, b = a + 0.014, b + 0.014
        x = max(-a * 0.999, min(a * 0.999, x))
        y = cy - b * math.sqrt(max(1 - (x / a) ** 2, 0))
        n = Vector((x / a ** 2, (y - cy) / b ** 2, 0)).normalized()
        return Vector((x, y, z)), n

    for s in (+1, -1):
        K = 14
        rows = []
        for k in range(K):
            z = 1.40 - 0.19 * k / (K - 1)
            w = float(np.interp(z, [1.0, 1.20, 1.30, 1.40, 1.43], [0.0, 0.0, 0.024, 0.048, 0.052]))
            row = []
            for ui, u in enumerate((-1.0, -0.33, 0.33, 1.0)):
                p, n = sp(s * (w + u * 0.016), z)
                lift = 0.003 + 0.004 * (1 + math.sin(k * 1.9 + ui * 2.1)) * 0.5 + 0.006 * (ui >= 2)
                row.append(p + n * lift)
            rows.append(row)
        rv = [[bm.verts.new(p) for p in r] for r in rows]
        for k in range(K - 1):
            for i in range(3):
                bm.faces.new((rv[k][i], rv[k][i + 1], rv[k + 1][i + 1], rv[k + 1][i]) if s > 0
                             else (rv[k][i], rv[k + 1][i], rv[k + 1][i + 1], rv[k][i + 1]))


def ray_front(bvh, x, z):
    """Попадание луча спереди (+Y) в пальто; нормаль всегда наружу (к -Y)."""
    loc, nrm, _, _ = bvh.ray_cast(Vector((x, -0.9, z)), Vector((0, 1, 0)))
    if loc is not None and nrm.y > 0:
        nrm = -nrm
    return loc, nrm


def build_buttons(coat_bvh):
    bm = bmesh.new()
    spots = [(s * 0.050, z) for s in (+1, -1) for z in (0.935, 0.875)]
    for x, z in spots:
        loc, nrm = ray_front(coat_bvh, x, z)
        if loc is None:
            continue
        M = Matrix.Translation(loc + nrm * 0.003) @ nrm.to_track_quat("Z", "Y").to_matrix().to_4x4()
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=8, radius1=0.0085, radius2=0.0065,
                              depth=0.008, matrix=M)
    return bm


def surf_r(coat_bvh, z, t, cy):
    """Расстояние от оси талии до пальто (луч), если нет - до тела + 3 см."""
    d = Vector((math.cos(t), math.sin(t), 0.0))
    o = Vector((0.0, cy, z))
    loc, _, _, dist = coat_bvh.ray_cast(o, d, 1.0)
    if loc is not None:
        return dist
    loc, _, _, dist = BVH_BODY.ray_cast(o, d, 1.0)
    return (dist + 0.028) if loc is not None else 0.2


def build_belt(coat_bvh):
    bm = bmesh.new()
    zt, zb, n = 1.012, 0.977, 32
    cy = torso_at(0.99)[1]
    rows = []
    for t in ang(n):
        ri = surf_r(coat_bvh, 0.994, t, cy) + 0.0015
        d = Vector((math.cos(t), math.sin(t), 0.0))
        c = Vector((0.0, cy, 0.0))
        rows.append([c + d * (ri + 0.006) + Vector((0, 0, zt)), c + d * (ri + 0.006) + Vector((0, 0, zb)),
                     c + d * ri + Vector((0, 0, zb)), c + d * ri + Vector((0, 0, zt))])
    V = [[bm.verts.new(p) for p in r] for r in rows]
    for i in range(n):
        j = (i + 1) % n
        for k in range(4):
            kk = (k + 1) % 4
            bm.faces.new((V[i][k], V[j][k], V[j][kk], V[i][kk]))
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    return bm


def rr_loop(w, h, r, k=3):
    pts = []
    for (cx, cz, a0) in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90),
                         (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for i in range(k + 1):
            a = math.radians(a0 + 90.0 * i / k)
            pts.append((cx + r * math.cos(a), cz + r * math.sin(a)))
    return pts


def build_buckle(coat_bvh):
    """Большая золотая пряжка: рамка (скруглённая), язычок, перемычка."""
    bm = bmesh.new()
    loc, nrm = ray_front(coat_bvh, 0.0, 0.994)
    if loc is None:
        loc, nrm = Vector((0.0, -0.17, 0.994)), Vector((0, -1, 0))
    base = loc + nrm * 0.012
    ow, oh, iw, ih = 0.074, 0.050, 0.052, 0.030
    outer, inner = rr_loop(ow, oh, 0.012), rr_loop(iw, ih, 0.006)
    d = 0.007

    def P(x, z, y):
        return Vector((base.x + x, base.y - y, base.z + z))   # y>0 - наружу (к -Y)

    layers = {}
    for name, loop, yy in (("of", outer, d), ("ob", outer, 0.0), ("if", inner, d), ("ib", inner, 0.0)):
        layers[name] = [bm.verts.new(P(x, z, yy)) for x, z in loop]
    N = len(outer)
    for i in range(N):
        j = (i + 1) % N
        bm.faces.new((layers["of"][i], layers["of"][j], layers["if"][j], layers["if"][i]))   # лицо
        bm.faces.new((layers["ob"][j], layers["ob"][i], layers["ib"][i], layers["ib"][j]))   # тыл
        bm.faces.new((layers["ob"][i], layers["ob"][j], layers["of"][j], layers["of"][i]))   # внешняя стенка
        bm.faces.new((layers["if"][i], layers["if"][j], layers["ib"][j], layers["ib"][i]))   # внутренняя
    # перемычка-ось слева и язычок
    box_c = P(-iw / 2 + 0.004, 0.0, d * 0.5)
    bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation(box_c) @ Matrix.Diagonal(Vector((0.008, 0.006, ih + 0.004))).to_4x4())
    tc = P(0.004, 0.0, d + 0.001)
    bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation(tc) @ Matrix.Diagonal(Vector((iw - 0.004, 0.0035, 0.006))).to_4x4())
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    return bm


def build_sash(coat_bvh):
    bm = bmesh.new()
    n = 18
    cy = torso_at(1.08)[1]
    c = Vector((0.0, cy, 0.0))
    rows = []
    for i, t in enumerate(ang(n)):
        ri = surf_r(coat_bvh, 1.085, t, cy) + 0.003
        zb = 1.050 + 0.007 * math.sin(5 * t + 0.6)
        zt = 1.122 + 0.007 * math.sin(4 * t + 2.0)
        zm = 0.5 * (zb + zt)
        bulge = 0.005 + 0.004 * (1 + math.sin(7 * t)) * 0.5
        d = Vector((math.cos(t), math.sin(t), 0.0))
        o = ri + 0.007 + bulge
        rows.append([c + d * (ri + 0.007) + Vector((0, 0, zt)), c + d * o + Vector((0, 0, zm)),
                     c + d * (ri + 0.007) + Vector((0, 0, zb)), c + d * ri + Vector((0, 0, zb)),
                     c + d * ri + Vector((0, 0, zm)), c + d * ri + Vector((0, 0, zt))])
    V = [[bm.verts.new(p) for p in r] for r in rows]
    for i in range(n):
        j = (i + 1) % n
        for k in range(6):
            kk = (k + 1) % 6
            bm.faces.new((V[i][k], V[j][k], V[j][kk], V[i][kk]))

    def hit(x, z, off):
        loc, nrm = ray_front(coat_bvh, x, z)
        if loc is None:
            return Vector((x, -0.15, z)), Vector((0, -1, 0))
        return loc + nrm * off, nrm

    # узел на левом боку спереди
    kp, kn = hit(0.105, 1.086, 0.020)
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0,
                               matrix=Matrix.Translation(kp) @ Matrix.Diagonal(Vector((0.028, 0.017, 0.030))).to_4x4())
    for ang_z, sx in ((35, +1), (-35, -1)):
        M = Matrix.Translation(kp + Vector((sx * 0.026, -0.003, 0.012))) @ Matrix.Rotation(math.radians(ang_z), 4, "Y") \
            @ Matrix.Diagonal(Vector((0.034, 0.012, 0.016))).to_4x4()
        bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=4, radius=1.0, matrix=M)
    # два хвоста до колена: мягкие волнистые ленты на поверхности пальто
    for (x0, x1, zend, w0, ph) in ((0.098, 0.150, 0.540, 0.048, 0.0), (0.088, 0.112, 0.610, 0.042, 1.7)):
        K = 11
        rows = []
        for k in range(K):
            t = k / (K - 1)
            z = 1.070 - (1.070 - zend) * t
            x = x0 + (x1 - x0) * t + 0.012 * math.sin(3.4 * t + ph)
            w = w0 * (1.0 + 0.15 * t)
            row = []
            for ui, u in enumerate((-1.0, -0.33, 0.33, 1.0)):
                zz = z + (0.03 if (k == K - 1 and ui in (1, 2)) else 0.0)     # ласточкин хвост
                p, nn = hit(x + u * w * 0.5, zz, 0.0)
                bump = 0.003 * math.sin(6.0 * t + ui * 1.3 + ph)
                row.append((p, nn, bump))
            rows.append(row)
        for layer, (dz, flip) in enumerate(((0.016, False), (0.0135, True))):
            rv = [[bm.verts.new(p + nn * (dz + bump)) for p, nn, bump in r] for r in rows]
            for k in range(K - 1):
                for i in range(3):
                    f = (rv[k][i], rv[k][i + 1], rv[k + 1][i + 1], rv[k + 1][i])
                    bm.faces.new(f[::-1] if flip else f)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    return bm


# ---------------------------------------------------------------- прижим, материалы, веса
def _nearest_signed(co):
    """Ближайшая точка тела (для рук - минимум по телу и по BVH руки) и знаковый зазор."""
    best = None
    bvhs = [BVH_BODY]
    if abs(co.x) > 0.10 and co.z > 0.85:
        bvhs.append(arm_bvh(1 if co.x > 0 else -1))
    for bv in bvhs:
        loc, nrm, _, _ = bv.find_nearest(co)
        if loc is None:
            continue
        d = (co - loc).dot(nrm)
        if best is None or d < best[0]:
            best = (d, loc, nrm)
    return best


def clearance_fit(bm, target, min_clear, passes=0):
    """Вершины и центры граней ближе min_clear к телу выносятся по нормали тела."""
    moved = 0
    bm.verts.ensure_lookup_table()
    for v in bm.verts:
        if v.co.z > 1.405 and abs(v.co.x) < 0.12:      # воротник у шеи: рядом челюсть, не прижимаем
            continue
        r = _nearest_signed(v.co)
        if r and r[0] < min_clear:
            v.co = r[1] + r[2] * target
            moved += 1
    for _ in range(passes):
        bm.verts.ensure_lookup_table()
        bm.normal_update()
        d = np.zeros(len(bm.verts))
        for v in bm.verts:
            if v.co.z > 1.405 and abs(v.co.x) < 0.12:
                continue
            r = _nearest_signed(v.co)
            if r:
                d[v.index] = max(d[v.index], min_clear - r[0])
        for f in bm.faces:
            c = f.calc_center_median()
            if c.z > 1.405 and abs(c.x) < 0.12:
                continue
            r = _nearest_signed(c)
            if r and r[0] < min_clear:
                for v in f.verts:
                    d[v.index] = max(d[v.index], (min_clear - r[0]) * 1.3)
        for _s in range(2):                      # сглаживаем дефицит по соседям - без шипов
            nb = np.zeros_like(d)
            cnt = np.zeros_like(d)
            for e in bm.edges:
                a_, b_ = e.verts[0].index, e.verts[1].index
                nb[a_] += d[b_]
                nb[b_] += d[a_]
                cnt[a_] += 1
                cnt[b_] += 1
            d = np.maximum(d * 0.6, 0.5 * d + 0.5 * nb / np.maximum(cnt, 1))
        for v in bm.verts:
            if d[v.index] > 1e-4:
                v.co += v.normal * float(d[v.index])
        moved += int((d > 1e-4).sum())
    return moved


def make_object(name, bm, mat_name):
    me = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    me.validate()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    me.materials.append(bpy.data.materials.get(mat_name))
    return ob


def orient_outward(ob):
    """Грани смотрят от тела (открытые оболочки recalc не ориентирует надёжно)."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    flip = []
    for f in bm.faces:
        c = f.calc_center_median()
        loc, nrm, _, _ = BVH_BODY.find_nearest(c)
        if f.normal.dot(c - loc) < 0:
            flip.append(f)
    bmesh.ops.reverse_faces(bm, faces=flip)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    return len(flip)


def subsurf_apply(ob, levels=1):
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    mod = ob.modifiers.new("Subd", "SUBSURF")
    mod.levels = levels
    mod.render_levels = levels
    bpy.ops.object.modifier_apply(modifier=mod.name)
    ob.select_set(False)


def refit(ob, target, min_clear):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    n = clearance_fit(bm, target, min_clear)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    return n


def sleeve_fix(ob, min_clear=0.014, iters=8):
    """Рукава: вершины и центры граней глубже min_clear от руки выносятся по нормали руки
    (вектора смещения сглаживаются по соседям, чтобы не было шипов)."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    tot = 0
    for _ in range(iters):
        bm.verts.ensure_lookup_table()
        P = np.zeros((len(bm.verts), 3))
        act = np.zeros(len(bm.verts), bool)

        def deficit(co):
            sh = False          # плечевая зона: торс + рука
            if not (sh or (abs(co.x) >= 0.19 and co.z >= 0.85)):
                return None
            best = None
            for bv in (BVH_BODY, arm_bvh(1 if co.x > 0 else -1)):
                loc, nrm, _, _ = bv.find_nearest(co)
                if loc is not None:
                    dd = (co - loc).dot(nrm)
                    if best is None or dd < best[0]:
                        best = (dd, nrm)
            return None if best is None else (min_clear - best[0], best[1])
        for v in bm.verts:
            r = deficit(v.co)
            if r is not None:
                act[v.index] = True
                if r[0] > 0:
                    P[v.index] = np.array(r[1]) * r[0]
        for f in bm.faces:
            r = deficit(f.calc_center_median())
            if r is not None and r[0] > -0.012:
                for v in f.verts:
                    q = np.array(r[1]) * (r[0] + 0.012) * 1.3
                    if np.linalg.norm(q) > np.linalg.norm(P[v.index]):
                        P[v.index] = q
        if np.abs(P).max() < 0.002:
            break
        nb = np.zeros_like(P)
        cnt = np.zeros(len(P))
        for e in bm.edges:
            a_, b_ = e.verts[0].index, e.verts[1].index
            nb[a_] += P[b_]
            nb[b_] += P[a_]
            cnt[a_] += 1
            cnt[b_] += 1
        Q = 0.6 * P + 0.4 * nb / np.maximum(cnt, 1)[:, None]
        Q = np.where((np.linalg.norm(Q, axis=1) < np.linalg.norm(P, axis=1))[:, None], 0.5 * (Q + P), Q)
        for v in bm.verts:
            if act[v.index]:
                dv = Q[v.index]
                n_ = np.linalg.norm(dv)
                if n_ > 1e-4:
                    v.co += Vector(dv * min(1.0, 0.02 / n_))
                    tot += 1
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    return tot


def wrap_sleeves(ob):
    """Аналог Shrinkwrap с офсетом: вершины рукава (группа _slv) переносятся на ближайшую
    точку руки + офсет, зависящий от положения вдоль руки (запас у локтя, туго у запястья)."""
    gi = {g.name: g.index for g in ob.vertex_groups}
    me = ob.data
    n = 0
    for v in me.vertices:
        w = {g.group: g.weight for g in v.groups}
        if w.get(gi["_cuff"], 0.0) > 0.5 or w.get(gi["_slv"], 0.0) < 0.5:
            continue
        s = 1 if v.co.x > 0 else -1
        S, E, W = arm_axis(s)
        ud, ld = (E - S).normalized(), (W - E).normalized()
        if (v.co - S).dot(ud) < 0.03:                 # плечевая крышка - не трогаем
            continue
        tw = (v.co - E).dot(ld)
        if tw > 0:
            t = min(1.0, tw / (W - E).length)
            off = 0.034 - 0.017 * (t * t * (3 - 2 * t))
        else:
            t = min(1.0, max(0.0, (v.co - S).dot(ud) / (E - S).length))
            off = 0.022 + 0.012 * (t * t * (3 - 2 * t))
        loc, nrm, _, _ = arm_bvh(s).find_nearest(v.co)
        if loc is not None:
            v.co = loc + nrm * off
            n += 1
    me.update()
    return n


def smooth_shade(ob, hard=False):
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.shade_smooth()
    ob.data.use_auto_smooth = True
    ob.data.auto_smooth_angle = math.radians(40.0)
    ob.select_set(False)


def make_materials():
    specs = {
        "Coat": ((0.13, 0.008, 0.025, 1), 0.0, 0.5),
        "Shirt": ((0.96, 0.95, 0.92, 1), 0.0, 0.7),
        "Belt": ((0.30, 0.17, 0.09, 1), 0.0, 0.5),
        "Sash": ((0.24, 0.012, 0.02, 1), 0.0, 0.55),
        "Gold": ((1.0, 0.72, 0.28, 1), 1.0, 0.35),
    }
    for name, (col, metal, rough) in specs.items():
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        bsdf = m.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = col
        bsdf.inputs["Metallic"].default_value = metal
        bsdf.inputs["Roughness"].default_value = rough


def smooth_weights(ob, iters=int(os.environ.get('CLO_SM', '8')), f=float(os.environ.get('CLO_SF', '0.5')), keep=4):
    me = ob.data
    n = len(me.vertices)
    names = [g.name for g in ob.vertex_groups]
    W = np.zeros((n, len(names)))
    for v in me.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    E = np.array([e.vertices[:] for e in me.edges]) if me.edges else np.zeros((0, 2), int)
    for _ in range(iters):
        acc = np.zeros_like(W)
        cnt = np.zeros(n)
        np.add.at(acc, E[:, 0], W[E[:, 1]])
        np.add.at(acc, E[:, 1], W[E[:, 0]])
        np.add.at(cnt, E[:, 0], 1)
        np.add.at(cnt, E[:, 1], 1)
        m = cnt > 0
        W[m] = (1 - f) * W[m] + f * acc[m] / cnt[m, None]
    for i in range(n):                         # максимум keep влияний
        row = W[i]
        if np.count_nonzero(row) > keep:
            thr = np.sort(row)[-keep]
            row[row < thr] = 0.0
        s = row.sum()
        W[i] = row / s if s > 0 else row
    ob.vertex_groups.clear()
    for nm in names:
        ob.vertex_groups.new(name=nm)
    for gi in range(len(names)):
        vg = ob.vertex_groups[gi]
        for i in np.nonzero(W[:, gi] > 0.004)[0]:
            vg.add([int(i)], float(W[i, gi]), "REPLACE")


def sleeve_weights(ob):
    """Рукава: ниже плеча веса - плавный переход UpperArm -> LowerArm вокруг локтя (окно +-7 см),
    поверх переноса с тела (кроссфейд у плеча). Так предплечье не режет рукав в локте."""
    n = 0
    for v in ob.data.vertices:
        co = v.co
        if abs(co.x) < 0.20:
            continue
        sd = 1 if co.x > 0 else -1
        S, E, W = arm_axis(sd)
        ud, ld = (E - S).normalized(), (W - E).normalized()
        u = (co - S).dot(ud) / (E - S).length
        if u < 0.30:
            continue
        # расстояние до полилинии руки
        def seg_d(A, B):
            AB = B - A
            t = max(0.0, min(1.0, (co - A).dot(AB) / AB.dot(AB)))
            return (co - (A + AB * t)).length
        if min(seg_d(S, E), seg_d(E, W)) > 0.20:
            continue
        bis = (ud + ld).normalized()
        sg = (co - E).dot(bis)
        k = max(0.0, min(1.0, (sg + 0.07) / 0.14))
        wl = k * k * (3 - 2 * k)
        a = max(0.0, min(1.0, (u - 0.30) / 0.25))
        a = a * a * (3 - 2 * a)
        side = "L" if sd > 0 else "R"
        cur = {ob.vertex_groups[g.group].name: g.weight for g in v.groups}
        proc = {f"UpperArm.{side}": 1.0 - wl, f"LowerArm.{side}": wl}
        new = {}
        for nm in set(cur) | set(proc):
            new[nm] = (1 - a) * cur.get(nm, 0.0) + a * proc.get(nm, 0.0)
        tot = sum(new.values())
        for g in list(v.groups):
            ob.vertex_groups[g.group].remove([v.index])
        for nm, w in new.items():
            if w / tot > 0.004:
                ob.vertex_groups[nm].add([v.index], w / tot, "REPLACE")
        n += 1
    return n


def transfer_weights(ob, smooth=True):
    for n in BONES:
        ob.vertex_groups.new(name=n)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    mod = ob.modifiers.new("ПереносВесов", "DATA_TRANSFER")
    mod.object = body
    mod.use_vert_data = True
    mod.data_types_verts = {"VGROUP_WEIGHTS"}
    mod.vert_mapping = "POLYINTERP_NEAREST"
    mod.layers_vgroup_select_src = "ALL"
    mod.layers_vgroup_select_dst = "NAME"
    bpy.ops.object.modifier_apply(modifier=mod.name)
    ob.select_set(False)
    me = ob.data
    for v in me.vertices:                       # пустые вершины - на Chest
        if not any(g.weight > 0 for g in v.groups):
            ob.vertex_groups["Chest"].add([v.index], 1.0, "REPLACE")
    if smooth:
        smooth_weights(ob)
    ob.modifiers.new("Скелет", "ARMATURE").object = arm


# ---------------------------------------------------------------- сборка
make_materials()
log("строю Coat (+ Subd 1)")
objs = {}
objs["Coat"] = make_object("Coat", build_coat(), "Coat")
objs["Coat"].vertex_groups.new(name="_cuff")
objs["Coat"].vertex_groups.new(name="_slv")
subsurf_apply(objs["Coat"], 1)
subsurf_n = orient_outward(objs["Coat"])
log("  прижим Coat:", refit(objs["Coat"], 0.025, 0.020), "перевёрнуто граней", subsurf_n)
log("строю Shirt (+ Subd 1)")
objs["Shirt"] = make_object("Shirt", build_shirt(), "Shirt")
subsurf_apply(objs["Shirt"], 1)
log("  прижим Shirt:", refit(objs["Shirt"], 0.012, 0.008), "перевёрнуто", orient_outward(objs["Shirt"]))
bm = bmesh.new()
bm.from_mesh(objs["Shirt"].data)
shirt_frill(bm)
bm.to_mesh(objs["Shirt"].data)
bm.free()

objs["Coat"].vertex_groups.clear()
# BVH готового пальто - для ремня, кушака, пряжки, пуговиц
_cv = [tuple(v.co) for v in objs["Coat"].data.vertices]
_cp = [tuple(p.vertices) for p in objs["Coat"].data.polygons]
COAT_BVH = BVHTree.FromPolygons(_cv, _cp)

log("строю Coat_Trim, Buttons, Belt, Buckle, Sash")
# золотой кант: граничные кромки пальто + внешний край лацканов
tbm = bmesh.new()
for pts, ins, closed in boundary_loops(objs["Coat"].data):
    ctr = sum((p for p in pts), Vector()) / len(pts)
    skip = False
    for sd in (+1, -1):                       # торец рукава спрятан под манжетой - без канта
        _S, _E, _W = arm_axis(sd)
        if (ctr - (_W - (_W - _E).normalized() * 0.045)).length < 0.025:
            skip = True
    if skip:
        continue
    # корневые кольца рукавов внутри торса пропускаем: крышка закрывает их, остаются только кромки
    P, N, I = [], [], []
    for p, i in zip(pts, ins):
        loc, nrm, _, _ = COAT_BVH.find_nearest(p)
        P.append(loc)
        N.append(nrm)
        I.append((i - nrm * i.dot(nrm)).normalized())
    if len(P) > 1:
        ribbon_edge(tbm, P, N, I, 0.009, 0.002, closed)
for s in (+1, -1):
    P, N, I = [], [], []
    for k in range(len(LAP_Z)):
        for kk in (0, 1):
            pass
    zs = np.linspace(1.385, 1.10, 9)
    for z in zs:
        z = float(z)
        xi = max(w_coat(z) + 0.004, 0.004)
        xo = max(float(np.interp(z, LAP_Z[::-1], LAP_X[::-1])), xi + 0.03)
        p, n = front_pt(s * xo, z, 0.0)
        # лацкан: к поверхности пальто + высота у внешней кромки
        loc, nrm, _, _ = COAT_BVH.find_nearest(p + n * 0.012)
        P.append(loc)
        N.append(nrm)
        inw = Vector((-s, 0, 0))
        I.append((inw - nrm * inw.dot(nrm)).normalized())
    ribbon_edge(tbm, P, N, I, 0.008, 0.0025)
objs["Coat_Trim"] = make_object("Coat_Trim", tbm, "Gold")
objs["Buttons"] = make_object("Buttons", build_buttons(COAT_BVH), "Gold")
objs["Belt"] = make_object("Belt", build_belt(COAT_BVH), "Belt")
objs["Buckle"] = make_object("Buckle", build_buckle(COAT_BVH), "Gold")
objs["Sash"] = make_object("Sash", build_sash(COAT_BVH), "Sash")
order = ["Shirt", "Coat", "Coat_Trim", "Buttons", "Belt", "Buckle", "Sash"]
objs = {k: objs[k] for k in order}

for name, ob in objs.items():
    transfer_weights(ob, smooth=(name in ("Coat", "Shirt", "Sash", "Belt")))
    if name in ("Coat", "Shirt"):
        log(f"  веса рукава {name}: переписано вершин", sleeve_weights(ob))
    smooth_shade(ob)

garments = list(objs.values())


def tri_count(ob):
    return sum(len(p.vertices) - 2 for p in ob.data.polygons)


total = 0
for name, ob in objs.items():
    t = tri_count(ob)
    total += t
    log(f"  {name}: {len(ob.data.vertices)} вершин, {t} треугольников, "
        f"групп {len(ob.vertex_groups)}, нет привязки: "
        f"{sum(1 for v in ob.data.vertices if not v.groups)}")
log(f"ИТОГО торс: {total} треугольников (бюджет {TRI_BUDGET}, "
    f"{'OK' if total <= TRI_BUDGET else 'ПРЕВЫШЕН'})")


def width_at(ob, z0, xmax=0.26):
    xs = [v.co.x for v in ob.data.vertices if abs(v.co.z - z0) < 0.012 and abs(v.co.x) < xmax]
    return (max(xs) - min(xs)) if xs else 0.0


log(f"ширина пальто на груди 1.20: {width_at(objs['Coat'], 1.20) * 100:.1f} см, "
    f"на талии 1.05: {width_at(objs['Coat'], 1.05) * 100:.1f} см, подол 0.52: {width_at(objs['Coat'], 0.52, 0.4) * 100:.1f} см")
# ---------------------------------------------------------------- тест деформации
def evaluate(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    verts = [ev.matrix_world @ v.co for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons]
    ev.to_mesh_clear()
    return verts, polys


def deform_test():
    log("тест деформации: 22 кости x 3 оси x 0.35 рад")
    bpy.context.view_layer.update()
    base = {}
    bv, bp = evaluate(body)
    bvh = BVHTree.FromPolygons(bv, bp)
    for g in garments:
        gv, _ = evaluate(g)
        base[g.name] = [bvh.find_nearest(v)[3] for v in gv]
    worst = {g.name: [0.0, "", 0.0, "", None] for g in garments}
    for bone in BONES:
        pb = arm.pose.bones[bone]
        for axis in range(3):
            e = [0.0, 0.0, 0.0]
            e[axis] = ANGLE
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = e
            bpy.context.view_layer.update()
            bv, bp = evaluate(body)
            bvh = BVHTree.FromPolygons(bv, bp)
            case = f"{bone} {'XYZ'[axis]}"
            for g in garments:
                gv, _ = evaluate(g)
                depth, gap, deep_i = 0.0, 0.0, -1
                for i, v in enumerate(gv):
                    loc, nrm, _, dist = bvh.find_nearest(v)
                    if (v - loc).dot(nrm) < 0.0 and dist > depth:
                        depth = dist
                        deep_i = i
                    gap = max(gap, dist - base[g.name][i])
                w = worst[g.name]
                if depth > w[0]:
                    w[0], w[1] = depth, case
                    w[4] = (deep_i, tuple(round(c, 3) for c in gv[deep_i]),
                            [(g.vertex_groups[x.group].name, round(x.weight, 2)) for x in g.data.vertices[deep_i].groups])
                if gap > w[2]:
                    w[2], w[3] = gap, case
            pb.rotation_euler = (0.0, 0.0, 0.0)
    reset_pose()
    for name, (depth, dcase, gap, gcase, deep) in worst.items():
        log(f"  {name}: макс. проникновение {depth * 100:.1f} см ({dcase or '-'}), "
            f"макс. отрыв +{gap * 100:.1f} см ({gcase or '-'})")
        if deep:
            log(f"    глубокая вершина {deep[0]} {deep[1]} веса {deep[2]}")


deform_test()


# ---------------------------------------------------------------- превью
def render_views():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x = 540
    scene.render.resolution_y = 810
    scene.render.image_settings.file_format = "PNG"
    world = bpy.data.worlds.new("Фон")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.57, 0.62, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.9
    scene.world = world
    sun = bpy.data.lights.new("Солнце", "SUN")
    sun.energy = 3.0
    sun_ob = bpy.data.objects.new("Солнце", sun)
    bpy.context.scene.collection.objects.link(sun_ob)
    sun_ob.rotation_euler = (math.radians(50), 0.0, math.radians(-30))
    fill = bpy.data.lights.new("Заполнение", "AREA")
    fill.energy = 150.0
    fill.size = 3.0
    fill_ob = bpy.data.objects.new("Заполнение", fill)
    bpy.context.scene.collection.objects.link(fill_ob)
    fill_ob.location = (-2.5, -3.0, 2.2)
    fill_ob.rotation_euler = (math.radians(60), 0.0, math.radians(-40))

    cam_data = bpy.data.cameras.new("Камера")
    cam_data.lens = 70
    cam = bpy.data.objects.new("Камера", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    scene.camera = cam
    target = Vector((0.0, 0.0, 0.86))
    views = {
        "front": Vector((0.0, -3.4, 0.86)),
        "side": Vector((3.4, 0.0, 0.86)),
        "back": Vector((0.0, 3.4, 0.86)),
    }
    os.makedirs(DOCS, exist_ok=True)
    if os.environ.get("CLO_CLOSE"):
        v = [float(x) for x in os.environ["CLO_CLOSE"].split(",")]
        cam.location = Vector(v[:3])
        cam.rotation_euler = (Vector(v[3:]) - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam_data.lens = 100
        scene.render.filepath = os.path.join(os.environ.get("CLO_OUT", "/tmp"), "close.png")
        bpy.ops.render.render(write_still=True)
        return
    for name, loc in views.items():
        cam.location = loc
        d = target - loc
        cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = os.path.join(DOCS, f"clothes_torso_{name}.png")
        bpy.ops.render.render(write_still=True)
        log("превью:", scene.render.filepath)


if RENDER:
    render_views()


# ---------------------------------------------------------------- сохранение
for ob in list(bpy.data.objects):
    if ob.name not in objs:
        bpy.data.objects.remove(ob, do_unlink=True)
for ob in objs.values():
    for m in list(ob.modifiers):
        ob.modifiers.remove(m)
bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)
log("сохранено:", OUT_BLEND)
