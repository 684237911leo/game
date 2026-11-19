"""Одежда торса героини: Shirt, Coat, Coat_Trim, Buttons, Belt, Buckle, Sash.

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
TRI_BUDGET = 4500
OFF_SHIRT = 0.018
OFF_COAT = 0.040
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
    (1.23, 0.190, 0.016, 0.094),
    (1.29, 0.195, 0.029, 0.076),
    (1.35, 0.188, 0.032, 0.049),
    (1.40, 0.066, 0.005, 0.072),
    (1.44, 0.072, -0.006, 0.085),
]
_ZS = np.array([r[0] for r in TORSO])
_AS = np.array([r[1] for r in TORSO])
_CS = np.array([r[2] for r in TORSO])
_BS = np.array([r[3] for r in TORSO])

SHIRT_ZS = [0.88, 0.93, 0.99, 1.05, 1.11, 1.17, 1.23, 1.29, 1.35, 1.40]
COAT_ZS = [0.66, 0.72, 0.78, 0.84, 0.88, 0.93, 0.99, 1.02, 1.05, 1.11, 1.17,
           1.23, 1.29, 1.35, 1.40, 1.44]


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

BV = np.array([v.co[:] for v in body.data.vertices])
BODY_VERTS = [tuple(v.co) for v in body.data.vertices]
BODY_POLYS = [tuple(p.vertices) for p in body.data.polygons]
BVH_BODY = BVHTree.FromPolygons(BODY_VERTS, BODY_POLYS)


def arm_radius(s, A, B):
    """Радиус руки по вершинам тела: 90-й перцентиль расстояния до оси сегмента."""
    AB = B - A
    L2 = AB.dot(AB)
    ds = []
    for p in BV:
        if s * p[0] < 0.20 or not (0.9 <= p[2] <= 1.36):
            continue
        P = Vector(p)
        t = (P - A).dot(AB) / L2
        if 0.3 <= t <= 0.7:
            d = (P - (A + AB * t)).length
            if d < 0.08:
                ds.append(d)
    r = float(np.percentile(ds, 90)) if ds else 0.05
    log(f"  радиус руки s={s:+d} отрезок {tuple(round(c, 2) for c in A)}->{tuple(round(c, 2) for c in B)}: "
        f"{r:.3f} м по {len(ds)} вершинам")
    return r


ARM_CACHE = {}


def arm_pts(s):
    """Оси рук (как у костей UpperArm/LowerArm) и радиусы по сегментам."""
    if s in ARM_CACHE:
        return ARM_CACHE[s]
    S = Vector((s * 0.152, 0.0, 1.30))
    if s > 0:
        E, W = Vector((0.366, 0.0, 1.12)), Vector((0.555, -0.043, 0.962))
    else:
        E, W = Vector((-0.366, 0.0, 1.12)), Vector((-0.541, -0.102, 0.973))
    rU = arm_radius(s, S, E)
    rL = arm_radius(s, E, W)
    C = [S, S.lerp(E, 0.33), S.lerp(E, 0.66), E, E.lerp(W, 0.5), W]
    R = [rU + 0.02, rU, rU * 0.98, rU * 0.96, rL, rL * 0.85]
    ARM_CACHE[s] = (C, R)
    return C, R


def sleeve_rings(s, off, n=12):
    C, R = arm_pts(s)
    rings, frames = [], []
    for i, c in enumerate(C):
        if i == 0:
            T = C[1] - C[0]
        elif i == len(C) - 1:
            T = C[-1] - C[-2]
        else:
            T = C[i + 1] - C[i - 1]
        T.normalize()
        ref = Vector((0, 1, 0)) if abs(T.y) < 0.9 else Vector((1, 0, 0))
        U = T.cross(ref).normalized()
        V = T.cross(U).normalized()
        r = R[i] + off
        pts = [c + (U * math.cos(t) + V * math.sin(t)) * r for t in ang(n)]
        rings.append(pts)
        frames.append((c, U, V, r))
    return rings, frames


def torso_rings(zs, n, off, flare=0.0):
    return [[ell_pt(z, t, off, flare)[0] for t in ang(n)] for z in zs]


# ---------------------------------------------------------------- построители
def tube(bm, rings):
    rv = [[bm.verts.new(p) for p in ring] for ring in rings]
    for k in range(len(rv) - 1):
        A, B = rv[k], rv[k + 1]
        n = len(A)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((A[i], A[j], B[j], B[i]))
    return rv


def drop_faces(bm, pred):
    fs = [f for f in bm.faces if pred(f.calc_center_median())]
    bmesh.ops.delete(bm, geom=fs, context="FACES")
    tidy(bm)


def tidy(bm):
    loose_v = [v for v in bm.verts if not v.link_faces]
    if loose_v:
        bmesh.ops.delete(bm, geom=loose_v, context="VERTS")
    loose_e = [e for e in bm.edges if not e.link_faces]
    if loose_e:
        bmesh.ops.delete(bm, geom=loose_e, context="EDGES")


def ribbon(bm, pts, nrms, width, closed=False):
    """Плоская полоса вдоль полилинии; ширина перпендикулярна касательной и нормали."""
    n = len(pts)
    L, R = [], []
    for i in range(n):
        if closed:
            prv, nxt = pts[(i - 1) % n], pts[(i + 1) % n]
        else:
            prv, nxt = pts[max(i - 1, 0)], pts[min(i + 1, n - 1)]
        T = (nxt - prv).normalized()
        side = T.cross(nrms[i]).normalized() * (width * 0.5)
        L.append(bm.verts.new(pts[i] + side))
        R.append(bm.verts.new(pts[i] - side))
    m = n if closed else n - 1
    for i in range(m):
        j = (i + 1) % n
        bm.faces.new((L[i], R[i], R[j], L[j]))


def box(bm, center, size):
    M = Matrix.Translation(center) @ Matrix.Diagonal(Vector(size)).to_4x4()
    bmesh.ops.create_cube(bm, size=1.0, matrix=M)


def prism_band(bm, zb, zt, off, thick, n):
    """Закрытая лента прямоугольного сечения вокруг талии."""
    rings = []
    for t in ang(n):
        pt, _ = ell_pt(zt, t, off + thick)
        pb, _ = ell_pt(zb, t, off + thick)
        ib, _ = ell_pt(zb, t, off)
        it, _ = ell_pt(zt, t, off)
        rings.append([pt, pb, ib, it])
    V = [[bm.verts.new(p) for p in r] for r in rings]
    for i in range(n):
        j = (i + 1) % n
        for k in range(4):
            kk = (k + 1) % 4
            bm.faces.new((V[i][k], V[j][k], V[j][kk], V[i][kk]))


def runs_skip(samples, skip):
    out, cur = [], []
    for item, sk in zip(samples, skip):
        if sk:
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(item)
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------- геометрия
def build_shirt():
    bm = bmesh.new()
    tube(bm, torso_rings(SHIRT_ZS, 32, OFF_SHIRT))
    # V-вырез спереди: открывает грудь над расстёгнутым камзолом
    drop_faces(bm, lambda c: c.z > 1.20 and c.y < garment_params(c.z, 0.0)[1]
               and abs(c.x) < 0.5 * (c.z - 1.20))
    for s in (+1, -1):
        rings, _ = sleeve_rings(s, OFF_SHIRT)
        tube(bm, rings)
    return bm


def lapel(bm, s):
    zs = np.linspace(1.36, 1.02, 6)
    I, O = [], []
    for z in zs:
        z = float(z)
        pi, ni = front_pt(s * w_open(z), z, OFF_COAT)
        I.append(pi + ni * 0.008)
        xo = s * (0.14 - 0.05 * (1.36 - z) / 0.34)
        po, no = front_pt(xo, z, OFF_COAT)
        O.append(po + no * 0.008)
    VI = [bm.verts.new(p) for p in I]
    VO = [bm.verts.new(p) for p in O]
    for k in range(len(zs) - 1):
        bm.faces.new((VI[k], VO[k], VO[k + 1], VI[k + 1]))


def build_coat():
    bm = bmesh.new()
    tube(bm, torso_rings(COAT_ZS, 48, OFF_COAT, flare=0.035))
    # передний разлёт на груди (V от шеи до талии)
    drop_faces(bm, lambda c: c.z >= 1.0 and c.y < garment_params(c.z, 0.0)[1]
               and abs(c.x) < w_open(c.z))
    # разрезы юбки спереди и сзади
    drop_faces(bm, lambda c: c.z < 0.86 and abs(c.x) < 0.045)
    for s in (+1, -1):
        rings, _ = sleeve_rings(s, OFF_COAT)
        tube(bm, rings)
        lapel(bm, s)
    return bm


def build_trim():
    bm = bmesh.new()
    zs = np.linspace(1.36, 1.02, 14)
    for s in (+1, -1):
        # кант по передним краям разлёта
        P, N = [], []
        for z in zs:
            p, n = front_pt(s * w_open(float(z)), float(z), OFF_COAT)
            P.append(p + n * 0.004)
            N.append(n)
        ribbon(bm, P, N, 0.009)
        # кант по внешнему краю лацкана
        P, N = [], []
        for z in np.linspace(1.36, 1.02, 8):
            z = float(z)
            p, n = front_pt(s * (0.14 - 0.05 * (1.36 - z) / 0.34), z, OFF_COAT)
            P.append(p + n * 0.012)
            N.append(n)
        ribbon(bm, P, N, 0.009)
        # манжета: кольцо у запястья
        rings, frames = sleeve_rings(s, OFF_COAT)
        cuff = rings[-1]
        cN = [((f[1] * math.cos(t) + f[2] * math.sin(t))).normalized() for f, t in
              zip([frames[-1]] * len(ang(12)), ang(12))]
        cP = [p + n * 0.004 for p, n in zip(cuff, cN)]
        ribbon(bm, cP, cN, 0.012, closed=True)
    # воротник: верхнее кольцо камзола
    P, N = [], []
    for t in ang(64):
        p, n = ell_pt(1.44, t, OFF_COAT + 0.004)
        P.append(p)
        N.append(n)
    ribbon(bm, P, N, 0.012, closed=True)
    # подол: кольцо у середины бедра, без участков разреза
    ts = ang(96)
    skip = [abs(garment_params(0.66, OFF_COAT, 0.035)[0] * math.cos(t)) < 0.045 for t in ts]
    for run in runs_skip(list(zip(ts, range(len(ts)))), skip):
        P, N = [], []
        for t, _ in run:
            p, n = ell_pt(0.66, t, OFF_COAT, flare=0.035)
            P.append(p + n * 0.004)
            N.append(n)
        if len(P) > 1:
            ribbon(bm, P, N, 0.012)
    return bm


def build_buttons():
    bm = bmesh.new()
    for s in (+1, -1):
        for z in (1.02, 1.14, 1.26):
            x = s * (w_open(z) + 0.014)
            p, n = front_pt(x, z, OFF_COAT)
            box(bm, p + n * 0.006, (0.014, 0.008, 0.014))
    return bm


def build_belt():
    bm = bmesh.new()
    prism_band(bm, 0.97, 1.03, OFF_BELT, 0.012, 24)
    return bm


def build_buckle():
    bm = bmesh.new()
    p, n = front_pt(0.0, 1.00, OFF_BELT + 0.012)
    box(bm, p + n * 0.008, (0.05, 0.010, 0.036))
    return bm


def build_sash():
    bm = bmesh.new()
    # узел на левом боку спереди
    kc, kn = front_pt(0.07, 1.09, OFF_SASH + 0.004)
    kc = kc + kn * 0.014
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0,
                               matrix=Matrix.Translation(kc) @ Matrix.Diagonal(Vector((0.032, 0.02, 0.034))).to_4x4())
    prism_band(bm, 1.045, 1.115, OFF_SASH, 0.008, 24)
    # два свисающих хвоста
    for (x0, x1, z1), w in (((0.07, 0.12, 0.80), 0.030), ((0.06, 0.09, 0.84), 0.026)):
        P, N = [], []
        for k in range(6):
            t = k / 5.0
            x = x0 + (x1 - x0) * t
            z = 1.07 - (1.07 - z1) * t
            p, n = front_pt(x, z, OFF_SASH + 0.004)
            P.append(p + n * 0.006)
            N.append(n)
        ribbon(bm, P, N, w)
    return bm


# ---------------------------------------------------------------- прижим и вес
def clearance_fit(bm, target, min_clear):
    """Вершины внутри тела или ближе min_clear выносятся на target по нормали тела."""
    moved = 0
    for v in bm.verts:
        loc, nrm, _, _ = BVH_BODY.find_nearest(v.co)
        if (v.co - loc).dot(nrm) < min_clear:
            v.co = loc + nrm * target
            moved += 1
    return moved


def make_object(name, bm, mat_name):
    me = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    me.validate()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    mat = bpy.data.materials.get(mat_name)
    me.materials.append(mat)
    return ob


def make_materials():
    specs = {
        "Coat": ((0.42, 0.03, 0.06, 1), 0.0, 0.55),
        "Shirt": ((0.92, 0.91, 0.88, 1), 0.0, 0.7),
        "Belt": ((0.30, 0.17, 0.09, 1), 0.0, 0.5),
        "Sash": ((0.45, 0.04, 0.05, 1), 0.0, 0.6),
        "Gold": ((1.0, 0.72, 0.28, 1), 1.0, 0.35),
    }
    for name, (col, metal, rough) in specs.items():
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        bsdf = m.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = col
        bsdf.inputs["Metallic"].default_value = metal
        bsdf.inputs["Roughness"].default_value = rough


def transfer_weights(ob):
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
    # нормализация: сумма весов каждой вершины = 1
    me = ob.data
    gidx = {g.index: g for g in ob.vertex_groups}
    for v in me.vertices:
        ws = [(g.group, g.weight) for g in v.groups if g.group in gidx and g.weight > 0.0]
        tot = sum(w for _, w in ws)
        if tot <= 0.0:
            ws = [(ob.vertex_groups["Chest"].index, 1.0)]
            tot = 1.0
        for gi, w in ws:
            ob.vertex_groups[gi].add([v.index], w / tot, "REPLACE")
    ob.modifiers.new("Скелет", "ARMATURE").object = arm


# ---------------------------------------------------------------- сборка
make_materials()
log("строю Shirt")
shirt_bm = build_shirt()
log("  прижим Shirt:", clearance_fit(shirt_bm, OFF_SHIRT, 0.010))
log("строю Coat")
coat_bm = build_coat()
log("  прижим Coat:", clearance_fit(coat_bm, OFF_COAT, 0.022))
log("строю Coat_Trim, Buttons, Belt, Buckle, Sash")
trim_bm = build_trim()
buttons_bm = build_buttons()
belt_bm = build_belt()
log("  прижим Belt:", clearance_fit(belt_bm, OFF_BELT, 0.030))
buckle_bm = build_buckle()
sash_bm = build_sash()

objs = {}
objs["Shirt"] = make_object("Shirt", shirt_bm, "Shirt")
objs["Coat"] = make_object("Coat", coat_bm, "Coat")
objs["Coat_Trim"] = make_object("Coat_Trim", trim_bm, "Gold")
objs["Buttons"] = make_object("Buttons", buttons_bm, "Gold")
objs["Belt"] = make_object("Belt", belt_bm, "Belt")
objs["Buckle"] = make_object("Buckle", buckle_bm, "Gold")
objs["Sash"] = make_object("Sash", sash_bm, "Sash")

for name, ob in objs.items():
    transfer_weights(ob)

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
    worst = {g.name: [0.0, "", 0.0, ""] for g in garments}
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
                depth, gap = 0.0, 0.0
                for i, v in enumerate(gv):
                    loc, nrm, _, dist = bvh.find_nearest(v)
                    if (v - loc).dot(nrm) < 0.0:
                        depth = max(depth, dist)
                    gap = max(gap, dist - base[g.name][i])
                w = worst[g.name]
                if depth > w[0]:
                    w[0], w[1] = depth, case
                if gap > w[2]:
                    w[2], w[3] = gap, case
            pb.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    for name, (depth, dcase, gap, gcase) in worst.items():
        log(f"  {name}: макс. проникновение {depth * 100:.1f} см ({dcase or '-'}), "
            f"макс. отрыв +{gap * 100:.1f} см ({gcase or '-'})")


deform_test()


# ---------------------------------------------------------------- превью
def render_views():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False
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
