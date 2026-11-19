"""Одежда ног и рук героини, v2: Pants, Boots, Gloves, Gloves_Trim.

Запуск (headless):
  PYTHONPATH=<scratchpad>/pylib blender -b -P tools/blender/clothes_legs_v2.py

Что изменилось относительно v1:
  * Причина «рваных» краёв v1: glTF-импорт расщепляет вершины по UV/материалам,
    поэтому смещение по нормалям разрывало сетку. Теперь тело сначала сваривается.
  * Штаны: срез тела по плоскостям (Bisect), сглаженная промежность, талия ровная.
  * Ботфорты: голенище и манжета строятся кольцами по сечениям тела (ray cast),
    стопа - отдельный лофт (без пальцев), подошва и каблук - материал Leather.
  * Краги: раструб-кольца по предплечью + золотая манжета, кисть - отдельный
    сваренный меш на 4 мм снаружи тела.
"""
import math
import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

REPO = "/home/user/game"
BODY_GLB = REPO + "/models/adapted_body.glb"
OUT_BLEND = REPO + "/source_art/clothes_legs.blend"
PREV_PATTERN = REPO + "/docs/models/clothes_legs_%s.png"

TRI_TOTAL_LIMIT = 6000
POSE_ANGLE = 0.35
PEN_LIMIT = 0.02
GAP_LIMIT = 0.05

MATS = {
    "Pants": ((0.08, 0.06, 0.07), 0.0, 0.70),
    "Boots": ((0.16, 0.09, 0.05), 0.0, 0.55),
    "Boots_Cuff": ((0.28, 0.16, 0.08), 0.0, 0.55),
    "Leather": ((0.10, 0.06, 0.035), 0.0, 0.55),
    "Gloves": ((0.075, 0.035, 0.018), 0.0, 0.45),
    "Gold": ((1.00, 0.76, 0.32), 1.0, 0.35),
}

PANTS_Z = (0.42, 1.06)      # низ скрыт под голенищем, верх чуть выше бёдер
SHAFT_Z = (0.13, 0.565)     # голенище
CUFF_Z = (0.455, 0.585)     # отворот
SHAFT_OFF = 0.018
SOLE_H = 0.015
HEEL_H = 0.035
GAUNT_T = (0.225, 0.54)
TRIM_T = (0.20, 0.245)
HAND_T = 0.47               # кисть как отдельный меш начинается отсюда
HAND_OFF = 0.0060
HAND_TRIS = 430
PANTS_TRIS = 1450

BODY = None
ARM = None
BVH = None
REST_D = {}


def log(*a):
    print("[legs]", *a)


# ---------------- материалы / утилиты ----------------
def make_material(name, rgb, metal, rough):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    m.diffuse_color = (*rgb, 1.0)
    return m


def apply_modifier(ob, mtype, **props):
    m = ob.modifiers.new("tmp_" + mtype.lower(), mtype)
    for k, v in props.items():
        setattr(m, k, v)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.modifier_apply(modifier=m.name)


def tri_count(ob):
    return sum(len(p.vertices) - 2 for p in ob.data.polygons)


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def new_object(name, bm, mats, smooth=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    for mn in mats:
        me.materials.append(bpy.data.materials[mn])
    for p in me.polygons:
        p.use_smooth = smooth
    for vg in BODY.vertex_groups:
        ob.vertex_groups.new(name=vg.name)
    return ob


def transfer_weights(ob):
    m = ob.modifiers.new("weights_from_body", "DATA_TRANSFER")
    m.object = BODY
    m.use_object_transform = False
    m.use_vert_data = True
    m.data_types_verts = {"VGROUP_WEIGHTS"}
    m.vert_mapping = "POLYINTERP_NEAREST"
    m.layers_vgroup_select_src = "ALL"
    m.layers_vgroup_select_dst = "NAME"
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.modifier_apply(modifier=m.name)


def drop_empty_groups(ob):
    used = set()
    for v in ob.data.vertices:
        for g in v.groups:
            if g.weight > 1e-3:
                used.add(g.group)
    for name in [vg.name for vg in ob.vertex_groups if vg.index not in used]:
        ob.vertex_groups.remove(ob.vertex_groups[name])


def finish(ob):
    transfer_weights(ob)
    drop_empty_groups(ob)


def join_objects(objs, name):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.name = name
    ob.data.name = name
    return ob


# ---------------- рука ----------------
def arm_axis(side):
    a = ARM.data.bones["LowerArm." + side].head_local.copy()
    b = ARM.data.bones["Hand." + side].tail_local.copy()
    d = b - a
    return a, d / d.length, d.length


def arm_t(c):
    side = "L" if c.x > 0 else "R"
    a, u, ln = arm_axis(side)
    w = c - a
    p = w.dot(u)
    return p / ln, (w - u * p).length


# ---------------- тело: сварка и срезы ----------------
def welded_body_bm():
    bm = bmesh.new()
    bm.from_mesh(BODY.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=2e-4)
    return bm


def bisect(bm, co, no, inner=False, outer=False):
    geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
    bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-5, plane_co=co, plane_no=no,
                           clear_inner=inner, clear_outer=outer)


def clean_loose(bm):
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")


def body_piece(prefilter, cuts, offset, smooth_fn=None):
    bm = welded_body_bm()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if not prefilter(f.calc_center_median())],
                     context="FACES_ONLY")
    clean_loose(bm)
    cuts(bm)
    clean_loose(bm)
    bm.normal_update()
    for v in bm.verts:
        v.co = v.co + v.normal * offset(v.co)
    if smooth_fn:
        smooth_fn(bm)
    bm.normal_update()
    return bm


def push_out(ob, clear):
    """Вершины не ближе clear к поверхности тела (после сглаживания/децимации)."""
    for v in ob.data.vertices:
        loc, nrm, _, d = BVH.find_nearest(v.co)
        if loc is None:
            continue
        sg = (v.co - loc).dot(nrm)
        if sg < clear:
            v.co = v.co + nrm * (clear - sg)


def pants_bm():
    def pre(c):
        return PANTS_Z[0] - 0.1 <= c.z <= PANTS_Z[1] + 0.1 and abs(c.x) < 0.25

    def cuts(bm):
        bisect(bm, Vector((0, 0, PANTS_Z[0])), Vector((0, 0, 1)), inner=True)
        bisect(bm, Vector((0, 0, PANTS_Z[1])), Vector((0, 0, 1)), outer=True)

    def off(c):
        crotch = 0.0
        if 0.70 <= c.z <= 0.98 and abs(c.x) < 0.14:
            crotch = 0.010 * (1.0 - abs(c.x) / 0.14)
        return 0.011 + crotch

    def smooth(bm):
        # гладкий шов в промежности: лапласово сглаживание только там
        vs = [v for v in bm.verts if abs(v.co.x) < 0.07 and 0.70 < v.co.z < 0.92 and v.co.y < 0.0]
        for _ in range(4):
            bmesh.ops.smooth_vert(bm, verts=vs, factor=0.5, use_axis_x=True,
                                  use_axis_y=True, use_axis_z=True)
        for v in vs:   # после сглаживания не уходим внутрь тела
            loc, nrm, _, d = BVH.find_nearest(v.co)
            sg = (v.co - loc).dot(nrm)
            if sg < 0.008:
                v.co = v.co + nrm * (0.008 - sg)
    return body_piece(pre, cuts, off, smooth)


def hand_bm(side):
    def pre(c):
        if (c.x > 0) != (side == "L"):
            return False
        t, perp = arm_t(c)
        return t > HAND_T - 0.1 and perp < 0.14

    def cuts(bm):
        a, u, ln = arm_axis(side)
        bisect(bm, a + u * (ln * HAND_T), u, inner=True)
    def off(c):
        t, _ = arm_t(c)
        return HAND_OFF + 0.004 * smoothstep((t - 0.95) / 0.12)   # ногти не торчат
    return body_piece(pre, cuts, off)


# ---------------- лофты по кольцам ----------------
def loft(bm, rings, closed_start=False, closed_end=False):
    """rings: список списков вершин-координат (одинаковая длина n).
    Квады между кольцами; нормали наружу при e1 x e2 = направление обхода."""
    vr = [[bm.verts.new(p) for p in ring] for ring in rings]
    n = len(rings[0])
    for i in range(len(vr) - 1):
        for j in range(n):
            k = (j + 1) % n
            try:
                bm.faces.new((vr[i][j], vr[i][k], vr[i + 1][k], vr[i + 1][j]))
            except ValueError:
                pass
    return vr


def ray_radii(center, e1, e2, n, rmax=0.2):
    """Радиусы поверхности тела вокруг center (первое попадание изнутри)."""
    rs = []
    for j in range(n):
        a = 2 * math.pi * j / n
        d = e1 * math.cos(a) + e2 * math.sin(a)
        hit = BVH.ray_cast(center, d, rmax)
        rs.append(hit[3] if hit[0] is not None else None)
    good = [r for r in rs if r is not None]
    mean = sum(good) / len(good) if good else 0.05
    rs = [mean if r is None else r for r in rs]
    return rs


def smooth_ring(rs, passes=1):
    n = len(rs)
    for _ in range(passes):
        rs = [(rs[j - 1] + 2 * rs[j] + rs[(j + 1) % n]) / 4 for j in range(n)]
    return rs


def section_ring(center, e1, e2, n, rmax=0.2):
    """Находит центр сечения (две итерации) и радиусы."""
    c = center.copy()
    for _ in range(2):
        pts = []
        for j in range(n):
            a = 2 * math.pi * j / n
            d = e1 * math.cos(a) + e2 * math.sin(a)
            h = BVH.ray_cast(c, d, rmax)
            if h[0] is not None:
                pts.append(h[0])
        if pts:
            c = sum(pts, Vector()) / len(pts)
    return c, ray_radii(c, e1, e2, n, rmax)


def leg_sections(side, zs, n=16):
    sx = 0.092 if side == "L" else -0.092
    out = []
    for z in zs:
        c, rs = section_ring(Vector((sx, 0.0, z)), Vector((1, 0, 0)), Vector((0, 1, 0)), n, 0.16)
        out.append((c, smooth_ring(rs, 2)))
    return out


def ring_pts(c, e1, e2, rs):
    n = len(rs)
    pts = []
    for j, r in enumerate(rs):
        a = 2 * math.pi * j / n
        pts.append(c + (e1 * math.cos(a) + e2 * math.sin(a)) * r)
    return pts


def foot_profile(side):
    """Огибающая стопы по Y: (y, x_center, top_z, half_width). Без пальцев."""
    sx = 0.092 if side == "L" else -0.092
    vs = [v.co for v in BODY.data.vertices if v.co.z < 0.22 and abs(v.co.x - sx) < 0.07]
    ymin = min(v.y for v in vs)
    ymax = max(v.y for v in vs)
    ys = []
    steps = 8
    prof = []
    for i in range(steps + 1):
        y = ymax - (ymax - ymin) * i / steps
        sl = [v for v in vs if abs(v.y - y) < (ymax - ymin) / steps]
        if not sl:
            sl = vs
        top = max(v.z for v in sl)
        w = max(abs(v.x - sx) for v in sl)
        prof.append([y, sx, top, w])
    # сглаживание (убирает пальцы)
    for _ in range(6):
        for k in range(1, len(prof) - 1):
            prof[k][2] = (prof[k - 1][2] + 2 * prof[k][2] + prof[k + 1][2]) / 4
            prof[k][3] = (prof[k - 1][3] + 2 * prof[k][3] + prof[k + 1][3]) / 4
    return prof, ymin, ymax


def superellipse_ring(c, hw, hh, n, p=2.2):
    pts = []
    for j in range(n):
        a = 2 * math.pi * j / n
        ca, sa = math.cos(a), math.sin(a)
        x = hw * math.copysign(abs(ca) ** (2 / p), ca)
        z = hh * math.copysign(abs(sa) ** (2 / p), sa)
        pts.append(Vector((c.x + x, c.y, c.z + z)))
    return pts


def foot_upper_bm(side):
    prof, ymin, ymax = foot_profile(side)
    ext = 0.035                       # удлинённый носок
    rings = []
    n = 10
    # профиль идёт от пятки (ymax) к носку (ymin) -> обход e1=x, e2=z даёт нормали наружу
    heel = [(ymax + 0.016, 0.55), (ymax + 0.008, 0.85)]
    seq = []
    for hy, k in heel:
        y0 = prof[0]
        seq.append((hy, y0[1], y0[2], y0[3], k))
    for y, x, top, w in prof:
        seq.append((y, x, top, w, 1.0))
    ylast = prof[-1]
    tip = ymin - ext
    for f in (0.4, 0.8, 1.0):
        y = ylast[0] + (tip - ylast[0]) * f
        k = math.sqrt(max(0.0, 1.0 - f ** 2.6))
        seq.append((y, ylast[1], ylast[2] * (1.0 - 0.12 * f), ylast[3], max(k, 0.05)))
    for y, x, top, w, k in seq:
        zt = top + 0.014
        zb = 0.004
        hh = max((zt - zb) / 2 * k, 0.004)
        hw = (w + 0.016) * k
        # у верха (подъёма) стопа шире только внизу: низ плотнее
        c = Vector((x, y, (zt + zb) / 2 if k > 0.99 else zb + hh))
        rings.append(superellipse_ring(c, max(hw, 0.004), hh, n, 3.6))
    bm = bmesh.new()
    vr = loft(bm, rings)
    # закрыть носок и пятку
    for ring in (vr[0], vr[-1]):
        try:
            bm.faces.new(ring if ring is vr[-1] else list(reversed(ring)))
        except ValueError:
            pass
    return bm, prof, ymax, tip


def sole_bm(side, prof, ymax, tip):
    sx = prof[0][1]
    n = 8
    bm = bmesh.new()
    # подошва: выступает на 6 мм за контур
    rings = []
    seq = [(ymax + 0.022, 0.5, prof[0][3])]
    for y, x, top, w in prof:
        seq.append((y, 1.0, w))
    seq.append((prof[-1][0] + (tip - prof[-1][0]) * 0.5, 1.0, prof[-1][3]))
    seq.append((tip - 0.006, 0.85, prof[-1][3] * 0.9))
    seq.append((tip - 0.012, 0.35, prof[-1][3] * 0.6))
    for y, k, w in seq:
        hw = (w + 0.022) * k
        rings.append(superellipse_ring(Vector((sx, y, (SOLE_H - 0.003) / 2)), max(hw, 0.004),
                                       (SOLE_H + 0.003) / 2, n, 3.0))
    vr = loft(bm, rings)
    for ring in (vr[0], vr[-1]):
        try:
            bm.faces.new(ring if ring is vr[-1] else list(reversed(ring)))
        except ValueError:
            pass
    # каблук: блок под пяткой
    hrings = []
    for y, k in ((ymax + 0.030, 0.7), (ymax + 0.018, 1.0), (ymax - 0.03, 1.0), (ymax - 0.055, 0.9)):
        hrings.append(superellipse_ring(Vector((sx, y, (HEEL_H - 0.003) / 2)), 0.040 * k, (HEEL_H + 0.003) / 2, n, 3.0))
    bm2 = bm
    vr2 = loft(bm2, hrings)
    for ring in (vr2[0], vr2[-1]):
        try:
            bm2.faces.new(ring if ring is vr2[-1] else list(reversed(ring)))
        except ValueError:
            pass
    return bm


def shaft_bm(side):
    zs = [SHAFT_Z[0] + (SHAFT_Z[1] - SHAFT_Z[0]) * i / 7 for i in range(8)]
    secs = leg_sections(side, zs)
    e1, e2 = Vector((1, 0, 0)), Vector((0, 1, 0))
    rings = []
    for (c, rs), z in zip(secs, zs):
        # у щиколотки голенище толще на 1.2 см (чтобы скрыть переход к стопе), выше 1.8
        off = SHAFT_OFF - 0.006 * (1 - smoothstep((z - SHAFT_Z[0]) / 0.12))
        rings.append(ring_pts(c, e1, e2, [r + off for r in rs]))
    ext = []
    for zz in (0.07,):   # продолжение вниз внутри стопы: ровное пересечение
        ext.append([Vector((p.x, p.y, zz)) for p in rings[0]])
    rings = ext + rings
    bm = bmesh.new()
    loft(bm, rings)
    return bm, secs, zs


def cuff_bm(side):
    zs = [CUFF_Z[0] + (CUFF_Z[1] - CUFF_Z[0]) * i / 3 for i in range(4)]
    secs = leg_sections(side, zs)
    e1, e2 = Vector((1, 0, 0)), Vector((0, 1, 0))
    rings = []
    for (c, rs), z in zip(secs, zs):
        s = (z - CUFF_Z[0]) / (CUFF_Z[1] - CUFF_Z[0])
        flare = SHAFT_OFF + 0.003 + 0.015 * smoothstep(s * 1.1)
        rings.append(ring_pts(c, e1, e2, [r + flare for r in rs]))
    bm = bmesh.new()
    loft(bm, rings)
    return bm


def make_boots_v2():
    parts = []
    slot = {"Boots": 0, "Boots_Cuff": 1, "Leather": 2}
    for side in ("L", "R"):
        built = []
        sh, _, _ = shaft_bm(side)
        built.append((sh, "Boots", False))
        built.append((cuff_bm(side), "Boots_Cuff", True))
        up, prof, ymax, tip = foot_upper_bm(side)
        built.append((up, "Boots", False))
        built.append((sole_bm(side, prof, ymax, tip), "Leather", False))
        for bm, mat, solid in built:
            o = new_object("bp_" + mat + side, bm, ["Boots", "Boots_Cuff", "Leather"])
            for p in o.data.polygons:
                p.material_index = slot[mat]
            if solid:
                apply_modifier(o, "SOLIDIFY", thickness=0.004, offset=-1.0)
            parts.append(o)
    ob = join_objects(parts, "Boots")
    finish(ob)
    return ob


def make_pants():
    ob = new_object("Pants", pants_bm(), ["Pants"])
    push_out(ob, 0.008)
    finish(ob)
    return ob


def gauntlet_bm(side, t0, t1, extra_fn, n=16, steps=4):
    a, u, ln = arm_axis(side)
    # базис: e1 x e2 = u
    ref = Vector((0, 0, 1)) if abs(u.z) < 0.9 else Vector((1, 0, 0))
    e1 = (ref - u * ref.dot(u)).normalized()
    e2 = u.cross(e1)
    e2 = e2.normalized()
    # u = e1 x e2 ?  e1 x e2 = e1 x (u x e1) = u(e1.e1) - e1(e1.u) = u  -> верно
    rings = []
    for i in range(steps + 1):
        t = t0 + (t1 - t0) * i / steps
        c0 = a + u * (ln * t)
        c, rs = section_ring(c0, e1, e2, n, 0.16)
        rs = smooth_ring(rs, 1)
        ex = extra_fn(t)
        rings.append(ring_pts(c, e1, e2, [r + ex for r in rs]))
    bm = bmesh.new()
    loft(bm, rings)
    return bm


def make_gloves():
    parts = []
    for side in ("L", "R"):
        def ex(t):
            s = smoothstep((t - GAUNT_T[0]) / 0.17)
            return 0.007 + 0.022 * (1 - s)
        g = gauntlet_bm(side, GAUNT_T[0], GAUNT_T[1], ex)
        o = new_object("gg_" + side, g, ["Gloves"])
        apply_modifier(o, "SOLIDIFY", thickness=0.003, offset=-1.0)
        parts.append(o)
        h = new_object("gh_" + side, hand_bm(side), ["Gloves"])
        n0 = tri_count(h)
        apply_modifier(h, "DECIMATE", decimate_type="COLLAPSE", ratio=min(1.0, HAND_TRIS / n0))
        push_out(h, 0.0045)
        parts.append(h)
    ob = join_objects(parts, "Gloves")
    finish(ob)
    return ob


def make_trim():
    parts = []
    for side in ("L", "R"):
        def ex(t):
            s = smoothstep((t - GAUNT_T[0]) / 0.17)
            return 0.007 + 0.022 * (1 - s) + 0.006
        g = gauntlet_bm(side, TRIM_T[0], TRIM_T[1], ex, steps=2)
        o = new_object("gt_" + side, g, ["Gold"])
        apply_modifier(o, "SOLIDIFY", thickness=0.004, offset=-1.0)
        parts.append(o)
    ob = join_objects(parts, "Gloves_Trim")
    finish(ob)
    return ob


# ---------------- тест деформации ----------------
def pose_test(garments):
    pbs = list(ARM.pose.bones)
    for pb in pbs:
        pb.rotation_mode = "XYZ"
    for g in garments:
        m = g.modifiers.new("arm_test", "ARMATURE")
        m.object = ARM
    worst = {g.name: {"pen": (0.0, ""), "gap": (0.0, "")} for g in garments}
    poses = [("rest", None, 0, 0.0)]
    for pb in pbs:
        for ax in range(3):
            for s in (1.0, -1.0):
                poses.append((pb.name, pb, ax, s * POSE_ANGLE))
    for label, pb, ax, ang in poses:
        for o in pbs:
            o.rotation_euler = (0.0, 0.0, 0.0)
        if pb is not None:
            e = [0.0, 0.0, 0.0]
            e[ax] = ang
            pb.rotation_euler = e
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        be = BODY.evaluated_get(dg)
        md = be.to_mesh()
        bvh = BVHTree.FromPolygons([v.co.copy() for v in md.vertices],
                                   [tuple(p.vertices) for p in md.polygons])
        be.to_mesh_clear()
        for g in garments:
            gev = g.evaluated_get(dg)
            gme = gev.to_mesh()
            pen, gap = 0.0, 0.0
            rest = REST_D.setdefault(g.name, {})
            for v in gme.vertices:
                loc, nrm, _, dist = bvh.find_nearest(v.co)
                if loc is None:
                    continue
                sg = (v.co - loc).dot(nrm)
                if sg < 0.0:
                    pen = max(pen, -sg)
                if pb is None:
                    rest[v.index] = dist
                else:
                    gap = max(gap, dist - rest.get(v.index, 0.0))   # отрыв относительно покоя
            gev.to_mesh_clear()
            tag = label if pb is None else "%s %s%+.2f" % (label, "XYZ"[ax], ang)
            if pen > worst[g.name]["pen"][0]:
                worst[g.name]["pen"] = (pen, tag)
            if gap > worst[g.name]["gap"][0]:
                worst[g.name]["gap"] = (gap, tag)
    for pb in pbs:
        pb.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    for g in garments:
        for m in list(g.modifiers):
            g.modifiers.remove(m)
    log("pose test: %d poses" % len(poses))
    ok = True
    for name, w in worst.items():
        pen, pt = w["pen"]
        gap, gt = w["gap"]
        flag = "OK" if (pen <= PEN_LIMIT and gap <= GAP_LIMIT) else "FAIL"
        ok = ok and flag == "OK"
        log("  %-12s max_pen=%.4f (%s)  max_gap=%.4f (%s)  %s" % (name, pen, pt, gap, gt, flag))
    return ok


# ---------------- превью ----------------
def add_camera(name, loc, target, lens=50):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    ob = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = Vector(loc)
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return ob


def render_previews():
    sc = bpy.context.scene
    sc.world = bpy.data.worlds.new("PreviewWorld")
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.42, 0.42, 0.46, 1.0)
    sun = bpy.data.lights.new("PreviewSun", "SUN")
    sun.energy = 3.0
    so = bpy.data.objects.new("PreviewSun", sun)
    so.rotation_euler = (0.9, 0.0, 0.6)
    sc.collection.objects.link(so)
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 16
    sc.cycles.use_denoising = False
    sc.render.image_settings.file_format = "PNG"
    views = {
        "front": ((0.0, -2.8, 0.82), (0.0, 0.0, 0.82), 50, (800, 1000)),
        "side": ((2.8, 0.0, 0.82), (0.0, 0.0, 0.82), 50, (800, 1000)),
        "back": ((0.0, 2.8, 0.82), (0.0, 0.0, 0.82), 50, (800, 1000)),
        "close_boots": ((-0.75, -0.95, 0.38), (0.0, -0.03, 0.26), 50, (900, 900)),
        "close_hands": ((1.15, -0.95, 1.25), (0.60, -0.08, 0.98), 70, (900, 900)),
    }
    for vn, (loc, tgt, lens, res) in views.items():
        cam = add_camera("cam_" + vn, loc, tgt, lens)
        sc.camera = cam
        sc.render.resolution_x, sc.render.resolution_y = res
        sc.render.filepath = PREV_PATTERN % vn
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam)


def save_garments_only(keep):
    for ob in list(bpy.data.objects):
        if ob.name not in keep:
            bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.orphans_purge(do_recursive=True)
    bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)
    log("saved", OUT_BLEND)


def main():
    global BODY, ARM, BVH
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=BODY_GLB)
    BODY = bpy.data.objects["Body"]
    ARM = bpy.data.objects["Armature"]
    BVH = BVHTree.FromPolygons([v.co.copy() for v in BODY.data.vertices],
                               [tuple(p.vertices) for p in BODY.data.polygons])
    for n, (rgb, mt, rg) in MATS.items():
        make_material(n, rgb, mt, rg)
    for m in bpy.data.materials:
        if m.name not in MATS:
            m.diffuse_color = (0.72, 0.68, 0.64, 1.0)

    garments = [make_pants(), make_boots_v2(), make_gloves(), make_trim()]
    total = 0
    for g in garments:
        t = tri_count(g)
        total += t
        log(g.name, "tris", t)
    log("TOTAL tris", total, "(limit %d)" % TRI_TOTAL_LIMIT)
    ok = pose_test(garments)
    log("pose test", "PASSED" if ok else "HAS FAILURES")
    render_previews()
    save_garments_only({g.name for g in garments})


main()
