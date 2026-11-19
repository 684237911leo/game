"""Одежда ног героини: Pants, Boots, Gloves, Gloves_Trim.

Запуск (headless):
  PYTHONPATH=<scratchpad>/pylib blender -b -P tools/blender/clothes_legs.py

Что делает:
  1. Импортирует models/adapted_body.glb.
  2. Вырезает куски тела по плоскостям (Bisect: ровные кромки), смещает их по
     нормалям на 1-2 см, переносит веса с тела (Data Transfer, Nearest Face
     Interpolated). Децимация только если не влезли в бюджет.
  3. Тест деформации: каждая кость на +-0.35 рад по X/Y/Z, считает проникновение
     в тело и отрыв одежды. Поза возвращается в исходную.
  4. Превью front/side/back на теле (Cycles CPU, 16 сэмплов) ->
     docs/models/clothes_legs_*.png.
  5. Сохраняет source_art/clothes_legs.blend с ТОЛЬКО объектами одежды.
"""
import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

REPO = "/home/user/game"
BODY_GLB = REPO + "/models/adapted_body.glb"
OUT_BLEND = REPO + "/source_art/clothes_legs.blend"
PREV_PATTERN = REPO + "/docs/models/clothes_legs_%s.png"

TRI_BUDGET = {"Pants": 1300, "Boots": 1000, "Gloves": 800, "Gloves_Trim": 160}
TRI_TOTAL_LIMIT = 3500
POSE_ANGLE = 0.35
PEN_LIMIT = 0.02   # м, допустимое проникновение
GAP_LIMIT = 0.05   # м, больше этого - одежда отвалилась

# name: (base color, metallic, roughness)
MATS = {
    "Pants": ((0.13, 0.10, 0.08), 0.0, 0.70),
    "Boots": ((0.30, 0.17, 0.08), 0.0, 0.50),
    "Leather": ((0.20, 0.12, 0.07), 0.0, 0.55),
    "Gloves": ((0.24, 0.15, 0.09), 0.0, 0.50),
    "Gold": ((1.00, 0.76, 0.32), 1.0, 0.35),
}

# Высоты срезов (м). Штаны: от талии (1.02) до голени, скрыты ботфортами с 0.40.
PANTS_Z = (0.40, 1.02)
BOOT_TOP_Z = 0.50          # колено
# Руки: t вдоль предплечья, 0 = локоть, 1 = конец кисти
GLOVE_T = (0.235, 1.10)    # низ краги (от манжета вверх не берём) .. кончики пальцев
TRIM_T = (0.20, 0.235)     # золотой манжет

BODY = None
ARM = None


def log(*args):
    print("[legs]", *args)


# ---------- рука: параметр t вдоль предплечья ----------
def arm_axis(side):
    a = ARM.data.bones["LowerArm." + side].head_local.copy()
    b = ARM.data.bones["Hand." + side].tail_local.copy()
    d = b - a
    length = d.length
    return a, d / length, length


def arm_t(c):
    side = "L" if c.x > 0 else "R"
    a, u, length = arm_axis(side)
    w = c - a
    proj = w.dot(u)
    perp = (w - u * proj).length
    return proj / length, perp


def arm_plane_point(side, t):
    a, u, length = arm_axis(side)
    return a + u * (length * t), u


# ---------- фильтры по центроидам (грубый отбор, точные кромки делает Bisect) ----------
def pre_pants(c):
    return PANTS_Z[0] - 0.1 <= c.z <= PANTS_Z[1] + 0.1 and abs(c.x) < 0.25


def pre_boots(c):
    return c.z <= BOOT_TOP_Z + 0.1 and abs(c.x) < 0.25


def pre_arm(side):
    def f(c):
        if (c.x > 0) != (side == "L"):
            return False
        t, perp = arm_t(c)
        return t > TRIM_T[0] - 0.1 and perp < 0.14
    return f


# ---------- смещение от тела (м) ----------
def off_pants(c):
    return 0.011


def off_boots(c):
    # голенище 1.8 см; у верхнего края отворот до +0.8 см
    return 0.018 + 0.008 * min(1.0, max(0.0, (c.z - 0.42) / 0.08))


def off_gloves(c):
    t, _ = arm_t(c)
    return 0.010 + 0.010 * min(1.0, max(0.0, (0.32 - t) / 0.12))


def off_trim(c):
    t, _ = arm_t(c)
    return 0.016 + 0.010 * min(1.0, max(0.0, (0.32 - t) / 0.12))


# ---------- срезы (Bisect) ----------
def _bisect(bm, co, no, clear_inner=False, clear_outer=False):
    geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
    bmesh.ops.bisect_plane(
        bm, geom=geom, dist=1e-5, plane_co=co, plane_no=no,
        use_snap_center=False, clear_inner=clear_inner, clear_outer=clear_outer)


def cut_z(z_lo, z_hi):
    def f(bm):
        if z_lo is not None:   # оставляем z >= z_lo
            _bisect(bm, Vector((0, 0, z_lo)), Vector((0, 0, 1)), clear_inner=True)
        if z_hi is not None:   # оставляем z <= z_hi
            _bisect(bm, Vector((0, 0, z_hi)), Vector((0, 0, 1)), clear_outer=True)
    return f


def cut_arm(side, t_lo, t_hi):
    def f(bm):
        co, u = arm_plane_point(side, t_lo)
        _bisect(bm, co, u, clear_inner=True)
        co, u = arm_plane_point(side, t_hi)
        _bisect(bm, co, u, clear_outer=True)
    return f


# ---------- утилиты ----------
def make_material(name, rgb, metal, rough):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Roughness"].default_value = rough
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


def build_part(name, prefilter, cut, offset, mats, sole=None, floor=False):
    """Кусок тела -> объект: префильтр, срез, смещение, пустые группы костей."""
    bm = bmesh.new()
    bm.from_mesh(BODY.data)
    dele = [f for f in bm.faces if not prefilter(f.calc_center_median())]
    bmesh.ops.delete(bm, geom=dele, context="FACES_ONLY")
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    cut(bm)
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")

    bm.normal_update()
    for v in bm.verts:
        d = offset(v.co)
        v.co = v.co + v.normal * d
        if floor and v.co.z < 0.0:
            v.co.z = 0.0  # подошва стоит на Z=0
    bm.normal_update()

    for f in bm.faces:
        f.material_index = 1 if (sole and sole(f.calc_center_median())) else 0

    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    for mname in mats:
        me.materials.append(bpy.data.materials[mname])
    for p in me.polygons:
        p.use_smooth = True
    for vg in BODY.vertex_groups:  # те же имена и порядок, что у тела
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
    # удаляем по именам: индексы групп сдвигаются после каждого remove
    empty = [vg.name for vg in ob.vertex_groups if vg.index not in used]
    for name in empty:
        ob.vertex_groups.remove(ob.vertex_groups[name])


def finish_part(ob):
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


def fit_budget(ob, budget):
    n = tri_count(ob)
    if n > budget:
        apply_modifier(ob, "DECIMATE", decimate_type="COLLAPSE", ratio=budget / n)
    log(ob.name, "tris", tri_count(ob), "(before budget %d: %d)" % (budget, n))


def make_pants():
    ob = build_part("Pants", pre_pants, cut_z(*PANTS_Z), off_pants, ["Pants"])
    finish_part(ob)
    fit_budget(ob, TRI_BUDGET["Pants"])
    return ob


def make_boots():
    ob = build_part("Boots", pre_boots, cut_z(None, BOOT_TOP_Z), off_boots,
                    ["Boots", "Leather"], sole=lambda c: c.z < 0.035, floor=True)
    finish_part(ob)
    fit_budget(ob, TRI_BUDGET["Boots"])
    return ob


def make_gloves_like(name, t_range, offset, mat):
    parts = []
    for side in ("L", "R"):
        p = build_part(name + "_" + side, pre_arm(side), cut_arm(side, *t_range),
                       offset, [mat])
        finish_part(p)
        parts.append(p)
    ob = join_objects(parts, name)
    return ob


def make_gloves():
    ob = make_gloves_like("Gloves", GLOVE_T, off_gloves, "Gloves")
    fit_budget(ob, TRI_BUDGET["Gloves"])
    return ob


def make_trim():
    ob = make_gloves_like("Gloves_Trim", TRIM_T, off_trim, "Gold")
    fit_budget(ob, TRI_BUDGET["Gloves_Trim"])
    return ob


# ---------- тест деформации ----------
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
        for other in pbs:
            other.rotation_euler = (0.0, 0.0, 0.0)
        if pb is not None:
            e = [0.0, 0.0, 0.0]
            e[ax] = ang
            pb.rotation_euler = e
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        body_eval = BODY.evaluated_get(dg)
        bmesh_data = body_eval.to_mesh()
        bvh = BVHTree.FromPolygons(
            [v.co.copy() for v in bmesh_data.vertices],
            [tuple(p.vertices) for p in bmesh_data.polygons])
        body_eval.to_mesh_clear()
        for g in garments:
            gev = g.evaluated_get(dg)
            gme = gev.to_mesh()
            pen, gap = 0.0, 0.0
            for v in gme.vertices:
                loc, nrm, _, dist = bvh.find_nearest(v.co)
                if loc is None:
                    continue
                signed = (v.co - loc).dot(nrm)
                if signed < 0.0:
                    pen = max(pen, -signed)
                gap = max(gap, dist)
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

    log("pose test: %d poses (rest + 22 bones x 3 axes x +-0.35 rad)" % len(poses))
    ok = True
    for name, w in worst.items():
        pen, ptag = w["pen"]
        gap, gtag = w["gap"]
        flag = "OK" if (pen <= PEN_LIMIT and gap <= GAP_LIMIT) else "FAIL"
        if flag == "FAIL":
            ok = False
        log("  %-12s max_pen=%.4f м (%s)  max_gap=%.4f м (%s)  %s"
            % (name, pen, ptag, gap, gtag, flag))
    return ok


def check_bounds(ob, mode):
    """Проверка срезов по уже собранной геометрии."""
    zs = [v.co.z for v in ob.data.vertices]
    ts = [arm_t(v.co)[0] for v in ob.data.vertices] if mode == "arm" else []
    if mode == "arm":
        log("  %-12s t range %.3f .. %.3f (ожидалось %.3f .. %.3f)"
            % (ob.name, min(ts), max(ts), TRIM_T[0] if ob.name == "Gloves_Trim" else GLOVE_T[0],
               TRIM_T[1] if ob.name == "Gloves_Trim" else GLOVE_T[1]))
    else:
        log("  %-12s z range %.3f .. %.3f" % (ob.name, min(zs), max(zs)))


# ---------- превью ----------
def add_camera(name, loc, target):
    cam = bpy.data.cameras.new(name)
    cam.lens = 50
    ob = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = Vector(loc)
    d = Vector(target) - Vector(loc)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return ob


def render_previews():
    sc = bpy.context.scene
    # Cycles CPU: Workbench в headless-режиме требует OpenGL/EGL, которого нет
    sc.world = bpy.data.worlds.new("PreviewWorld")
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.42, 0.42, 0.46, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    sun = bpy.data.lights.new("PreviewSun", "SUN")
    sun.energy = 3.0
    sun_ob = bpy.data.objects.new("PreviewSun", sun)
    sun_ob.rotation_euler = (0.9, 0.0, 0.6)
    bpy.context.scene.collection.objects.link(sun_ob)
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 16
    sc.cycles.use_denoising = False  # сборка Blender без OpenImageDenoiser
    sc.render.resolution_x = 800
    sc.render.resolution_y = 1000
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    views = {
        "front": ((0.0, -2.8, 0.82), (0.0, 0.0, 0.82)),   # лицо смотрит в -Y
        "side": ((2.8, 0.0, 0.82), (0.0, 0.0, 0.82)),
        "back": ((0.0, 2.8, 0.82), (0.0, 0.0, 0.82)),
    }
    for vname, (loc, tgt) in views.items():
        cam = add_camera("cam_" + vname, loc, tgt)
        sc.camera = cam
        sc.render.filepath = PREV_PATTERN % vname
        bpy.ops.render.render(write_still=True)
        log("preview", sc.render.filepath)
        bpy.data.objects.remove(cam)


# ---------- сохранение ----------
def save_garments_only(keep_names):
    for ob in list(bpy.data.objects):
        if ob.name not in keep_names:
            bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.orphans_purge(do_recursive=True)
    bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)
    log("saved", OUT_BLEND)


def main():
    global BODY, ARM
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=BODY_GLB)
    BODY = bpy.data.objects["Body"]
    ARM = bpy.data.objects["Armature"]

    for name, (rgb, metal, rough) in MATS.items():
        make_material(name, rgb, metal, rough)
    # тело в превью серое, чтобы одежда читалась
    for m in bpy.data.materials:
        if m.name not in MATS:
            m.diffuse_color = (0.72, 0.68, 0.64, 1.0)

    garments = [make_pants(), make_boots(), make_gloves(), make_trim()]
    log("bounds check:")
    check_bounds(garments[0], "z")
    check_bounds(garments[1], "z")
    check_bounds(garments[2], "arm")
    check_bounds(garments[3], "arm")
    total = sum(tri_count(g) for g in garments)
    log("TOTAL tris", total, "(limit %d)" % TRI_TOTAL_LIMIT)

    ok = pose_test(garments)
    log("pose test", "PASSED" if ok else "HAS FAILURES")

    render_previews()
    save_garments_only({g.name for g in garments})


main()
