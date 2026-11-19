"""Сборка героини в одежде: adapted_body.glb + clothes_{torso,legs,head}.blend -> models/hero_dressed.glb.

Запуск:
  PYTHONPATH=<scratchpad>/pylib blender -b -P tools/blender/assemble_dressed.py

Этапы: импорт тела (без Icosphere, поза сброшена в rest) -> append одежды и привязка Armature-модификатором
-> материалы (Gold) -> проверки -> экспорт GLB -> проверка JSON-чанка -> превью из переимпортированного GLB.
"""
import bpy
import json
import math
import os
import re
import struct
import subprocess
import sys

import numpy as np
from mathutils import kdtree

ROOT = "/home/user/game"
BODY_GLB = os.path.join(ROOT, "models/adapted_body.glb")
CLOTHES_BLENDS = [
    os.path.join(ROOT, "source_art/clothes_torso.blend"),
    os.path.join(ROOT, "source_art/clothes_legs.blend"),
    os.path.join(ROOT, "source_art/clothes_head.blend"),
]
OUT_GLB = os.path.join(ROOT, "models/hero_dressed_v2.glb")
DOCS = os.path.join(ROOT, "docs/models")
WORK = "/tmp/claude-0/-home-user-game/354ab7fc-40d7-5937-8c42-4d3716dfda7e/scratchpad/work"
RENDER_DIR = os.path.join(WORK, "renders")

ANIMS = ["Idle", "Walk", "Attack", "Death"]
LOOPING = {"Idle": True, "Walk": True, "Attack": False, "Death": False}
GOLD_BASE = (0.95, 0.75, 0.25, 1.0)
GOLD_METALLIC = 0.1
GOLD_ROUGHNESS = 0.45
OTHER_METALLIC_MAX = 0.2
TRI_BUDGET = 22000
# Расстояние от одежды до тела (м): для сравнения, не для правки.
STILL_SIZE = (480, 720)
STRIP_SIZE = (320, 480)
CAM_DIST_STILL = 3.9
CAM_DIST_STRIP = 3.9
CAM_Z = 0.92


def log(*args):
    print("[assemble]", *args, flush=True)


def norm(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def tris_of(obj):
    return sum(max(len(p.vertices) - 2, 0) for p in obj.data.polygons)


def mesh_objects():
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def reset_pose(arm):
    """Все кости в rest. Возвращает число костей, которые были не в rest."""
    reset = 0
    for pb in arm.pose.bones:
        if (tuple(pb.rotation_quaternion) != (1.0, 0.0, 0.0, 0.0)
                or any(abs(x) > 1e-7 for x in pb.location)
                or any(abs(x - 1.0) > 1e-7 for x in pb.scale)):
            reset += 1
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.location = (0.0, 0.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)
    return reset


# ---------------------------------------------------------------- 1. тело
def import_body():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=BODY_GLB)

    # лишний Icosphere (80 tris, r=1 м)
    for o in list(bpy.data.objects):
        if o.type == "MESH" and o.name.startswith("Icosphere"):
            name = o.name
            me = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
            log("removed", name)

    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    assert len(arms) == 1, f"ожидалась одна армматура, есть {len(arms)}"
    arm = arms[0]

    # actions не применяем: отвязываем активный action и NLA
    if arm.animation_data:
        arm.animation_data.action = None
        for t in list(arm.animation_data.nla_tracks):
            arm.animation_data.nla_tracks.remove(t)

    reset = reset_pose(arm)
    log("bones reset to rest:", len(arm.pose.bones), "(было не нулевых:", reset, ")")

    # имена actions: Blender добавил суффикс _Armature, в файле были Idle/Walk/Attack/Death
    for a in list(bpy.data.actions):
        base = next((n for n in ANIMS if a.name == n or a.name.startswith(n + "_")), None)
        if base is None:
            bpy.data.actions.remove(a)
            continue
        a.name = base
        a.use_fake_user = True
    log("actions:", sorted(a.name for a in bpy.data.actions))
    return arm


# ---------------------------------------------------------------- 2. одежда
def append_clothes(arm):
    bones = {b.name: b.name for b in arm.data.bones}
    bone_norm = {norm(n): n for n in bones}
    added = []
    for path in CLOTHES_BLENDS:
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.objects = list(src.objects)
        for o in dst.objects:
            if o is None:
                continue
            if o.type != "MESH":
                bpy.data.objects.remove(o, do_unlink=True)
                continue
            bpy.context.scene.collection.objects.link(o)
            added.append(o)
        log("appended from", os.path.basename(path))

    missing = []
    for o in added:
        for vg in o.vertex_groups:
            if vg.name in bones:
                continue
            target = bone_norm.get(norm(vg.name))
            if target:
                log(f"rename vg {o.name}: '{vg.name}' -> '{target}'")
                vg.name = target
            else:
                missing.append((o.name, vg.name))
        # Armature-модификатор, parent OBJECT, transform сохраняется (parent inverse = inv(arm))
        mod = o.modifiers.new("Armature", "ARMATURE")
        mod.object = arm
        mod.use_vertex_groups = True
        o.parent = arm
        o.parent_type = "OBJECT"
        o.matrix_parent_inverse = arm.matrix_world.inverted()
    if missing:
        log("ВНИМАНИЕ: vertex groups без кости:", missing)
    return added


# ---------------------------------------------------------------- 3. материалы
def fix_materials(clothes):
    golds = sorted([m for m in bpy.data.materials if re.fullmatch(r"Gold(\.\d+)?", m.name)],
                   key=lambda m: m.name)
    canonical = golds[0]
    gold_names = [g.name for g in golds]
    for o in clothes:
        for slot in o.material_slots:
            if slot.material in golds and slot.material != canonical:
                slot.material = canonical
    for g in golds:
        if g != canonical and g.users == 0:
            bpy.data.materials.remove(g)
    log("Gold variants merged:", gold_names, "->", canonical.name)

    def principled(mat):
        return next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")

    node = principled(canonical)
    node.inputs["Metallic"].default_value = GOLD_METALLIC
    node.inputs["Roughness"].default_value = GOLD_ROUGHNESS
    node.inputs["Base Color"].default_value = GOLD_BASE

    for o in clothes:
        for slot in o.material_slots:
            m = slot.material
            if m is None or m == canonical:
                continue
            n = principled(m)
            if n.inputs["Metallic"].default_value > OTHER_METALLIC_MAX:
                log(f"clamp metallic {m.name}: {n.inputs['Metallic'].default_value} -> {OTHER_METALLIC_MAX}")
                n.inputs["Metallic"].default_value = OTHER_METALLIC_MAX


def reweight_from_body(obj, body, k=3):
    """Перенос весов с тела по ближайшим вершинам (аналог Data Transfer Nearest, k=3).
    Нужен перчаткам: у левой стороны все вершины были привязаны к LowerArm.L, кисть не следовала за Hand.L."""
    body_w = []
    for v in body.data.vertices:
        body_w.append({body.vertex_groups[g.group].name: g.weight for g in v.groups if g.weight > 0})
    kd = kdtree.KDTree(len(body.data.vertices))
    for i, v in enumerate(body.data.vertices):
        kd.insert(body.matrix_world @ v.co, i)
    kd.balance()
    for vg in list(obj.vertex_groups):
        obj.vertex_groups.remove(vg)
    new_w = {}
    for v in obj.data.vertices:
        p = obj.matrix_world @ v.co
        acc = {}
        total = 0.0
        for _, idx, d in kd.find_n(p, k):
            wgt = 1.0 / (d + 1e-4)
            total += wgt
            for name, w in body_w[idx].items():
                acc[name] = acc.get(name, 0.0) + wgt * w
        new_w[v.index] = {n: w / total for n, w in acc.items() if w / total > 1e-4}
    names = sorted({n for d in new_w.values() for n in d})
    groups = {n: obj.vertex_groups.new(name=n) for n in names}
    for idx, d in new_w.items():
        s = sum(d.values())
        for n, w in d.items():
            groups[n].add([idx], w / s, "REPLACE")
    log(f"reweighted {obj.name}: groups {len(names)}, Hand.L verts",
        sum(1 for d in new_w.values() if d.get("Hand.L", 0) > 0.5),
        "Hand.R verts", sum(1 for d in new_w.values() if d.get("Hand.R", 0) > 0.5))


def fix_gloves(clothes):
    body = next(o for o in bpy.context.scene.objects if o.name == "Body")
    for o in clothes:
        if o.name in ("Gloves", "Gloves_Trim"):
            reweight_from_body(o, body)


# ---------------------------------------------------------------- 4. проверки
def checks(arm):
    bones = set(b.name for b in arm.data.bones)
    report = {"objects": [], "problems": []}
    total = 0
    body_tris = 0
    for o in mesh_objects():
        t = tris_of(o)
        total += t
        has_arm = any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)
        n_vg = len(o.vertex_groups)
        unweighted = 0
        for v in o.data.vertices:
            if not any(g.weight > 0 for g in v.groups):
                unweighted += 1
        bad_names = [vg.name for vg in o.vertex_groups if vg.name not in bones]
        if o.name in ("Body", "Hair", "Scalp", "Eyes", "Eyelashes"):
            body_tris += t
        report["objects"].append({"name": o.name, "tris": t, "armature_mod": has_arm,
                                  "vertex_groups": n_vg, "unweighted_verts": unweighted})
        if not has_arm:
            report["problems"].append(f"{o.name}: нет Armature-модификатора")
        if n_vg == 0:
            report["problems"].append(f"{o.name}: нет vertex groups")
        if unweighted:
            report["problems"].append(f"{o.name}: вершин без весов {unweighted}")
        if bad_names:
            report["problems"].append(f"{o.name}: группы не по костям {bad_names}")
        if o.parent != arm or o.parent_type != "OBJECT":
            report["problems"].append(f"{o.name}: parent не OBJECT к армматуре")
    report["total_tris"] = total
    report["body_tris"] = body_tris
    report["clothes_tris"] = total - body_tris
    report["budget_ok"] = total <= TRI_BUDGET
    if not report["budget_ok"]:
        report["problems"].append(f"бюджет: {total} > {TRI_BUDGET} (тело не трогали, см. отчёт)")
    if any(o.name == "Icosphere" for o in bpy.data.objects):
        report["problems"].append("Icosphere остался")
    log(json.dumps(report, indent=1, ensure_ascii=False))
    return report


def clothing_gap(arm, clothes):
    """Расстояние вершин одежды до ближайшей вершины тела (rest): проверка посадки."""
    body = [o for o in mesh_objects() if o.name == "Body"][0]
    size = len(body.data.vertices)
    kd = kdtree.KDTree(size)
    for i, v in enumerate(body.data.vertices):
        kd.insert(body.matrix_world @ v.co, i)
    kd.balance()
    for o in clothes:
        dists = []
        for v in o.data.vertices:
            _, _, d = kd.find(o.matrix_world @ v.co)
            dists.append(d)
        dists.sort()
        if dists:
            q = lambda p: dists[min(int(p * len(dists)), len(dists) - 1)]
            log(f"gap {o.name:12s} p5={q(0.05)*100:.2f}cm p50={q(0.5)*100:.2f}cm p95={q(0.95)*100:.2f}cm")


# ---------------------------------------------------------------- 5. экспорт
def export_glb(arm):
    keep = set(ANIMS)
    for a in list(bpy.data.actions):
        if a.name not in keep:
            bpy.data.actions.remove(a)
    arm.animation_data.action = None
    bpy.ops.export_scene.gltf(
        filepath=OUT_GLB,
        export_format="GLB",
        export_skins=True,
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_force_sampling=True,
        export_frame_range=False,
        export_yup=True,
        export_apply=False,
    )
    log("exported", OUT_GLB, os.path.getsize(OUT_GLB), "bytes")


def read_glb_json(path):
    b = open(path, "rb").read()
    assert b[:4] == b"glTF"
    length = struct.unpack("<I", b[12:16])[0]
    assert b[16:20] == b"JSON"
    return json.loads(b[20:20 + length].decode("utf-8")), len(b)


def verify_glb(path):
    d, size = read_glb_json(path)
    acc = d["accessors"]
    tris = 0
    for m in d["meshes"]:
        for p in m["primitives"]:
            if "indices" in p:
                tris += acc[p["indices"]]["count"] // 3
    info = {
        "size_bytes": size,
        "animations": [a.get("name") for a in d.get("animations", [])],
        "skins": [len(s["joints"]) for s in d.get("skins", [])],
        "meshes": [m.get("name") for m in d["meshes"]],
        "materials": [m.get("name") for m in d.get("materials", [])],
        "triangles": tris,
    }
    log("GLB JSON:", json.dumps(info, ensure_ascii=False))
    return info


# ---------------------------------------------------------------- 6. превью
def preview_setup():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False
    scene.view_settings.view_transform = "Standard"
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"

    world = bpy.data.worlds.new("PreviewWorld")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.5, 0.52, 0.56, 1.0)
    bg.inputs[1].default_value = 1.0
    scene.world = world

    for name, energy, rot in [("Key", 3.0, (math.radians(55), 0, math.radians(25))),
                              ("Fill", 1.2, (math.radians(70), 0, math.radians(-150)))]:
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = energy
        lo = bpy.data.objects.new(name, ld)
        lo.rotation_euler = rot
        scene.collection.objects.link(lo)

    pivot = bpy.data.objects.new("Pivot", None)
    pivot.location = (0.0, 0.0, CAM_Z)
    scene.collection.objects.link(pivot)
    cd = bpy.data.cameras.new("Cam")
    cd.lens = 50.0
    cd.sensor_width = 36.0
    cam = bpy.data.objects.new("Cam", cd)
    scene.collection.objects.link(cam)
    cam.parent = pivot
    cam.location = (0.0, -CAM_DIST_STILL, 0.0)
    cam.rotation_euler = (math.radians(90), 0.0, 0.0)
    scene.camera = cam
    return pivot, cam


def set_camera(pivot, cam, angle_deg, dist):
    cam.location = (0.0, -dist, 0.0)
    pivot.rotation_euler = (0.0, 0.0, math.radians(angle_deg))


def render_to(path, size):
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def preview(glb_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb_path)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    actions = {}
    for a in bpy.data.actions:
        base = next((n for n in ANIMS if a.name == n or a.name.startswith(n + "_")), None)
        if base:
            actions[base] = a
    assert set(actions) == set(ANIMS), f"в переимпорте нет анимаций: {sorted(actions)}"
    log("re-import actions:", {k: v.name for k, v in actions.items()})
    # импортёр применяет первый кадр активного action к позе: для стиллов сбрасываем в rest
    arm.animation_data_create()
    arm.animation_data.action = None
    log("re-import pose was non-rest on", reset_pose(arm), "bones")

    pivot, cam = preview_setup()
    os.makedirs(RENDER_DIR, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_set(0)
    arm.animation_data.action = None
    scene.frame_set(0)

    # стиллы: поза покоя (bind), 3 ракурса
    stills = [("dressed_front.png", 0), ("dressed_34.png", 35), ("dressed_back.png", 180)]
    for fname, ang in stills:
        set_camera(pivot, cam, ang, CAM_DIST_STILL)
        render_to(os.path.join(DOCS, fname), STILL_SIZE)
        log("still", fname)

    # крупные планы (камера Pivot не нужна: ставим вручную)
    for fname, loc, tgt, lens in [
        ("dressed_close_shoulder.png", (1.0, -0.9, 1.5), (0.33, 0.0, 1.35), 50),
        ("dressed_close_hands.png", (1.15, -0.95, 1.25), (0.60, -0.08, 0.98), 70),
        ("dressed_close_shoulder_R.png", (-1.0, -0.9, 1.5), (-0.33, 0.0, 1.35), 50),
        ("dressed_close_shoulder_back.png", (-0.9, 1.0, 1.5), (-0.33, 0.0, 1.35), 50),
        ("dressed_close_hands_R.png", (-1.15, -0.95, 1.25), (-0.58, -0.10, 0.96), 70),
    ]:
        cam.parent = None
        cam.location = loc
        cam.rotation_euler = (__import__("mathutils").Vector(tgt) - __import__("mathutils").Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        cam.data.lens = lens
        render_to(os.path.join(DOCS, fname), (800, 800))
        log("close", fname)
    cam.parent = pivot
    cam.data.lens = 50.0
    cam.rotation_euler = (math.radians(90), 0.0, 0.0)

    # полосы: 6 кадров, вид 3/4
    for anim in ANIMS:
        act = actions[anim]
        arm.animation_data.action = act
        n = int(round(act.frame_range[1]))
        if LOOPING[anim]:
            frames = [round(i * n / 6) for i in range(6)]
        else:
            frames = [round(i * n / 5) for i in range(6)]
        frame_paths = []
        set_camera(pivot, cam, 35, CAM_DIST_STRIP)
        for i, f in enumerate(frames):
            scene.frame_set(f)
            p = os.path.join(RENDER_DIR, f"{anim}_{i}.png")
            render_to(p, STRIP_SIZE)
            frame_paths.append(p)
        out = os.path.join(DOCS, f"dressed_{anim.lower()}.png")
        stitch(frame_paths, out)
        log("strip", anim, frames, "->", out)
    arm.animation_data.action = None


def stitch(paths, out):
    code = (
        "import sys\nfrom PIL import Image\n"
        "out=sys.argv[1]; ims=[Image.open(p).convert('RGB') for p in sys.argv[2:]]\n"
        "h=max(i.height for i in ims); w=sum(i.width for i in ims)\n"
        "c=Image.new('RGB',(w,h),(255,255,255)); x=0\n"
        "for i in ims: c.paste(i,(x,0)); x+=i.width\n"
        "c.save(out)\n"
    )
    subprocess.run(["python3", "-I", "-c", code, out] + paths, check=True)


# ---------------------------------------------------------------- main
def main():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(DOCS, exist_ok=True)
    arm = import_body()
    clothes = append_clothes(arm)
    fix_materials(clothes)
    fix_gloves(clothes)
    clothing_gap(arm, clothes)
    report = checks(arm)
    export_glb(arm)
    glb = verify_glb(OUT_GLB)
    with open(os.path.join(WORK, "report.json"), "w") as f:
        json.dump({"checks": report, "glb": glb}, f, indent=1, ensure_ascii=False)
    if "--no-preview" not in sys.argv:
        preview(OUT_GLB)
    log("done")


if __name__ == "__main__":
    main()
