"""Адаптация скачанной женской модели (CC3-подобный FBX, 101 кость) под наш 22-костный риг.

Запуск из корня репозитория:
  PYTHONPATH=<pylib с numpy> blender -b -P tools/blender/adapt_body.py

Переменные окружения (необязательные):
  ADAPT_FBX     путь к входному FBX (по умолчанию см. FBX_DEFAULT)
  ADAPT_WORK    рабочая папка для промежуточных текстур/рендеров
  ADAPT_STOP    остановиться после стадии: import | decimate | retarget | export
  ADAPT_NORENDER=1  не рендерить превью

Вход (только чтение): FBX-модель, source_art/hero_anim.blend (риг + 4 action).
Выход: models/adapted_body.glb, docs/models/adapted_*.png,
       docs/models/adapted_body_report.json.

Что делает:
  1. Импорт FBX в REST-позе. Удаляет Teeth, Tongue, TearLine, EyeOcclusion,
     top_cloth, bottome, ногти и роговицы глаз. Остаются: Body (кожа),
     Eyelashes, Eyes, Hair, Scalp.
  2. Децимация (Collapse) до бюджета треугольников.
  3. «Ретаргет геометрии»: исходная модель в A-позе 30 градусов, наш риг в A-позе
     40 градусов и другие пропорции. Каждой нашей кости сопоставлены суставы
     исходной модели; геометрия переводится в нашу rest-позу линейным скиннингом
     (поворот + растяжение вдоль сегмента вокруг сустава), веса сливаются по
     таблице соответствий в наши 22 кости.
  4. Привязка к нашему арматуру (Armature-модификатор + группы вершин с нашими
     именами), копирование 4 action (Idle, Walk, Attack, Death) из hero_anim.blend.
  5. Текстуры до 512 px, материалы пересобраны (glTF-совместимые).
  6. Экспорт GLB, проверка JSON-чанка, тест поворота костей, превью.

Системы координат: вперёд = -Y, вверх = +Z, левая сторона = +X (как у героя).
"""

import json
import math
import os
import struct
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Quaternion, Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if not os.path.isdir(os.path.join(ROOT, "models")):
    ROOT = "/home/user/game"
MODELS = os.path.join(ROOT, "models")
DOCS = os.path.join(ROOT, "docs", "models")

SCRATCH = "/tmp/claude-0/-home-user-game/354ab7fc-40d7-5937-8c42-4d3716dfda7e/scratchpad"
FBX_DEFAULT = os.path.join(SCRATCH, "fbxout", "fbx clean.fbx")
FBX = os.environ.get("ADAPT_FBX", FBX_DEFAULT)
WORK = os.environ.get("ADAPT_WORK", os.path.join(SCRATCH, "work"))
STOP = os.environ.get("ADAPT_STOP", "")
NORENDER = os.environ.get("ADAPT_NORENDER", "") == "1"
FLOOR_FIX = os.environ.get("ADAPT_FLOORFIX", "1") == "1"
RIG_BLENDS = [os.path.join(ROOT, "source_art", "hero_anim.blend"),
              os.path.join(MODELS, "hero_anim.blend")]

OUT_GLB = os.path.join(MODELS, "adapted_body.glb")

# --- бюджет (треугольники) ---
TRI_BODY = 6000
TRI_HAIR = 3000
TRI_SCALP = 500
TRI_EYES = 300
TRI_LASH = 400
TEX_MAX = 512
EYE_TEX = 256

# --- косметика (исходные текстуры серые/белые, см. отчёт) ---
SKIN_TINT = (0.94, 0.79, 0.70)     # sRGB множитель тона кожи
HAIR_TINT = (0.46, 0.30, 0.20)     # sRGB цвет волос (каштан), умножается на яркость текстуры
HAIR_ALPHA_Q = (0.50, 0.80)        # квантили альфы для порога волос
LASH_ALPHA_GAIN = 2.5

FPS = 24
OUR_BONES = ["Hips", "Spine", "Chest", "Neck", "Head",
             "Shoulder.L", "UpperArm.L", "LowerArm.L", "Hand.L",
             "Shoulder.R", "UpperArm.R", "LowerArm.R", "Hand.R", "WeaponSocket.R",
             "UpperLeg.L", "LowerLeg.L", "Foot.L", "Toes.L",
             "UpperLeg.R", "LowerLeg.R", "Foot.R", "Toes.R"]

# ---------------------------------------------------------------- соответствие костей
# наша кость -> список исходных костей, чьи веса в неё сливаются
FINGERS = ["Thumb1", "Thumb2", "Thumb3", "Index1", "Index2", "Index3",
           "Mid1", "Mid2", "Mid3", "Ring1", "Ring2", "Ring3", "Pinky1", "Pinky2", "Pinky3"]
TOES = ["ToeBase", "ToeBaseShareBone", "BigToe1", "IndexToe1", "MidToe1", "RingToe1", "PinkyToe1"]

BONE_MAP = {
    "Hips": ["BoneRoot", "Hip", "Pelvis", "Waist"],
    "Spine": ["Spine01"],
    "Chest": ["Spine02", "L_RibsTwist", "R_RibsTwist", "L_Breast", "R_Breast"],
    "Neck": ["NeckTwist01", "NeckTwist02"],
    "Head": ["Head", "FacialBone", "JawRoot", "UpperJaw", "L_Eye", "R_Eye",
             "Teeth01", "Teeth02", "Tongue01", "Tongue02", "Tongue03"],
    "WeaponSocket.R": [],
}
for _s, _k in (("L", "L"), ("R", "R")):
    BONE_MAP["Shoulder." + _s] = [_k + "_Clavicle"]
    BONE_MAP["UpperArm." + _s] = [_k + "_Upperarm", _k + "_UpperarmTwist01", _k + "_UpperarmTwist02"]
    BONE_MAP["LowerArm." + _s] = [_k + "_Forearm", _k + "_ForearmTwist01", _k + "_ForearmTwist02",
                                  _k + "_ElbowShareBone"]
    BONE_MAP["Hand." + _s] = [_k + "_Hand"] + [_k + "_" + f for f in FINGERS]
    BONE_MAP["UpperLeg." + _s] = [_k + "_Thigh", _k + "_ThighTwist01", _k + "_ThighTwist02"]
    BONE_MAP["LowerLeg." + _s] = [_k + "_Calf", _k + "_CalfTwist01", _k + "_CalfTwist02",
                                  _k + "_KneeShareBone"]
    BONE_MAP["Foot." + _s] = [_k + "_Foot"]
    BONE_MAP["Toes." + _s] = [_k + "_" + t for t in TOES]

SRC2OUR = {}
for _o, _lst in BONE_MAP.items():
    for _s in _lst:
        SRC2OUR[_s] = _o

# пометки неочевидных мест (попадают в отчёт)
BONE_NOTES = {
    "Hips": "Pelvis+Hip+Waist (Waist 0.97-1.01 м лежит ниже нашего сустава Spine 1.04 м)",
    "Spine": "Spine01; сустав модели на ~3 см ниже нашего",
    "Chest": "Spine02 + RibsTwist + Breast (корректирующие кости груди слиты без отдельной кости)",
    "Neck": "NeckTwist01/02; у исходника шея 1.38-1.45 м, у нас Neck 1.31-1.40 м: сдвиг ~5-7 см",
    "Head": "Head + челюсть/глаза/язык/зубы (JawRoot, UpperJaw, L/R_Eye, FacialBone): у нас нет челюсти и глаз",
    "Shoulder.L/R": "L/R_Clavicle",
    "UpperArm.L/R": "ВНИМАНИЕ: кости L/R_Upperarm и L/R_Forearm в FBX без весов, их rest лежит в T-позе; "
                    "суставы берутся из весовых костей UpperarmTwist01 / ForearmTwist01 / Hand",
    "LowerArm.L/R": "Forearm + ForearmTwist01/02 + ElbowShareBone (корректирующая)",
    "Hand.L/R": "Hand + 15 костей пальцев слиты в одну кость (у нас пальцев нет)",
    "UpperLeg.L/R": "Thigh + ThighTwist01/02 (Thigh без весов; сустав берётся из ThighTwist01)",
    "LowerLeg.L/R": "Calf + CalfTwist01/02 + KneeShareBone",
    "Foot.L/R": "Foot; наш сустав лодыжки на 3.8 см выше анатомического (у нас предусмотрены сапоги)",
    "Toes.L/R": "ToeBase + ToeBaseShareBone + 5 пальцев стопы слиты в одну кость",
    "WeaponSocket.R": "весов нет: кость-крепление шпаги (как в hero.glb)",
}


# ---------------------------------------------------------------- утилиты

def log(*a):
    print("ADAPT", *a, flush=True)


def tri_count(o):
    me = o.data
    me.calc_loop_triangles()
    return len(me.loop_triangles)


def np_verts(o):
    me = o.data
    n = len(me.vertices)
    a = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", a)
    return a.reshape(n, 3)


def set_verts(o, arr):
    o.data.vertices.foreach_set("co", np.asarray(arr, dtype=np.float32).ravel())
    o.data.update()


def select_only(o):
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o


def bake_world(o):
    """Переносит мировую матрицу в геометрию, убирает родителя и модификаторы."""
    mw = o.matrix_world.copy()
    o.modifiers.clear()
    o.data.transform(mw)
    o.parent = None
    o.matrix_basis = Matrix.Identity(4)


def remove_shape_keys(o):
    if o.data.shape_keys:
        select_only(o)
        bpy.ops.object.shape_key_remove(all=True, apply_mix=True)


def faces_by_material(o, names):
    """Удаляет (mode='delete') либо отделяет (mode='split') грани по именам материалов."""
    pass


def delete_material_faces(o, mat_names):
    me = o.data
    bm = bmesh.new()
    bm.from_mesh(me)
    idx = {i for i, m in enumerate(me.materials) if m and m.name in mat_names}
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index in idx], context="FACES")
    bm.to_mesh(me)
    bm.free()


def split_material_faces(o, mat_name, new_name):
    """Отделяет грани материала mat_name в новый объект, сохраняя группы вершин."""
    me = o.data
    idx = [i for i, m in enumerate(me.materials) if m and m.name == mat_name]
    select_only(o)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for p in me.polygons:
        p.select = p.material_index in idx
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.ops.object.mode_set(mode="OBJECT")
    new = [x for x in bpy.context.selected_objects if x is not o][0]
    new.name = new_name
    new.data.name = new_name
    return new


def clean_loose(o):
    """Удаляет вершины без граней."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(o.data)
    bm.free()


def group_weights(o):
    """dict имя_группы -> массив весов (по вершинам)."""
    n = len(o.data.vertices)
    out = {}
    names = {vg.index: vg.name for vg in o.vertex_groups}
    for v in o.data.vertices:
        for g in v.groups:
            nm = names[g.group]
            if nm not in out:
                out[nm] = np.zeros(n)
            out[nm][v.index] = g.weight
    return out


def decimate(o, target, protect=None, delimit=("MATERIAL",)):
    """Collapse-децимация до target треугольников (итеративно уточняет ratio).

    protect: массив 0..1 по вершинам: где 1 - вершину стараемся сохранить (через группу 'keep').
    """
    select_only(o)
    orig = o.data.copy()
    n0 = tri_count(o)
    ratio = min(1.0, target / n0)
    best = None
    for it in range(6):
        if it:
            o.data = orig.copy()
        if protect is not None:
            for vg in [g for g in o.vertex_groups if g.name == "keep"]:
                o.vertex_groups.remove(vg)
            vg = o.vertex_groups.new(name="keep")
            for i, w in enumerate(protect):
                vg.add([i], float(w), "REPLACE")
        m = o.modifiers.new("dec", "DECIMATE")
        m.decimate_type = "COLLAPSE"
        m.ratio = ratio
        m.use_collapse_triangulate = True
        m.delimit = set(delimit)
        if protect is not None:
            m.vertex_group = "keep"
            m.invert_vertex_group = True       # вершины с весом 1 - защищены (проверено на результате)
            m.vertex_group_factor = 1.0
        select_only(o)
        bpy.ops.object.modifier_apply(modifier=m.name)
        n = tri_count(o)
        log("decimate", o.name, "iter", it, "ratio %.4f" % ratio, "tris", n0, "->", n, "target", target)
        if abs(n - target) / target < 0.04:
            break
        ratio *= target / max(n, 1)
    bpy.data.meshes.remove(orig)
    if protect is not None:
        for vg in [g for g in o.vertex_groups if g.name == "keep"]:
            o.vertex_groups.remove(vg)
    return tri_count(o)


# ---------------------------------------------------------------- стадия 1: импорт

def stage_import():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=FBX)
    arm = bpy.data.objects["Armature"]
    arm.name = "SrcArmature"
    arm.data.name = "SrcArmature"
    arm.data.pose_position = "REST"
    arm.animation_data_clear()
    for a in list(bpy.data.actions):
        bpy.data.actions.remove(a)
    bpy.context.view_layer.update()

    # лишнее
    for nm in ("Teeth", "Tongue", "TearLine", "EyeOcclusion", "top_cloth", "bottome"):
        bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)

    # запоминаем суставы исходного скелета в мировых координатах (REST)
    joints = {}
    for b in arm.data.bones:
        joints[b.name] = (arm.matrix_world @ b.head_local).copy()
    tails = {b.name: (arm.matrix_world @ b.tail_local).copy() for b in arm.data.bones}

    body = bpy.data.objects["Body"]
    eye = bpy.data.objects["Eye"]
    for o in (body, eye, bpy.data.objects["hair"], bpy.data.objects["Scalp_Male"]):
        remove_shape_keys(o)
    for o in (body, eye, bpy.data.objects["hair"], bpy.data.objects["Scalp_Male"]):
        bake_world(o)
    # корректируем имена
    bpy.data.objects["hair"].name = "Hair"
    bpy.data.objects["Scalp_Male"].name = "Scalp"
    eye.name = "Eyes"

    lash = split_material_faces(body, "Std_Eyelash", "Eyelashes")
    # ногти - часть поверхности пальцев (удаление оставляет дыры): отдаём их коже
    # (пальцы стоп -> Leg, пальцы рук -> Arm)
    me = body.data
    mi = {m.name: i for i, m in enumerate(me.materials) if m}
    for p in me.polygons:
        if p.material_index == mi["Std_Nails"]:
            zc = sum(me.vertices[v].co.z for v in p.vertices) / len(p.vertices)
            p.material_index = mi["Std_Skin_Leg"] if zc < 0.2 else mi["Std_Skin_Arm"]
    clean_loose(body)
    delete_material_faces(eye, {"Std_Cornea_R", "Std_Cornea_L"})
    clean_loose(eye)
    log("after import: tris", {o.name: tri_count(o) for o in bpy.data.objects if o.type == "MESH"})
    return arm, joints, tails


# ---------------------------------------------------------------- стадия 2: децимация

def stage_decimate(src_joints):
    body = bpy.data.objects["Body"]
    # защита вершин: лицо, кисти, стопы - по весам исходного скелета
    gw = group_weights(body)
    prot_names = ["Head", "JawRoot", "L_Eye", "R_Eye"] + \
                 ["%s_%s" % (s, f) for s in "LR" for f in ["Hand"] + FINGERS + TOES[2:]] + \
                 ["L_Foot", "R_Foot"]
    n = len(body.data.vertices)
    prot = np.zeros(n)
    for nm in prot_names:
        if nm in gw:
            prot = np.maximum(prot, gw[nm])
    # шея/голова ниже по весам тоже в Head; усиливаем только лицо (z > 1.42)
    co = np_verts(body)
    prot = np.where(co[:, 2] > 1.38, np.maximum(prot, 0.0), prot)
    log("protected verts", int((prot > 0.5).sum()), "of", n)
    PROT_STRENGTH = float(os.environ.get("ADAPT_PROT", "0"))
    # Эксперименты показали: любая ненулевая защита группы (даже 0.1) убирает децимацию
    # на всём остальном теле; равномерный Collapse даёт лучшее распределение (лицо ~1400, кисти ~500, стопы ~400).
    decimate(body, TRI_BODY, protect=(prot * PROT_STRENGTH if PROT_STRENGTH > 0 else None), delimit=("MATERIAL",))
    decimate(bpy.data.objects["Hair"], TRI_HAIR, delimit=("MATERIAL",))
    decimate(bpy.data.objects["Scalp"], TRI_SCALP, delimit=())
    decimate(bpy.data.objects["Eyes"], TRI_EYES, delimit=("MATERIAL",))
    decimate(bpy.data.objects["Eyelashes"], TRI_LASH, delimit=())


# ---------------------------------------------------------------- стадия 3: ретаргет

def load_our_rig():
    path = next(p for p in RIG_BLENDS if os.path.isfile(p))
    log("rig source", path)
    with bpy.data.libraries.load(path, link=False) as (df, dt):
        dt.objects = ["Armature"]
        dt.actions = ["Idle", "Walk", "Attack", "Death"]
    arm = dt.objects[0]
    bpy.context.scene.collection.objects.link(arm)
    arm.name = "Armature"
    for a in dt.actions:
        a.use_fake_user = True
    return arm, path


def rot_between(u, v):
    q = Vector(u).normalized().rotation_difference(Vector(v).normalized())
    return np.array(q.to_matrix())


def seg_matrix(pivot_src, pivot_dst, u_src_vec, v_tgt_vec, clamp=(0.8, 1.25), stretch=True):
    """Аффинное преобразование v' = pivot_dst + A (v - pivot_src): поворот u->v и растяжение вдоль u."""
    u = np.array(u_src_vec, dtype=float)
    L = np.linalg.norm(u)
    u = u / L
    d = np.array(v_tgt_vec, dtype=float)
    Lt = np.linalg.norm(d)
    R = rot_between(u, d / Lt)
    k = 1.0
    if stretch:
        k = min(max(Lt / L, clamp[0]), clamp[1])
    A = R @ (np.eye(3) + (k - 1.0) * np.outer(u, u))
    ps = np.array(pivot_src, dtype=float)
    pd = np.array(pivot_dst, dtype=float)
    return A, pd - A @ ps, k


def apply_aff(A, t, p):
    return A @ np.asarray(p, dtype=float) + t


def build_warps(sj, arm):
    """Матрицы (A, t) для каждой нашей кости. sj - мировые головы костей исходника."""
    P = lambda n: np.array(sj[n])
    ob = arm.data.bones
    H = lambda n: np.array(ob[n].head_local)
    T = lambda n: np.array(ob[n].tail_local)
    warps = {n: (np.eye(3), np.zeros(3), 1.0) for n in OUR_BONES}
    info = {}
    for s in "LR":
        # --- рука
        S, E, W = P(s + "_UpperarmTwist01"), P(s + "_ForearmTwist01"), P(s + "_Hand")
        K = P(s + "_Mid1")
        u_t = H("LowerArm." + s) - H("UpperArm." + s)
        f_t = H("Hand." + s) - H("LowerArm." + s)
        h_t = T("Hand." + s) - H("Hand." + s)
        A1, t1, k1 = seg_matrix(S, S, E - S, u_t)
        E1 = apply_aff(A1, t1, E)
        A2, t2, k2 = seg_matrix(E, E1, W - E, f_t)
        W2 = apply_aff(A2, t2, W)
        A3, t3, k3 = seg_matrix(W, W2, K - W, h_t, stretch=False)
        warps["UpperArm." + s] = (A1, t1, k1)
        warps["LowerArm." + s] = (A2, t2, k2)
        warps["Hand." + s] = (A3, t3, k3)
        info["arm." + s] = {"shoulder_src": S.round(4).tolist(), "elbow_src": E.round(4).tolist(),
                            "wrist_src": W.round(4).tolist(), "k_upper": round(k1, 3), "k_lower": round(k2, 3),
                            "shoulder_offset_to_rig": (S - H("UpperArm." + s)).round(4).tolist(),
                            "elbow_after": E1.round(4).tolist(), "elbow_rig": H("LowerArm." + s).round(4).tolist(),
                            "wrist_after": W2.round(4).tolist(), "wrist_rig": H("Hand." + s).round(4).tolist()}
        # --- нога
        Hp, Kn, An = P(s + "_ThighTwist01"), P(s + "_CalfTwist01"), P(s + "_Foot")
        th_t = H("LowerLeg." + s) - H("UpperLeg." + s)
        B1, b1, m1 = seg_matrix(Hp, Hp, Kn - Hp, th_t)
        Kn1 = apply_aff(B1, b1, Kn)
        ca_t = H("Foot." + s) - H("LowerLeg." + s)
        ca_t = np.array([ca_t[0], ca_t[1], An[2] - Kn1[2]])   # лодыжка остаётся на высоте исходной (подошва на z=0)
        B2, b2, m2 = seg_matrix(Kn, Kn1, An - Kn, ca_t)
        An1 = apply_aff(B2, b2, An)
        shift = An1 - An
        warps["UpperLeg." + s] = (B1, b1, m1)
        warps["LowerLeg." + s] = (B2, b2, m2)
        warps["Foot." + s] = (np.eye(3), shift, 1.0)
        warps["Toes." + s] = (np.eye(3), shift, 1.0)
        info["leg." + s] = {"hip_src": Hp.round(4).tolist(), "knee_src": Kn.round(4).tolist(),
                            "ankle_src": An.round(4).tolist(), "k_thigh": round(m1, 3), "k_calf": round(m2, 3),
                            "knee_after": Kn1.round(4).tolist(), "knee_rig": H("LowerLeg." + s).round(4).tolist(),
                            "ankle_after": An1.round(4).tolist(), "ankle_rig": H("Foot." + s).round(4).tolist(),
                            "foot_shift": shift.round(4).tolist()}
    return warps, info


def merged_weights(o):
    """Матрица (nv, 22) весов в наших костях из групп исходных костей."""
    n = len(o.data.vertices)
    W = np.zeros((n, len(OUR_BONES)))
    idx = {b: i for i, b in enumerate(OUR_BONES)}
    unknown = {}
    names = {vg.index: vg.name for vg in o.vertex_groups}
    for v in o.data.vertices:
        for g in v.groups:
            nm = names[g.group]
            if nm in SRC2OUR:
                W[v.index, idx[SRC2OUR[nm]]] += g.weight
            elif g.weight > 1e-4:
                unknown[nm] = unknown.get(nm, 0) + 1
    if unknown:
        log("UNMAPPED GROUPS", o.name, unknown)
    return W, unknown


def limit_weights(W, maxn=4, eps=0.01):
    W = W.copy()
    for i in range(W.shape[0]):
        row = W[i]
        order = np.argsort(-row)
        keep = order[:maxn]
        mask = np.zeros_like(row, dtype=bool)
        mask[keep] = True
        row[~mask] = 0.0
        row[row < eps] = 0.0
        s = row.sum()
        if s > 0:
            row /= s
    return W


def retarget_mesh(o, warps, arm_obj):
    """Применяет линейный скиннинг (веса сливаются в наши кости), переписывает группы."""
    W, unk = merged_weights(o)
    s = W.sum(axis=1)
    unweighted = int((s < 1e-6).sum())
    log("retarget", o.name, "verts", len(W), "unweighted before:", unweighted)
    if unweighted:
        # вершины без весов - в Head (только не должно случаться)
        W[s < 1e-6, OUR_BONES.index("Head")] = 1.0
        s = W.sum(axis=1)
    W = W / s[:, None]
    W = limit_weights(W)
    co = np_verts(o)
    out = np.zeros_like(co)
    for j, b in enumerate(OUR_BONES):
        A, t, _ = warps[b]
        out += W[:, j:j + 1] * (co @ A.T + t)
    set_verts(o, out)
    for vg in list(o.vertex_groups):
        o.vertex_groups.remove(vg)
    for j, b in enumerate(OUR_BONES):
        nz = np.nonzero(W[:, j] > 0)[0]
        if len(nz) == 0:
            continue
        vg = o.vertex_groups.new(name=b)
        for i in nz:
            vg.add([int(i)], float(W[i, j]), "REPLACE")
    o.parent = arm_obj
    o.matrix_parent_inverse = Matrix.Identity(4)
    m = o.modifiers.new("Armature", "ARMATURE")
    m.object = arm_obj
    return W, unweighted


# ---------------------------------------------------------------- текстуры и материалы

def img_to_np(im):
    w, h = im.size
    a = np.empty(w * h * im.channels, dtype=np.float32)
    im.pixels.foreach_get(a)
    return a.reshape(h, w, im.channels)


def np_to_img(name, arr, colorspace="sRGB"):
    h, w, c = arr.shape
    if c == 3:
        arr = np.concatenate([arr, np.ones((h, w, 1), dtype=np.float32)], axis=2)
    im = bpy.data.images.new(name, w, h, alpha=True)
    im.colorspace_settings.name = colorspace      # до записи пикселей: смена пространства сбрасывает буфер
    im.pixels.foreach_set(arr.astype(np.float32).ravel())
    return im


def srgb_to_lin(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def scaled_copy(name_src, size_max, new_name=None, colorspace=None):
    im = bpy.data.images[name_src].copy()
    if colorspace:                      # до scale(): смена пространства перечитывает буфер
        im.colorspace_settings.name = colorspace
    w, h = im.size
    k = min(1.0, size_max / max(w, h))
    nw, nh = max(1, int(round(w * k))), max(1, int(round(h * k)))
    if (nw, nh) != (w, h):
        im.scale(nw, nh)
    if new_name:
        im.name = new_name
    return im


def save_image(im, fname, fmt):
    os.makedirs(os.path.join(WORK, "tex"), exist_ok=True)
    p = os.path.join(WORK, "tex", fname)
    im.file_format = fmt
    if fmt == "JPEG":
        bpy.context.scene.render.image_settings.quality = 88
    im.filepath_raw = p
    im.save()
    out = bpy.data.images.load(p, check_existing=False)
    out.name = fname
    out.colorspace_settings.name = im.colorspace_settings.name
    out.pack()
    bpy.data.images.remove(im)
    return out


def new_material(name, color_img=None, normal_img=None, alpha=False, rough=0.55, cutoff=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if color_img:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = color_img
        t.interpolation = "Linear"
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
        if alpha:
            nt.links.new(t.outputs["Alpha"], bsdf.inputs["Alpha"])
    if normal_img:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = normal_img
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if alpha:
        m.blend_method = "CLIP"
        m.alpha_threshold = cutoff
    return m


def remap_materials(o, old_to_new, default=None):
    me = o.data
    old = [m.name if m else None for m in me.materials]
    new_list = []
    for nm in old_to_new.values():
        if nm not in new_list:
            new_list.append(nm)
    idx = [new_list.index(old_to_new.get(old[p.material_index] if p.material_index < len(old) else None,
                                         new_list[0])) for p in me.polygons]
    me.materials.clear()                # clear() обнуляет material_index, поэтому индексы выставляем после
    for nm in new_list:
        me.materials.append(bpy.data.materials[nm])
    for p, i in zip(me.polygons, idx):
        p.material_index = i


def build_textures():
    """Создаёт все материалы с текстурами <= 512 px. Возвращает отчёт."""
    rep = []

    def note(im, src, fmt):
        rep.append({"name": im.name, "src": src, "size": list(im.size), "format": fmt})

    skin_tint = srgb_to_lin(SKIN_TINT)
    for part in ("Head", "Body", "Arm", "Leg"):
        d = scaled_copy("Std_Skin_%s_Diffuse" % part, TEX_MAX)
        a = img_to_np(d)
        a[..., :3] = np.clip(a[..., :3] * skin_tint, 0, 1)
        a[..., 3] = 1.0
        bpy.data.images.remove(d)
        d2 = save_image(np_to_img("skin_%s_d" % part.lower(), a), "skin_%s_d.png" % part.lower(), "PNG")
        n = scaled_copy("Std_Skin_%s_Normal" % part, TEX_MAX, colorspace="Non-Color")
        n = save_image(n, "skin_%s_n.png" % part.lower(), "PNG")
        new_material("Skin_" + part, d2, n, rough=0.6)
        note(d2, "Std_Skin_%s_Diffuse" % part, "PNG")
        note(n, "Std_Skin_%s_Normal" % part, "PNG")

    # волосы: яркость исходной серой текстуры, цвет - каштан, альфа - порог по квантилям
    for src, tag, tint, qs in (("Default_Material_Transparency_Diffuse", "hair", HAIR_TINT, HAIR_ALPHA_Q),
                               ("Scalp1_Transparency_Diffuse", "scalp", HAIR_TINT, (0.55, 0.85))):
        im = scaled_copy(src, TEX_MAX)
        a = img_to_np(im)
        bpy.data.images.remove(im)
        lum = a[..., :3].mean(axis=2, keepdims=True)
        lum = np.clip(lum / max(float(np.quantile(lum, 0.5)), 1e-3), 0.0, 1.6) * 0.5 + 0.55
        col = srgb_to_lin(tint)[None, None, :] * lum
        al = a[..., 3]
        lo, hi = float(np.quantile(al, qs[0])), float(np.quantile(al, qs[1]))
        al2 = np.clip((al - lo) / max(hi - lo, 1e-4), 0, 1)
        rep.append({"name": tag + "_alpha", "alpha_q": [round(lo, 4), round(hi, 4)],
                    "coverage_after": round(float((al2 > 0.5).mean()), 3),
                    "coverage_before_at_0.5": round(float((al > 0.5).mean()), 3)})
        arr = np.concatenate([np.clip(col, 0, 1), al2[..., None]], axis=2)
        d = save_image(np_to_img("%s_d" % tag, arr), "%s_d.png" % tag, "PNG")
        new_material("Hair" if tag == "hair" else "Scalp", d, None, alpha=True, rough=0.45)
        note(d, src, "PNG")

    # ресницы
    im = scaled_copy("Std_Eyelash_Diffuse", TEX_MAX)
    a = img_to_np(im)
    bpy.data.images.remove(im)
    a[..., :3] = 0.02
    a[..., 3] = np.clip(a[..., 3] * LASH_ALPHA_GAIN, 0, 1)
    d = save_image(np_to_img("lash_d", a), "lash_d.png", "PNG")
    new_material("Eyelash", d, None, alpha=True, rough=0.8)
    note(d, "Std_Eyelash_Diffuse", "PNG")

    # глаза
    for s in "RL":
        im = scaled_copy("Std_Eye_%s_Diffuse" % s, EYE_TEX)
        d = save_image(im, "eye_%s_d.jpg" % s.lower(), "JPEG")
        new_material("Eye_" + s, d, None, rough=0.25)
        note(d, "Std_Eye_%s_Diffuse" % s, "JPEG")
    return rep


MAT_MAP_BODY = {"Std_Skin_Head": "Skin_Head", "Std_Skin_Body": "Skin_Body",
                "Std_Skin_Arm": "Skin_Arm", "Std_Skin_Leg": "Skin_Leg"}


def assign_materials():
    remap_materials(bpy.data.objects["Body"], MAT_MAP_BODY)
    remap_materials(bpy.data.objects["Hair"], {"Default_Material_Transparency": "Hair"})
    remap_materials(bpy.data.objects["Scalp"], {"Scalp1_Transparency": "Scalp"})
    remap_materials(bpy.data.objects["Eyelashes"], {"Std_Eyelash": "Eyelash"})
    remap_materials(bpy.data.objects["Eyes"], {"Std_Eye_R": "Eye_R", "Std_Eye_L": "Eye_L"})


def purge_old_images():
    keep = {i.name for i in bpy.data.images if i.name.endswith((".png", ".jpg")) and i.packed_file}
    keep |= {"skin_head_d", "skin_body_d", "skin_arm_d", "skin_leg_d"}
    for im in list(bpy.data.images):
        if im.users == 0 or im.name not in keep:
            if im.name in keep:
                continue
            bpy.data.images.remove(im)


# ---------------------------------------------------------------- GLB

def read_glb(path):
    data = open(path, "rb").read()
    clen, ctype = struct.unpack("<I4s", data[12:20])
    doc = json.loads(data[20:20 + clen].decode("utf-8"))
    binoff = 20 + clen
    return doc, data, binoff + 8


def image_dims(doc, data, binstart, im):
    bv = doc["bufferViews"][im["bufferView"]]
    b = data[binstart + bv.get("byteOffset", 0): binstart + bv.get("byteOffset", 0) + bv["byteLength"]]
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", b[16:24])
        return w, h, "png", bv["byteLength"]
    i = 2
    while i < len(b):
        if b[i] != 0xFF:
            i += 1
            continue
        mk = b[i + 1]
        if mk in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", b[i + 5:i + 9])
            return w, h, "jpeg", bv["byteLength"]
        ln = struct.unpack(">H", b[i + 2:i + 4])[0]
        i += 2 + ln
    return 0, 0, "?", bv["byteLength"]


def glb_report(path):
    doc, data, binstart = read_glb(path)
    tris = 0
    per_mesh = {}
    for m in doc["meshes"]:
        t = 0
        for p in m["primitives"]:
            if "indices" in p:
                t += doc["accessors"][p["indices"]]["count"] // 3
            else:
                t += doc["accessors"][p["attributes"]["POSITION"]]["count"] // 3
        per_mesh[m["name"]] = t
        tris += t
    anims = []
    for a in doc.get("animations", []):
        tmax = max(doc["accessors"][s["input"]]["max"][0] for s in a["samplers"])
        anims.append((a["name"], round(tmax, 4), len(a["channels"])))
    imgs = [(im.get("name"),) + image_dims(doc, data, binstart, im) for im in doc.get("images", [])]
    wmin, bad = 9.0, 0
    for m in doc["meshes"]:
        for pr in m["primitives"]:
            if "WEIGHTS_0" not in pr["attributes"]:
                bad += 1
                continue
            ac = doc["accessors"][pr["attributes"]["WEIGHTS_0"]]
            bv = doc["bufferViews"][ac["bufferView"]]
            off = binstart + bv.get("byteOffset", 0) + ac.get("byteOffset", 0)
            ct = {5126: ("<f4", 1.0), 5123: ("<u2", 65535.0), 5121: ("u1", 255.0)}[ac["componentType"]]
            w = np.frombuffer(data, dtype=ct[0], count=ac["count"] * 4, offset=off).reshape(-1, 4) / ct[1]
            wmin = min(wmin, float(w.sum(axis=1).min()))
    if "skins" in doc:
        pass
    return {"file_bytes": len(data), "meshes": len(doc["meshes"]), "mesh_tris": per_mesh, "tris_total": tris,
            "skins": len(doc.get("skins", [])), "joints": len(doc["skins"][0]["joints"]) if doc.get("skins") else 0,
            "joint_names": [doc["nodes"][j]["name"] for j in doc["skins"][0]["joints"]] if doc.get("skins") else [],
            "animations": anims, "materials": [m["name"] for m in doc["materials"]],
            "alpha_modes": {m["name"]: m.get("alphaMode", "OPAQUE") for m in doc["materials"]},
            "images": imgs, "textures": len(doc.get("textures", [])),
            "min_weight_sum": round(wmin, 4), "primitives_without_weights": bad}


# ---------------------------------------------------------------- проверки

def meshes():
    return [bpy.data.objects[n] for n in ("Body", "Eyelashes", "Eyes", "Hair", "Scalp")]


def name_mesh_data():
    for o in meshes():
        o.data.name = o.name


def eval_coords(o):
    dg = bpy.context.evaluated_depsgraph_get()
    e = o.evaluated_get(dg)
    me = e.to_mesh()
    n = len(me.vertices)
    a = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", a)
    e.to_mesh_clear()
    return a.reshape(n, 3)


def edges_of(o):
    me = o.data
    a = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", a)
    return a.reshape(-1, 2)


def rotation_test(arm, amount=0.35):
    """Поворачивает каждую кость на 0.35 рад вокруг X, Y, Z (локально); меряет растяжение рёбер."""
    arm.animation_data_clear()
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
    bpy.context.view_layer.update()
    res = {}
    rest = {o.name: eval_coords(o) for o in meshes()}
    ed = {o.name: edges_of(o) for o in meshes()}
    rl = {n: np.linalg.norm(rest[n][ed[n][:, 0]] - rest[n][ed[n][:, 1]], axis=1) for n in rest}
    for b in OUR_BONES:
        pb = arm.pose.bones[b]
        worst = {"stretch_ratio": 1.0, "stretch_abs_cm": 0.0, "mesh": "", "axis": ""}
        moved = 0
        for ax in "XYZ":
            pb.rotation_quaternion = Quaternion(Vector({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[ax]), amount)
            bpy.context.view_layer.update()
            for o in meshes():
                c = eval_coords(o)
                e = ed[o.name]
                nl = np.linalg.norm(c[e[:, 0]] - c[e[:, 1]], axis=1)
                msk = rl[o.name] > 0.004
                ratio = nl[msk] / rl[o.name][msk]
                dabs = (nl - rl[o.name])
                if ratio.size and ratio.max() > worst["stretch_ratio"]:
                    worst["stretch_ratio"] = float(ratio.max())
                    worst["mesh"], worst["axis"] = o.name, ax
                worst["stretch_abs_cm"] = max(worst["stretch_abs_cm"], float(dabs.max() * 100))
                moved = max(moved, int((np.linalg.norm(c - rest[o.name], axis=1) > 1e-3).sum()))
        pb.rotation_quaternion = (1, 0, 0, 0)
        bpy.context.view_layer.update()
        worst["verts_moved"] = moved
        worst["stretch_ratio"] = round(worst["stretch_ratio"], 3)
        worst["stretch_abs_cm"] = round(worst["stretch_abs_cm"], 2)
        res[b] = worst
    return res


def action_floor(arm, meshes_, frames_by_action):
    """Минимальный Z вершин по кадрам каждого action (проваливание)."""
    out = {}
    scene = bpy.context.scene
    for name, (f0, f1) in frames_by_action.items():
        arm.animation_data_create()
        arm.animation_data.action = bpy.data.actions[name]
        zs = []
        for f in range(f0, f1 + 1):
            scene.frame_set(f)
            z = min(eval_coords(o)[:, 2].min() for o in meshes_)
            zs.append(float(z))
        out[name] = {"min_z_min": round(min(zs) + 0.0, 4), "min_z_max": round(max(zs) + 0.0, 4),
                     "min_z_mean": round(float(np.mean(zs)), 4)}
    return out


def floor_fix(arm, frames_by_action):
    """Выравнивает каждую позу по полу (min Z вершин == 0), как в hero_anim.py: поправка к ключам
    Hips.location в наших КОПИЯХ action (исходный .blend не сохраняется). Поправка в мировом Z
    переводится в локальную систему кости Hips."""
    scene = bpy.context.scene
    inv = np.linalg.inv(np.array(arm.data.bones["Hips"].matrix_local.to_3x3()))
    rep = {}
    for name, (f0, f1) in frames_by_action.items():
        act = bpy.data.actions[name]
        arm.animation_data_create()
        arm.animation_data.action = act
        fcs = {fc.array_index: fc for fc in act.fcurves if fc.data_path == 'pose.bones["Hips"].location'}
        shifts = []
        for f in range(f0, f1 + 1):
            scene.frame_set(f)
            d = -min(eval_coords(o)[:, 2].min() for o in meshes())
            shifts.append(d)
            loc = inv @ np.array([0.0, 0.0, d])
            for i, fc in fcs.items():
                ks = [k for k in fc.keyframe_points if abs(k.co[0] - f) < 1e-4]
                if not ks:
                    raise RuntimeError("нет ключа Hips.location кадр %d в %s" % (f, name))
                for h in ("co", "handle_left", "handle_right"):
                    v = getattr(ks[0], h)
                    v[1] += loc[i]
        for fc in fcs.values():
            fc.update()
        rep[name] = {"shift_min_cm": round(float(min(shifts)) * 100, 2), "shift_max_cm": round(float(max(shifts)) * 100, 2)}
    return rep


# ---------------------------------------------------------------- превью

def stitch(paths, out_path):
    imgs = []
    for p in paths:
        im = bpy.data.images.load(p, check_existing=False)
        w, h = im.size
        buf = np.empty(w * h * 4, dtype=np.float32)
        im.pixels.foreach_get(buf)
        imgs.append(buf.reshape(h, w, 4))
        bpy.data.images.remove(im)
    strip = np.concatenate(imgs, axis=1)
    h, w = strip.shape[0], strip.shape[1]
    out = bpy.data.images.new("strip_tmp", w, h, alpha=True)
    out.pixels.foreach_set(strip.ravel())
    out.filepath_raw = out_path
    out.file_format = "PNG"
    out.save()
    bpy.data.images.remove(out)


def aim_cam(cam_obj, pos, target):
    cam_obj.location = pos
    cam_obj.rotation_euler = (Vector(target) - Vector(pos)).normalized().to_track_quat("-Z", "Y").to_euler()


def render_strip(arm, action, frames, out_png, cam_pos, cam_target, scene, cam_obj):
    arm.animation_data_create()
    arm.animation_data.action = bpy.data.actions[action]
    aim_cam(cam_obj, cam_pos, cam_target)
    paths = []
    for i, f in enumerate(frames):
        scene.frame_set(f)
        p = os.path.join(bpy.app.tempdir, "prev_%s_%d_%d.png" % (action, int(cam_pos[0] * 10) + 50, i))
        scene.render.filepath = p
        bpy.ops.render.render(write_still=True)
        paths.append(p)
    stitch(paths, out_png)


def setup_render(scene, res=(320, 480), ortho=2.7):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 4
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    world = bpy.data.worlds.new("PrevWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.9, 0.9, 0.88, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    scene.world = world
    sd = bpy.data.lights.new("PrevSun", "SUN")
    sd.energy = 3.0
    sun = bpy.data.objects.new("PrevSun", sd)
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(-30))
    scene.collection.objects.link(sun)
    fd = bpy.data.lights.new("PrevFill", "AREA")
    fd.energy = 200.0
    fd.size = 3.0
    fill = bpy.data.objects.new("PrevFill", fd)
    fill.location = (-2.5, -3.0, 2.5)
    fill.rotation_euler = (math.radians(55), 0.0, math.radians(-40))
    scene.collection.objects.link(fill)
    cd = bpy.data.cameras.new("PrevCam")
    cd.type = "ORTHO"
    cd.ortho_scale = ortho
    cam = bpy.data.objects.new("PrevCam", cd)
    scene.collection.objects.link(cam)
    scene.camera = cam
    # пол
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
    fl = bpy.context.active_object
    fl.name = "PrevFloor"
    fm = bpy.data.materials.new("PrevFloor")
    fm.use_nodes = True
    fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.35, 0.37, 0.4, 1)
    fl.data.materials.append(fm)
    return cam, cd


def render_previews(arm, scene):
    cam, cd = setup_render(scene)
    t34 = (0.0, 0.3, 0.9)
    c34 = (-2.2, -2.8, 1.5)
    os.makedirs(DOCS, exist_ok=True)
    render_strip(arm, "Idle", [0, 8, 16, 24, 32, 40], os.path.join(DOCS, "adapted_idle.png"), c34, t34, scene, cam)
    render_strip(arm, "Walk", [0, 4, 8, 12, 16, 20], os.path.join(DOCS, "adapted_walk.png"), c34, t34, scene, cam)
    render_strip(arm, "Attack", [0, 4, 6, 9, 13, 20], os.path.join(DOCS, "adapted_attack.png"), c34, t34, scene, cam)
    render_strip(arm, "Death", [0, 8, 14, 20, 28, 36], os.path.join(DOCS, "adapted_death.png"), c34, (0.0, 0.3, 0.3),
                 scene, cam)
    # лицо: фронт, 3/4, профиль; поза покоя (Idle кадр 0)
    scene.render.resolution_x, scene.render.resolution_y = 480, 480
    cd.ortho_scale = 0.42
    arm.animation_data.action = bpy.data.actions["Idle"]
    scene.frame_set(0)
    ft = (0.0, 0.0, 1.50)
    paths = []
    for i, pos in enumerate(((0.0, -3.0, 1.50), (-1.6, -2.6, 1.58), (-3.0, 0.0, 1.50))):
        aim_cam(cam, pos, ft)
        p = os.path.join(bpy.app.tempdir, "face_%d.png" % i)
        scene.render.filepath = p
        bpy.ops.render.render(write_still=True)
        paths.append(p)
    stitch(paths, os.path.join(DOCS, "adapted_face.png"))


# ---------------------------------------------------------------- main

def main():
    report = {"inputs": {"fbx": FBX}, "bone_map": BONE_MAP, "bone_notes": BONE_NOTES,
              "budget": {"body": TRI_BODY, "hair": TRI_HAIR, "scalp": TRI_SCALP, "eyes": TRI_EYES, "lashes": TRI_LASH}}
    src_arm, sj, st = stage_import()
    if STOP == "import":
        return
    stage_decimate(sj)
    report["tris_after_decimate"] = {o.name: tri_count(o) for o in meshes()}
    log("tris after decimate", report["tris_after_decimate"])
    if STOP == "decimate":
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(WORK, "decimated.blend"))
        return

    arm, rig_path = load_our_rig()
    report["inputs"]["rig_blend"] = rig_path
    warps, winfo = build_warps(sj, arm)
    report["retarget"] = winfo
    report["segment_stretch"] = {b: round(float(warps[b][2]), 3) for b in OUR_BONES if abs(warps[b][2] - 1.0) > 1e-6}
    log("segment stretch", report["segment_stretch"])
    wreport = {}
    for o in meshes():
        W, unw = retarget_mesh(o, warps, arm)
        wreport[o.name] = {"verts": len(o.data.vertices), "unweighted_after_merge": unw,
                           "max_influences": int((W > 0).sum(axis=1).max()),
                           "zero_weight_rows": int((W.sum(axis=1) < 0.999).sum())}
    report["weights"] = wreport

    # материалы
    report["textures"] = build_textures()
    assign_materials()
    for im in list(bpy.data.images):
        if im.name in ("Render Result",):
            continue
    # удаляем исходные картинки и материалы FBX
    used = {n.image for m in bpy.data.materials if m.node_tree for n in m.node_tree.nodes
            if n.type == "TEX_IMAGE" and n.image}
    for m in list(bpy.data.materials):
        if m.users == 0:
            bpy.data.materials.remove(m)
    for im in list(bpy.data.images):
        if im not in used and im.name != "Render Result":
            bpy.data.images.remove(im)
    bpy.data.objects.remove(src_arm, do_unlink=True)
    name_mesh_data()
    for a in list(bpy.data.actions):
        if a.name not in ("Idle", "Walk", "Attack", "Death"):
            bpy.data.actions.remove(a)

    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.frame_start, scene.frame_end = 0, 48
    arm.animation_data_create()
    arm.animation_data.action = bpy.data.actions["Idle"]
    arm.data.pose_position = "POSE"
    bpy.context.view_layer.update()

    report["rotation_test"] = rotation_test(arm)
    for b, r in report["rotation_test"].items():
        log("rot0.35", b, r)
    arm.animation_data_create()
    arm.animation_data.action = bpy.data.actions["Idle"]
    report["floor"] = action_floor(arm, meshes(), {"Idle": (0, 48), "Walk": (0, 24), "Attack": (0, 20),
                                                    "Death": (0, 36)})
    log("floor before", report["floor"])
    if FLOOR_FIX:
        report["floor_fix_shifts"] = floor_fix(arm, {"Idle": (0, 48), "Walk": (0, 24), "Attack": (0, 20),
                                                     "Death": (0, 36)})
        report["floor_before_fix"] = report["floor"]
        report["floor"] = action_floor(arm, meshes(), {"Idle": (0, 48), "Walk": (0, 24), "Attack": (0, 20),
                                                        "Death": (0, 36)})
        log("floor fix shifts", report["floor_fix_shifts"])
        log("floor after", report["floor"])
    if STOP == "retarget":
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(WORK, "retargeted.blend"))
        return

    # экспорт
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    for o in meshes():
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_scene.gltf(
        filepath=OUT_GLB, export_format="GLB", use_selection=True,
        export_animations=True, export_animation_mode="ACTIONS", export_frame_range=False,
        export_force_sampling=True, export_skins=True, export_yup=True,
        export_image_format="AUTO", export_apply=False, export_tangents=False)
    report["glb"] = glb_report(OUT_GLB)
    report["glb"]["joints_match_our_rig"] = sorted(report["glb"]["joint_names"]) == sorted(OUR_BONES)
    log("GLB", json.dumps({k: v for k, v in report["glb"].items() if k != "joint_names"}))
    bpy.ops.object.select_all(action="DESELECT")

    if not NORENDER:
        render_previews(arm, scene)
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "adapted_body_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False, default=str)
    if os.environ.get("ADAPT_SAVE"):
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(WORK, "final.blend"))
    log("DONE")


if __name__ == "__main__":
    main()
