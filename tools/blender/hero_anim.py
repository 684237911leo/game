"""Анимации героини: Idle, Walk, Attack, Death; крепление шпаги; экспорт GLB и превью.

Запуск из корня репозитория:
  PYTHONPATH=<pylib с numpy> blender -b -P tools/blender/hero_anim.py

Вход (только чтение): models/hero.blend, models/rapier.blend
Выход: models/hero_anim.blend (со шпагой), models/hero_anim.glb (без шпаги),
       models/rapier_attach.json, docs/models/anim_*.png

Соглашения:
  - Все кости позы: rotation_mode = QUATERNION, ключи rotation_quaternion
    (+ location только у Hips). Ключи ставятся на каждый кадр.
  - Цикл Idle: 48 кадров, ключи 0..48, кадр 48 == кадр 0 (длительность 2.0 с).
  - Цикл Walk: 24 кадра, ключи 0..24, кадр 24 == кадр 0 (длительность 1.0 с).
  - Attack: 20 кадров, ключи 0..20, кадр 20 == стойка (не зациклен).
  - Death: 36 кадров, ключи 0..36, без возврата.
  - Положение: каждая поза выравнивается по полу: минимальная Z вершин тела
    (без шпаги) == 0 (поправка всей позы целиком, жёсткий сдвиг).
  - Ходьба: шаг 0.45 м, скорость 0.9 м/с, ступня в опоре неподвижна относительно
    земли при корневом движении такой скорости.

Константы и правила ориентации: вперёд = -Y, вверх = +Z, левая сторона = +X.
Положительный угол наклона вперёд (fwd) = поворот верха к -Y.
"""

import bpy
import json
import math
import os
import struct

import numpy as np
from mathutils import Matrix, Quaternion, Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if not os.path.isfile(os.path.join(ROOT, "models", "hero.blend")):
    ROOT = "/home/user/game"
MODELS = os.path.join(ROOT, "models")
DOCS = os.path.join(ROOT, "docs", "models")

FPS = 24
IDLE_N, WALK_N, ATTACK_N, DEATH_N = 48, 24, 20, 36
WALK_STEP = 0.45            # м, шаг (между контактами разных ног)
WALK_SPEED = 2 * WALK_STEP  # м/с: два шага за один цикл 1 с
WALK_STANCE = 0.6           # доля фазы опоры
WALK_EXCURSION = WALK_SPEED * WALK_STANCE / 2  # полувыход ступни вперёд/назад, м
ANKLE_REST_Z = 0.09


# ---------------------------------------------------------------- утилиты

def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def key(f, keys):
    """Кусочно-косинусная интерполяция по ключам [(кадр, значение)]."""
    if f <= keys[0][0]:
        return keys[0][1]
    for (f0, v0), (f1, v1) in zip(keys, keys[1:]):
        if f0 <= f <= f1:
            return v0 + (v1 - v0) * ease((f - f0) / (f1 - f0))
    return keys[-1][1]


def vdir(fwd_deg, side_deg):
    """Направление кости от вертикали: наклон вперёд и вбок (+X)."""
    a = math.radians(fwd_deg)
    b = math.radians(side_deg)
    return Vector((math.sin(b), -math.sin(a) * math.cos(b), math.cos(a) * math.cos(b)))


def arm_dir(side, fwd_deg, out=0.25):
    """Направление руки вниз: side=+1 левая, -1 правая; fwd — наклон вперёд."""
    a = math.radians(fwd_deg)
    return Vector((side * out, -math.sin(a), -math.cos(a)))


def foot_vec(pitch_deg):
    """Ступня: pitch>0 — носок вверх, pitch=0 — горизонтально вперёд (-Y)."""
    p = math.radians(pitch_deg)
    return Vector((0.0, -math.cos(p), math.sin(p)))


def V(x, y, z):
    return Vector((x, y, z))


# ---------------------------------------------------------------- риг

class Rig:
    """Прямая кинематика костей в Python.

    Для каждой кости задаётся желаемая ориентация в armature-пространстве
    (aim), сохраняется локальная поза P = inv(M_parent_posed @ L_rest) @ F.
    Совпадение с Blender проверяется функцией check_fk().
    """

    def __init__(self, arm):
        self.arm = arm
        self.pb = arm.pose.bones
        self.parent, self.B, self.len = {}, {}, {}
        for b in arm.data.bones:
            self.B[b.name] = b.matrix_local.copy()
            self.parent[b.name] = b.parent.name if b.parent else None
            self.len[b.name] = b.length

        def depth(n):
            d, p = 0, self.parent[n]
            while p:
                d, p = d + 1, self.parent[p]
            return d

        self.order = sorted(self.B, key=depth)
        self.L = {}
        for n in self.B:
            p = self.parent[n]
            self.L[n] = self.B[p].inverted() @ self.B[n] if p else self.B[n].copy()
        for pbone in self.pb:
            pbone.rotation_mode = "QUATERNION"
        self.reset()

    def reset(self):
        self.P = {n: Matrix.Identity(4) for n in self.order}
        self.F = {}
        self.fk_all()

    def M(self, n):
        p = self.parent[n]
        return (self.F[p] @ self.L[n]) if p else self.L[n]

    def fk_all(self):
        for n in self.order:
            self.F[n] = self.M(n) @ self.P[n]

    def aim(self, n, d, twist=0.0, head=None):
        """Ориентирует кость n вдоль d (armature-пространство), twist — вокруг d."""
        Mc = self.M(n)
        B3 = self.B[n].to_3x3()
        rest_dir = (B3 @ Vector((0.0, 1.0, 0.0))).normalized()
        d = Vector(d).normalized()
        R3 = rest_dir.rotation_difference(d).to_matrix() @ B3
        if twist:
            R3 = Quaternion(d, math.radians(twist)).to_matrix() @ R3
        h = Vector(head) if head is not None else Mc.translation.copy()
        Fd = Matrix.Translation(h) @ R3.to_4x4()
        self.P[n] = Mc.inverted() @ Fd
        self.F[n] = Fd
        return Fd

    def ik(self, up, lo, target, pole):
        """Двухкостный IK в 3D: цель — конец цепи (lo.tail), pole — направление колена/локтя."""
        S = self.M(up).translation
        L1, L2 = self.len[up], self.len[lo]
        T = Vector(target)
        d = T - S
        dist = d.length
        maxr = (L1 + L2) * 0.9995
        if dist > maxr:
            T = S + d.normalized() * maxr
            dist = maxr
        u = (T - S).normalized()
        a = (L1 * L1 - L2 * L2 + dist * dist) / (2.0 * dist)
        h = math.sqrt(max(L1 * L1 - a * a, 0.0))
        base = S + u * a
        pv = Vector(pole) - S
        pv = pv - u * pv.dot(u)
        if pv.length < 1e-6:
            pv = Vector((0.0, -1.0, 0.0))
        E = base + pv.normalized() * h
        self.aim(up, E - S)
        self.aim(lo, T - E)
        return E, T

    def apply(self, prev=None):
        """Записывает позы P в pose bones; знак кватерниона непрерывен относительно prev."""
        out = {}
        for n in self.order:
            P = self.P[n]
            q = P.to_quaternion()
            if prev is not None and n in prev and q.dot(prev[n]) < 0.0:
                q.negate()
            out[n] = q
            pb = self.pb[n]
            pb.rotation_quaternion = q
            pb.location = P.translation if n == "Hips" else Vector((0.0, 0.0, 0.0))
        return out

    def check_fk(self):
        """Сверка прямой кинематики с Blender (armature-пространство)."""
        bpy.context.view_layer.update()
        err = 0.0
        for n in self.order:
            e = (self.pb[n].matrix.translation - self.F[n].translation).length
            err = max(err, e)
        return err


# ---------------------------------------------------------------- блок ног, рук, туловища

def shoulders(R, sh=0.0):
    R.aim("Shoulder.L", Vector((1.0, 0.0, 0.05 + sh)))
    R.aim("Shoulder.R", Vector((-1.0, 0.0, 0.05 + sh)))


def legs(R, ank_l, ank_r, foot_l, foot_r):
    for side, ank, fd in (("L", ank_l, foot_l), ("R", ank_r, foot_r)):
        up, lo = "UpperLeg." + side, "LowerLeg." + side
        S = R.M(up).translation
        R.ik(up, lo, ank, S + Vector((0.0, -1.0, 0.0)))  # колено вперёд
        R.aim("Foot." + side, fd)
        R.aim("Toes." + side, fd)


def torso(R, hips_head, hips_dir, spine_dir, chest_dir, chest_twist, neck_dir, head_dir):
    R.aim("Hips", hips_dir, head=hips_head)
    R.aim("Spine", spine_dir)
    R.aim("Chest", chest_dir, twist=chest_twist)
    R.aim("Neck", neck_dir)
    R.aim("Head", head_dir)


UP_Z = Vector((0.0, 0.0, 0.0))


# ---------------------------------------------------------------- позы

def pose_idle(R, f, dz):
    u = 2.0 * math.pi * f / IDLE_N
    s, s2 = math.sin(u), math.sin(2.0 * u)
    z = Vector((0.0, 0.0, dz))
    torso(
        R,
        hips_head=V(0.006 * s, 0.0, 0.897) + z,
        hips_dir=vdir(0.0, 0.8 * s),
        spine_dir=vdir(0.8 - 0.4 * s, 0.0),
        chest_dir=vdir(1.5 - 1.2 * s, 0.0),   # дыхание: грудь ±1.2 град
        chest_twist=0.0,
        neck_dir=vdir(2.0, 0.0),
        head_dir=vdir(-1.0 + 0.8 * s2, 2.0 * math.sin(u + 0.7)),
    )
    shoulders(R, 0.01 * s)
    S_l = R.M("UpperArm.L").translation
    S_r = R.M("UpperArm.R").translation
    legs(R, V(0.092, 0.0, ANKLE_REST_Z) + z, V(-0.092, 0.0, ANKLE_REST_Z) + z,
         foot_vec(0.0), foot_vec(0.0))
    # левая рука расслаблена
    R.ik("UpperArm.L", "LowerArm.L", V(0.47, -0.02, 0.98) + z, S_l + V(0.4, 0.5, -0.3))
    R.aim("Hand.L", Vector((0.15, -0.1, -1.0)))
    # правая рука: гард, шпага низко перед корпусом
    R.ik("UpperArm.R", "LowerArm.R", V(-0.20, -0.22 - 0.01 * s, 0.98) + z,
         S_r + V(-0.6, 0.3, -0.5))
    R.aim("Hand.R", Vector((0.0, -0.8, 0.35)))


def walk_foot_state(phi):
    if phi < WALK_STANCE:
        y = WALK_EXCURSION - 2 * WALK_EXCURSION * (phi / WALK_STANCE)
        lift = 0.0
    else:
        s = (phi - WALK_STANCE) / (1.0 - WALK_STANCE)
        y = -WALK_EXCURSION + 2 * WALK_EXCURSION * ease(s)
        lift = 0.06 * math.sin(math.pi * s)
    pitch = key(phi, [(0.0, 8.0), (0.15, 0.0), (WALK_STANCE, -18.0), (0.8, -6.0), (1.0, 8.0)])
    return y, lift, pitch


def pose_walk(R, f, dz):
    p = f / WALK_N
    tp = 2.0 * math.pi * p
    z = Vector((0.0, 0.0, dz))
    hz = 0.8725 - 0.0225 * math.cos(4.0 * math.pi * p)  # таз: низ на контакте (0.85), верх в середине опоры (0.895)
    torso(
        R,
        hips_head=V(0.0, 0.0, hz) + z,
        hips_dir=vdir(1.0 * math.cos(2.0 * tp), 2.0 * math.sin(tp)),
        spine_dir=vdir(3.0, -1.5 * math.sin(tp)),
        chest_dir=vdir(3.0, -1.0 * math.sin(tp)),
        chest_twist=4.0 * math.cos(tp),
        neck_dir=vdir(0.0, 0.0),
        head_dir=vdir(-2.0, 0.0),
    )
    shoulders(R, 0.0)
    yL, liftL, pL = walk_foot_state(p % 1.0)
    yR, liftR, pR = walk_foot_state((p + 0.5) % 1.0)
    legs(R,
         V(0.092, yL, ANKLE_REST_Z + liftL) + z,
         V(-0.092, yR, ANKLE_REST_Z + liftR) + z,
         foot_vec(pL), foot_vec(pR))
    # левая рука: противофаза правой ноге; правая рука держит шпагу прижато
    thL = -20.0 * math.cos(tp)
    R.aim("UpperArm.L", arm_dir(+1, thL, 0.25))
    R.aim("LowerArm.L", arm_dir(+1, thL + 15.0, 0.22))
    R.aim("Hand.L", arm_dir(+1, thL + 25.0, 0.12))
    S_r = R.M("UpperArm.R").translation
    wR = V(-0.20, -0.22 - 0.03 * math.cos(tp), 0.95) + z
    R.ik("UpperArm.R", "LowerArm.R", wR, S_r + V(-0.6, 0.3, -0.5))
    R.aim("Hand.R", Vector((0.0, -0.7, 0.3)))


def pose_attack(R, f, dz):
    z = Vector((0.0, 0.0, dz))
    hy = key(f, [(0, 0.0), (4, 0.03), (9, -0.14), (13, -0.14), (20, 0.0)])
    hz = key(f, [(0, 0.885), (4, 0.88), (9, 0.80), (13, 0.80), (20, 0.885)])
    lean = key(f, [(0, 0.0), (4, -3.0), (9, 8.0), (13, 8.0), (20, 0.0)])
    torso(
        R,
        hips_head=V(0.0, hy, hz) + z,
        hips_dir=vdir(lean, 0.0),
        spine_dir=vdir(key(f, [(0, 1.0), (4, -4.0), (9, 10.0), (13, 10.0), (20, 1.0)]), 0.0),
        chest_dir=vdir(key(f, [(0, 1.5), (4, -6.0), (9, 16.0), (13, 16.0), (20, 1.5)]), 0.0),
        chest_twist=key(f, [(0, 0.0), (4, -25.0), (9, 25.0), (13, 25.0), (20, 0.0)]),
        neck_dir=vdir(key(f, [(0, 0.0), (9, 6.0), (20, 0.0)]), 0.0),
        head_dir=vdir(key(f, [(0, -1.0), (4, 2.0), (9, -4.0), (20, -1.0)]), 0.0),
    )
    shoulders(R, 0.0)
    yR = key(f, [(0, 0.0), (4, 0.0), (9, -0.45), (13, -0.45), (20, 0.0)])
    liftR = key(f, [(0, 0.0), (4, 0.0), (7, 0.12), (9, 0.0), (13, 0.0), (16, 0.08), (20, 0.0)])
    yL = key(f, [(0, 0.0), (9, 0.2), (13, 0.2), (20, 0.0)])
    pL = key(f, [(0, 0.0), (9, -20.0), (13, -20.0), (20, 0.0)])
    legs(R,
         V(0.092, yL, ANKLE_REST_Z) + z,
         V(-0.092, yR, ANKLE_REST_Z + liftR) + z,
         foot_vec(pL), foot_vec(0.0))
    # левая рука уравновешивает выпад
    thL = key(f, [(0, 0.0), (4, 15.0), (9, -30.0), (13, -30.0), (20, 0.0)])
    R.aim("UpperArm.L", arm_dir(+1, thL, 0.25))
    R.aim("LowerArm.L", arm_dir(+1, thL + 15.0, 0.22))
    R.aim("Hand.L", arm_dir(+1, thL + 25.0, 0.12))
    # правая рука с шпагой: замах, выпад, стойка
    wR = key(f, [(0, V(-0.20, -0.22, 0.98)), (2, V(-0.42, -0.10, 1.00)), (4, V(-0.42, 0.06, 1.05)),
                 (9, V(-0.20, -0.50, 1.22)), (13, V(-0.20, -0.50, 1.22)),
                 (20, V(-0.20, -0.22, 0.98))]) + z
    dR = key(f, [(0, V(0.0, -0.8, 0.35)), (2, V(-0.1, 0.0, 1.0)), (4, V(0.0, 0.85, 0.45)),
                 (9, V(0.0, -1.0, 0.05)), (13, V(0.0, -1.0, 0.05)), (20, V(0.0, -0.8, 0.35))])
    S_r = R.M("UpperArm.R").translation
    R.ik("UpperArm.R", "LowerArm.R", wR, S_r + V(-0.5, 0.3, -0.6))
    R.aim("Hand.R", dR)


def pose_death(R, f, dz):
    z = Vector((0.0, 0.0, dz))
    lean = key(f, [(0, 0.0), (8, -12.0), (20, -40.0), (28, -75.0), (36, -88.0)])
    hy = key(f, [(0, 0.0), (8, 0.15), (20, 0.12), (28, 0.25), (36, 0.30)])
    hz = key(f, [(0, 0.885), (8, 0.86), (20, 0.50), (28, 0.30), (36, 0.20)])
    head_extra = key(f, [(0, 0.0), (6, -25.0), (12, -18.0), (20, -10.0), (36, 0.0)])
    torso(
        R,
        hips_head=V(0.0, hy, hz) + z,
        hips_dir=vdir(lean, 0.0),
        spine_dir=vdir(lean, 0.0),
        chest_dir=vdir(lean, 0.0),
        chest_twist=0.0,
        neck_dir=vdir(lean, 0.0),
        head_dir=vdir(lean + head_extra, 0.0),
    )
    shoulders(R, 0.0)
    ank_l = key(f, [(0, V(0.092, 0.0, ANKLE_REST_Z)), (8, V(0.092, 0.0, ANKLE_REST_Z)),
                    (20, V(0.092, 0.42, 0.06)), (28, V(0.092, 0.10, 0.06)),
                    (36, V(0.092, -0.45, 0.07))]) + z
    ank_r = key(f, [(0, V(-0.092, 0.0, ANKLE_REST_Z)), (8, V(-0.092, 0.0, ANKLE_REST_Z)),
                    (20, V(-0.092, 0.42, 0.06)), (28, V(-0.092, 0.10, 0.06)),
                    (36, V(-0.092, -0.45, 0.07))]) + z
    pitch = key(f, [(0, 0.0), (8, 0.0), (20, 180.0), (28, 180.0), (36, 0.0)])
    legs(R, ank_l, ank_r, foot_vec(pitch), foot_vec(pitch))
    wL = key(f, [(0, V(0.47, -0.02, 0.98)), (6, V(0.75, 0.0, 1.20)),
                 (14, V(0.40, 0.45, 0.30)), (36, V(0.42, 0.95, 0.12))]) + z
    wR = key(f, [(0, V(-0.20, -0.22, 0.98)), (6, V(-0.35, -0.05, 0.85)),
                 (14, V(-0.50, 0.20, 0.40)), (36, V(-0.45, 0.85, 0.16))]) + z
    S_l = R.M("UpperArm.L").translation
    S_r = R.M("UpperArm.R").translation
    R.ik("UpperArm.L", "LowerArm.L", wL, S_l + V(0.3, 0.3, 0.6))
    R.ik("UpperArm.R", "LowerArm.R", wR, S_r + V(-0.4, 0.3, -0.6))
    R.aim("Hand.L", key(f, [(0, Vector((0.15, -0.1, -1.0))), (14, Vector((0.3, 0.5, -0.5))),
                             (36, Vector((0.3, 1.0, 0.1)))]))
    R.aim("Hand.R", key(f, [(0, Vector((0.0, -0.8, 0.35))), (14, Vector((0.0, 0.9, -0.2))),
                             (36, Vector((0.0, 1.0, -0.05)))]))


# ---------------------------------------------------------------- измерение пола

def world_points(objs):
    """Мировые координаты вершин деформированных мешей (с учётом позы)."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    pts = []
    for ob in objs:
        oe = ob.evaluated_get(dg)
        me = oe.to_mesh()
        mw = oe.matrix_world
        pts.extend([mw @ v.co for v in me.vertices])
        oe.to_mesh_clear()
    return pts


def body_min_z(meshes):
    """Минимальная мировая Z вершин деформированных мешей тела (без шпаги)."""
    return min(p.z for p in world_points(meshes))


def sword_clearance(body_pts, sword_pts, grip):
    """Минимальное расстояние клинка до тела, исключая зону хвата (0.15 м от grip)."""
    from mathutils import kdtree
    kd = kdtree.KDTree(len(body_pts))
    for i, p in enumerate(body_pts):
        kd.insert(p, i)
    kd.balance()
    dmin = 9.0
    for sp in sword_pts:
        if (sp - grip).length < 0.15:
            continue
        _, _, d = kd.find(sp)
        if d < dmin:
            dmin = d
    return dmin


def build_action(name, n, pose_fn, R, arm, meshes, sword=None):
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    if arm.animation_data is None:
        arm.animation_data_create()
    arm.animation_data.action = act
    prev = None
    zs_before, zs_after, zs_sword, clear = [], [], [], []
    for f in range(n + 1):
        R.reset()
        pose_fn(R, f, 0.0)
        R.fk_all()
        R.apply()
        z0 = body_min_z(meshes)
        zs_before.append(z0)
        R.reset()
        pose_fn(R, f, -z0)
        R.fk_all()
        prev = R.apply(prev)
        pts = world_points(meshes)
        zs_after.append(min(p.z for p in pts))
        if sword is not None:
            spts = world_points([sword])
            zs_sword.append(min(p.z for p in spts))
            clear.append(sword_clearance(pts, spts, sword.matrix_world.translation.copy()))
        fr = f
        for bn in R.order:
            R.pb[bn].keyframe_insert(data_path="rotation_quaternion", frame=fr, group=bn)
        R.pb["Hips"].keyframe_insert(data_path="location", frame=fr, group="Hips")
    return zs_before, zs_after, zs_sword, clear


# ---------------------------------------------------------------- шпага

def attach_sword(arm, R):
    path = os.path.join(MODELS, "rapier.blend")
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.objects = [n for n in src.objects if n == "Rapier"]
    sw = bpy.data.objects["Rapier"]
    bpy.context.scene.collection.objects.link(sw)
    sw.parent = arm
    sw.parent_type = "BONE"
    sw.parent_bone = "WeaponSocket.R"
    bone = arm.data.bones["WeaponSocket.R"]
    # Blender: дочерние объекты кости привязаны к её хвосту (tail).
    # Хват (центр рапиры) ставится в ладонь: origin на head кости WeaponSocket.R.
    sw.matrix_parent_inverse = Matrix.Identity(4)
    loc = Vector((0.0, -bone.length, 0.0))
    sw.location = loc
    sw.rotation_mode = "QUATERNION"
    sw.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    sw.scale = (1.0, 1.0, 1.0)
    return sw


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


def render_strip(arm, action, frames, out_png, cam_pos, cam_target, scene, cam_obj):
    arm.animation_data.action = bpy.data.actions[action]
    direction = (Vector(cam_target) - Vector(cam_pos)).normalized()
    cam_obj.location = cam_pos
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    paths = []
    for i, f in enumerate(frames):
        scene.frame_set(f)
        p = os.path.join(bpy.app.tempdir, "prev_%s_%d_%d.png" % (action, int(cam_pos[0] * 10) + 50, i))
        scene.render.filepath = p
        bpy.ops.render.render(write_still=True)
        paths.append(p)
    stitch(paths, out_png)


# ---------------------------------------------------------------- GLB

def glb_animations(path):
    data = open(path, "rb").read()
    clen, ctype = struct.unpack("<I4s", data[12:20])
    doc = json.loads(data[20:20 + clen].decode("utf-8"))
    res = []
    for a in doc.get("animations", []):
        tmax = 0.0
        for smp in a["samplers"]:
            acc = doc["accessors"][smp["input"]]
            tmax = max(tmax, acc["max"][0])
        res.append((a["name"], round(tmax, 4), len(a["channels"])))
    return res, len(doc.get("skins", [])), len(doc.get("meshes", []))


# ---------------------------------------------------------------- main

def main():
    bpy.ops.wm.open_mainfile(filepath=os.path.join(MODELS, "hero.blend"))
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.frame_start = 0
    scene.frame_end = IDLE_N

    arm = bpy.data.objects["Armature"]
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    R = Rig(arm)

    print("FK check error (m):", round(R.check_fk(), 9))
    z_rest = body_min_z(meshes)
    print("rest min Z (m):", round(z_rest, 5))

    R.reset()
    sw = attach_sword(arm, R)
    bpy.context.view_layer.update()

    report = {}
    for name, n, fn in (("Idle", IDLE_N, pose_idle), ("Walk", WALK_N, pose_walk),
                        ("Attack", ATTACK_N, pose_attack), ("Death", DEATH_N, pose_death)):
        zb, za, zs, cl = build_action(name, n, fn, R, arm, meshes, sword=sw)
        report[name] = {"frames": n, "body_minZ_after_max_abs": round(max(abs(z) for z in za), 5),
                        "body_shift_range": round(max(zb) - min(zb), 4),
                        "sword_minZ_min": round(min(zs), 4),
                        "sword_body_clearance_min": round(min(cl), 4),
                        "sword_body_clearance_at_frame": int(cl.index(min(cl)))}
        print("ACTION %-7s frames=%d body_minZ_after_max_abs=%.5f shift_range=%.4f "
              "sword_minZ_min=%.4f sword_body_clearance_min=%.4f at_frame=%d"
              % (name, n, max(abs(z) for z in za), max(zb) - min(zb), min(zs),
                 min(cl), cl.index(min(cl))))

    arm.animation_data.action = bpy.data.actions["Idle"]

    # JSON крепления шпаги (в rest-позе)
    bone = arm.data.bones["WeaponSocket.R"]
    head_w = arm.matrix_world @ bone.head_local
    tail_w = arm.matrix_world @ bone.tail_local
    attach = {
        "object": "Rapier",
        "source": "models/rapier.blend",
        "parent_type": "BONE",
        "parent_bone": "WeaponSocket.R",
        "origin": "bone tail (Blender bone parenting); location below is relative to it",
        "location": [round(c, 6) for c in sw.location],
        "rotation_quaternion": [round(c, 6) for c in sw.rotation_quaternion],
        "rotation_euler_xyz_deg": [round(math.degrees(c), 3) for c in sw.rotation_euler],
        "scale": [1.0, 1.0, 1.0],
        "grip_at": "palm: grip centre (rapier origin) at WeaponSocket.R head",
        "blade_axis": "rapier +Y == bone +Y (hand direction)",
        "rapier_length_m": 1.145,
        "bone_head_rest_armature_m": [round(c, 5) for c in head_w],
        "bone_tail_rest_armature_m": [round(c, 5) for c in tail_w],
        "armature_world_matrix_identity": True,
    }
    with open(os.path.join(MODELS, "rapier_attach.json"), "w", encoding="utf-8") as fh:
        json.dump(attach, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print("ATTACH", json.dumps(attach))

    # GLB без шпаги: выделяем только героиню
    bpy.ops.object.select_all(action="DESELECT")
    for ob in bpy.data.objects:
        if ob is arm or (ob.type == "MESH" and ob.parent is arm and ob.name != "Rapier"):
            ob.select_set(True)
    bpy.context.view_layer.objects.active = arm
    glb_path = os.path.join(MODELS, "hero_anim.glb")
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_frame_range=False,
        export_force_sampling=True,
        export_skins=True,
        export_yup=True,
    )
    anims, nskin, nmesh = glb_animations(glb_path)
    print("GLB animations:", anims, "skins:", nskin, "meshes:", nmesh)
    bpy.ops.object.select_all(action="DESELECT")

    # Превью
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False
    scene.render.resolution_x = 320
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    world = bpy.data.worlds.new("PrevWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.9, 0.9, 0.88, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    scene.world = world
    sun_data = bpy.data.lights.new("PrevSun", "SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("PrevSun", sun_data)
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(-30))
    scene.collection.objects.link(sun)
    fill_data = bpy.data.lights.new("PrevFill", "AREA")
    fill_data.energy = 200.0
    fill_data.size = 3.0
    fill = bpy.data.objects.new("PrevFill", fill_data)
    fill.location = (-2.5, -3.0, 2.5)
    fill.rotation_euler = (math.radians(55), 0.0, math.radians(-40))
    scene.collection.objects.link(fill)
    cam_data = bpy.data.cameras.new("PrevCam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 2.7
    cam = bpy.data.objects.new("PrevCam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    target34 = (0.0, 0.3, 0.9)
    cam34 = (-2.2, -2.8, 1.5)
    camside = (3.2, 0.3, 1.0)
    os.makedirs(DOCS, exist_ok=True)
    render_strip(arm, "Idle", [0, 8, 16, 24, 32, 40], os.path.join(DOCS, "anim_idle.png"),
                 cam34, target34, scene, cam)
    render_strip(arm, "Walk", [0, 4, 8, 12, 16, 20], os.path.join(DOCS, "anim_walk.png"),
                 cam34, target34, scene, cam)
    render_strip(arm, "Walk", [0, 4, 8, 12, 16, 20], os.path.join(DOCS, "anim_walk_side.png"),
                 camside, (0.0, 0.3, 0.9), scene, cam)
    render_strip(arm, "Attack", [0, 4, 6, 9, 13, 20], os.path.join(DOCS, "anim_attack.png"),
                 cam34, target34, scene, cam)
    render_strip(arm, "Death", [0, 8, 14, 20, 28, 36], os.path.join(DOCS, "anim_death.png"),
                 cam34, (0.0, 0.3, 0.3), scene, cam)
    for ob in (cam, sun, fill):
        bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.cameras.remove(cam_data)
    bpy.data.lights.remove(sun_data)
    bpy.data.lights.remove(fill_data)
    scene.world = None
    bpy.data.worlds.remove(world)

    arm.animation_data.action = bpy.data.actions["Idle"]
    scene.frame_set(0)
    out_blend = os.path.join(MODELS, "hero_anim.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print("SAVED", out_blend)
    print("REPORT", json.dumps(report))


if __name__ == "__main__":
    main()
