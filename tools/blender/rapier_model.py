"""Процедурная модель пиратской шпаги (rapier) для ARPG.

Запуск из корня проекта:  blender -b -P tools/blender/rapier_model.py
Выход:
  models/rapier.blend
  models/rapier.glb
  docs/models/rapier_a.png  (вид сбоку)
  docs/models/rapier_b.png  (3/4, крупно эфес)
  docs/models/rapier_c.png  (сверху)

Соглашения:
  единицы - метры, Blender Z-up, масштаб объекта 1, transforms применены (геометрия в координатах).
  начало координат - центр рукояти (хват). Клинок вдоль +Y.
  Плоскость клинка - XY: ширина клинка по X, толщина по Z (плоские грани смотрят вверх/вниз по Z),
  гарда (перекрестие, корзина) лежит в плоскости XZ-ориентации по X и Z - читается сверху.
  Материалы: слот 0 Blade, 1 Gold, 2 Leather.
"""
import math
import os

import bmesh
import bpy
from mathutils import Vector

ROOT = "/home/user/game"
MODELS_DIR = os.path.join(ROOT, "models")
DOCS_DIR = os.path.join(ROOT, "docs", "models")
BLEND_PATH = os.path.join(MODELS_DIR, "rapier.blend")
GLB_PATH = os.path.join(MODELS_DIR, "rapier.glb")

BLADE, GOLD, LEATHER = 0, 1, 2

bpy.ops.wm.read_factory_settings(use_empty=True)


# ---------------------------------------------------------------- геометрия

class Mesh:
    """Накопитель вершин/граней с индексом материала и проверкой объёма по частям."""

    def __init__(self):
        self.v = []
        self.f = []
        self.m = []
        self.report = []

    def add(self, name, verts, faces, mat):
        base = len(self.v)
        self.v.extend(verts)
        vol = 0.0
        for face in faces:
            self.f.append([base + i for i in face])
            self.m.append(mat)
            a = Vector(verts[face[0]])
            for k in range(1, len(face) - 1):
                b = Vector(verts[face[k]])
                c = Vector(verts[face[k + 1]])
                vol += a.dot(b.cross(c)) / 6.0
        self.report.append((name, vol))


def sweep(path, scales, shape, ref, closed=False, caps=(True, True), apex=None):
    """Протяжка 2D-профиля вдоль ломаной.

    path   - точки оси; scales - (su, sv) на каждой точке; shape - единичный профиль (u, v), CCW.
    ref    - опорный вектор для фрейма: N = ref x T, B = T x N (N x B = T, грани наружу).
    Грани (i,j),(i,j+1),(i+1,j+1),(i+1,j) дают внешнюю нормаль при таком фрейме.
    """
    ref = Vector(ref)
    n = len(path)
    k = len(shape)
    pts = [Vector(p) for p in path]
    verts = []
    ring = []
    for i in range(n):
        if closed:
            t = pts[(i + 1) % n] - pts[(i - 1) % n]
        else:
            t = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        t.normalize()
        nrm = ref.cross(t).normalized()
        bi = t.cross(nrm)
        su, sv = scales[i]
        ring.append(len(verts))
        for u, v in shape:
            p = pts[i] + nrm * (u * su) + bi * (v * sv)
            verts.append((p.x, p.y, p.z))
    faces = []
    for i in range(n if closed else n - 1):
        b0 = ring[i]
        b1 = ring[(i + 1) % n]
        for j in range(k):
            j2 = (j + 1) % k
            faces.append([b0 + j, b0 + j2, b1 + j2, b1 + j])
    if not closed:
        if caps[0]:
            c = len(verts)
            verts.append(tuple(pts[0]))
            b = ring[0]
            for j in range(k):
                faces.append([c, b + (j + 1) % k, b + j])
        last = ring[-1]
        if apex is not None:
            a = len(verts)
            verts.append(tuple(apex))
            for j in range(k):
                faces.append([last + j, last + (j + 1) % k, a])
        elif caps[1]:
            c = len(verts)
            verts.append(tuple(pts[-1]))
            for j in range(k):
                faces.append([c, last + j, last + (j + 1) % k])
    return verts, faces


def circle(k):
    return [(math.cos(2 * math.pi * j / k), math.sin(2 * math.pi * j / k)) for j in range(k)]


DIAMOND = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # ромб: кромки по +-X, ребра по +-Z


def sphere(mesh, name, center, radius, nlat, nlon, mat):
    """Сфера как протяжка окружности вдоль оси Y (полюса - малый радиус, без крышек)."""
    cx, cy, cz = center
    path, scales = [], []
    for i in range(nlat + 1):
        th = math.pi * (nlat - i) / nlat  # от низа к верху: y возрастает
        r = max(radius * math.sin(th), 1e-5)
        path.append((cx, cy + radius * math.cos(th), cz))
        scales.append((r, r))
    verts, faces = sweep(path, scales, circle(nlon), ref=(1, 0, 0), caps=(True, True))
    mesh.add(name, verts, faces, mat)


def build_model():
    m = Mesh()

    # Клинок: y 0.080 -> 1.030 (0.95 м), сужение + острие; у основания прячется в перекрестии.
    y0, y1 = 0.080, 1.030
    length = y1 - y0
    ns = 48
    path, scales = [], []
    for i in range(ns + 1):
        s = 0.99 * i / ns
        f = math.sqrt(max(0.0, 1.0 - ((s - 0.82) / 0.18) ** 2)) if s > 0.82 else 1.0
        w = 0.030 * (1.0 - 0.45 * s) * f
        th = 0.0085 * (1.0 - 0.30 * s) * f
        path.append((0.0, y0 + length * s, 0.0))
        scales.append((w / 2, th / 2))
    verts, faces = sweep(path, scales, DIAMOND, ref=(0, 0, -1), caps=(True, False),
                         apex=(0.0, y1, 0.0))
    m.add("blade", verts, faces, BLADE)

    # Перекрестие-дуга (quillon): от -0.095 до +0.095 по X, слегка выгнута к клинку.
    xs = [-0.095 + 0.19 * i / 16 for i in range(17)]
    path = [(x, 0.080 + 0.004 * (1 - (x / 0.095) ** 2), 0.0) for x in xs]
    verts, faces = sweep(path, [(0.0085, 0.0085)] * len(path), circle(8), ref=(0, 0, 1))
    m.add("quillon", verts, faces, GOLD)

    # Корзина (basket): замкнутая эллиптическая дуга вокруг хвата, концы совпадают с перекрестием.
    na = 48
    path = []
    for i in range(na):
        a = 2 * math.pi * i / na
        path.append((0.095 * math.cos(a), 0.030 + 0.050 * math.cos(a) ** 2, 0.070 * math.sin(a)))
    verts, faces = sweep(path, [(0.0075, 0.0075)] * na, circle(6), ref=(0, 1, 0), closed=True)
    m.add("basket", verts, faces, GOLD)

    # Навершия на концах перекрестия.
    for sx in (-1, 1):
        sphere(m, "finial", (sx * 0.095, 0.080, 0.0), 0.0105, 6, 8, GOLD)

    # Простые завитки: спираль в плоскости XY от каждого навершия наружу.
    for sgn in (-1, 1):
        r0 = 0.020
        cxp = sgn * (0.095 + r0)
        th0 = math.pi if sgn > 0 else 0.0
        path = []
        for i in range(31):
            t = i / 30
            th = th0 + sgn * 2.5 * math.pi * t
            r = r0 * (1 - 0.65 * t)
            path.append((cxp + r * math.cos(th), 0.080 + r * math.sin(th), 0.0))
        verts, faces = sweep(path, [(0.0032, 0.0032)] * len(path), circle(4), ref=(0, 0, 1))
        m.add("curl", verts, faces, GOLD)

    # Эфес-кольцо на хвате под перекрестием (замкнутый тор вокруг оси Y).
    nr = 12
    path = [(0.020 * math.cos(2 * math.pi * i / nr), 0.058, 0.020 * math.sin(2 * math.pi * i / nr))
            for i in range(nr)]
    verts, faces = sweep(path, [(0.0045, 0.0045)] * nr, circle(6), ref=(0, 1, 0), closed=True)
    m.add("finger_ring", verts, faces, GOLD)

    # Рукоять: y -0.06..0.06, обмотка - чередование радиуса даёт рёбра кожаной обмотки.
    ng = 23
    path, scales = [], []
    for i in range(ng):
        y = -0.06 + 0.12 * i / (ng - 1)
        r = 0.0135 if i % 2 == 0 else 0.0148
        path.append((0.0, y, 0.0))
        scales.append((r, r))
    verts, faces = sweep(path, scales, circle(16), ref=(1, 0, 0))
    m.add("grip", verts, faces, LEATHER)

    # Манжета у основания рукояти и навершие-шар.
    verts, faces = sweep([(0.0, -0.064, 0.0), (0.0, -0.054, 0.0)], [(0.0175, 0.0175)] * 2,
                         circle(12), ref=(1, 0, 0))
    m.add("collar", verts, faces, GOLD)
    sphere(m, "pommel", (0.0, -0.089, 0.0), 0.026, 8, 12, GOLD)
    return m


def make_materials():
    def mat(name, color, metallic, roughness):
        md = bpy.data.materials.new(name)
        md.use_nodes = True
        bsdf = md.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
        return md

    return [
        mat("Blade", (0.78, 0.80, 0.84), 1.0, 0.25),
        mat("Gold", (1.00, 0.71, 0.29), 1.0, 0.35),
        mat("Leather", (0.11, 0.055, 0.03), 0.0, 0.80),
    ]


def make_object(m, materials):
    me = bpy.data.meshes.new("Rapier")
    me.from_pydata(m.v, [], m.f)
    me.update()
    for md in materials:
        me.materials.append(md)
    for poly, mi in zip(me.polygons, m.m):
        poly.material_index = mi
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.update()
    obj = bpy.data.objects.new("Rapier", me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def check_mesh(obj, m):
    me = obj.data
    tris = sum(len(p.vertices) - 2 for p in me.polygons)
    bm = bmesh.new()
    bm.from_mesh(me)
    boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    nonmanifold = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    bm.free()
    print("=== ПРОВЕРКА ===")
    print("треугольников:", tris, "| вершин:", len(me.vertices))
    print("граничных рёбер (дыры):", boundary, "| неманифолдных рёбер:", nonmanifold)
    print("Dimensions (м):", tuple(round(c, 4) for c in obj.dimensions))
    print("Знак объёма по частям (ожидается > 0 у замкнутых):")
    for name, vol in m.report:
        flag = "" if vol > 0 else "  <-- ВНИМАНИЕ: внутрь"
        print("  %-12s %+.3e%s" % (name, vol, flag))
    return tris


# ---------------------------------------------------------------- сцена и рендер

def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def add_light(name, loc, target, energy, color, size):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = energy
    ld.color = color
    ld.size = size
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    bpy.context.scene.collection.objects.link(ob)
    look_at(ob, target)


def add_camera(loc, target, lens=50.0):
    cd = bpy.data.cameras.new("Cam")
    cd.lens = lens
    cd.sensor_width = 36.0
    cam = bpy.data.objects.new("Cam", cd)
    cam.location = loc
    bpy.context.scene.collection.objects.link(cam)
    look_at(cam, target)
    bpy.context.scene.camera = cam


def clear_extra_objects():
    for ob in list(bpy.data.objects):
        if ob.name != "Rapier":
            bpy.data.objects.remove(ob, do_unlink=True)
    for ld in list(bpy.data.lights):
        if ld.users == 0:
            bpy.data.lights.remove(ld)
    for cd in list(bpy.data.cameras):
        if cd.users == 0:
            bpy.data.cameras.remove(cd)


def setup_render():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 32
    sc.cycles.use_denoising = False
    sc.render.resolution_x = 800
    sc.render.resolution_y = 600
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.film_transparent = False
    world = bpy.data.worlds.new("Backdrop")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.09, 0.09, 0.10, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    sc.world = world


def render_shot(filename, cam_loc, target, lens=50.0):
    clear_extra_objects()
    tgt = Vector(target)
    # тёплый key спереди-слева-сверху, холодный rim сзади-справа
    add_light("key_warm", tgt + Vector((0.9, -1.2, 1.4)), tgt, 120.0,
              (1.0, 0.82, 0.62), 1.2)
    add_light("rim_cool", tgt + Vector((-1.2, 1.0, 0.8)), tgt, 90.0,
              (0.62, 0.78, 1.0), 1.0)
    add_camera(cam_loc, target, lens)
    sc = bpy.context.scene
    sc.render.filepath = os.path.join(DOCS_DIR, filename)
    bpy.ops.render.render(write_still=True)
    print("рендер:", sc.render.filepath)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    m = build_model()
    materials = make_materials()
    obj = make_object(m, materials)
    check_mesh(obj, m)

    bpy.context.preferences.filepaths.save_version = 0  # без .blend1
    bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH, check_existing=False)
    print("blend:", BLEND_PATH)

    bpy.ops.export_scene.gltf(
        filepath=GLB_PATH,
        export_format="GLB",
        export_yup=True,
        export_animations=False,
        use_selection=False,
    )
    print("glb:", GLB_PATH)

    setup_render()
    # a: сбоку целиком; b: 3/4 крупно эфес; c: сверху
    render_shot("rapier_a.png", cam_loc=(1.7, 0.45, 0.95), target=(0.0, 0.45, 0.0))
    render_shot("rapier_b.png", cam_loc=(0.28, 0.28, 0.26), target=(0.0, 0.06, 0.0))
    render_shot("rapier_c.png", cam_loc=(0.0, 0.45, 2.4), target=(0.0, 0.45, 0.0))


if __name__ == "__main__":
    main()
