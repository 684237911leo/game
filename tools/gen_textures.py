#!/usr/bin/env python3
"""Генератор бесшовных текстур земли для Diablo Slice.

Запуск из корня проекта:
    python3 -I tools/gen_textures.py

Пишет в assets/tex/ (512x512 PNG):
    ground_dirt.png   тёмная потрескавшаяся земля с мелкими камешками
    ground_grass.png  мёртвая блёклая трава (вытянутые штрихи)
    ground_stone.png  потёртые каменные плиты с тёмными швами
    ground_macro.png  крупный серый шум для разбивки повторов тайлинга

Все текстуры периодические (wrap): шум строится на торе, Voronoi-расстояния
считаются с учётом обхода краёв, поэтому швов при тайлинге нет.
Палитра мрачная, контраст низкий. Sidecar .import с mipmaps пишет этот же скрипт.
"""
import os

import numpy as np
from PIL import Image

SIZE = 512
SEED = 20260704
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "assets", "tex"))

IMPORT_TEMPLATE = """[remap]

importer="texture"
type="CompressedTexture2D"
path="res://.godot/imported/{name}-{hash}.ctex"
metadata={{
"vram_texture": false
}}

[deps]

source_file="res://assets/tex/{name}"
dest_files=["res://.godot/imported/{name}-{hash}.ctex"]

[params]

compress/mode=0
compress/high_quality=false
compress/lossy_quality=0.7
compress/hdr_compression=1
compress/normal_map=0
compress/channel_pack=0
mipmaps/generate=true
mipmaps/limit=-1
roughness/mode=0
roughness/src_normal=""
process/fix_alpha_border=true
process/premult_alpha=false
process/normal_map_invert_y=false
process/hdr_as_srgb=false
process/hdr_clamp_exposure=false
process/size_limit=0
detect_3d/compress_to=1
"""


def smooth(t):
    return t * t * (3.0 - 2.0 * t)


def value_noise(rng, px, py):
    """Периодический value noise: решётка px x py, по краям заворачивается."""
    lat = rng.random((py, px))
    ys = np.arange(SIZE) * py / SIZE
    xs = np.arange(SIZE) * px / SIZE
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = smooth(ys - y0)
    fx = smooth(xs - x0)
    y1 = (y0 + 1) % py
    x1 = (x0 + 1) % px
    y0 %= py
    x0 %= px
    a = lat[np.ix_(y0, x0)]
    b = lat[np.ix_(y0, x1)]
    c = lat[np.ix_(y1, x0)]
    d = lat[np.ix_(y1, x1)]
    top = a + (b - a) * fx[None, :]
    bot = c + (d - c) * fx[None, :]
    return top + (bot - top) * fy[:, None]


def fbm(rng, px, py, octaves, persistence=0.5):
    """Сумма октав value noise; возвращает значения примерно в [0, 1]."""
    total = np.zeros((SIZE, SIZE))
    amp = 1.0
    norm = 0.0
    for _ in range(octaves):
        total += amp * value_noise(rng, px, py)
        norm += amp
        amp *= persistence
        px *= 2
        py *= 2
    return total / norm


def normalize(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / max(hi - lo, 1e-6)


def voronoi(rng, n_points):
    """Периодический Voronoi: возвращает (F1, F2, метка ближайшей точки, точки)."""
    px = rng.random(n_points) * SIZE
    py = rng.random(n_points) * SIZE
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    f1 = np.full((SIZE, SIZE), 1e9)
    f2 = np.full((SIZE, SIZE), 1e9)
    label = np.zeros((SIZE, SIZE), dtype=np.int64)
    for i in range(n_points):
        dx = np.abs(xx - px[i])
        dx = np.minimum(dx, SIZE - dx)
        dy = np.abs(yy - py[i])
        dy = np.minimum(dy, SIZE - dy)
        d = np.sqrt(dx * dx + dy * dy)
        closer = d < f1
        f2 = np.where(closer, f1, np.minimum(f2, d))
        label = np.where(closer, i, label)
        f1 = np.where(closer, d, f1)
    return f1, f2, label, px, py


def mix(a, b, t):
    return a + (b - a) * t[..., None]


def hex_color(c):
    return np.array(c, dtype=np.float64) / 255.0


def save(name, rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    img = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    path = os.path.join(OUT_DIR, name)
    img.save(path, optimize=True)
    print("wrote", path)


def write_import_sidecar(name):
    """Sidecar с mipmaps: без них фильтр в дали даёт муар на земле."""
    import hashlib

    h = hashlib.md5(name.encode()).hexdigest()
    path = os.path.join(OUT_DIR, name + ".import")
    with open(path, "w", encoding="utf-8") as f:
        f.write(IMPORT_TEMPLATE.format(name=name, hash=h))


def ground_dirt(rng):
    # Базовая тёмная земля: крупные пятна и мелкая зернистость.
    blotch = normalize(fbm(rng, 4, 4, 5))
    grain = fbm(rng, 128, 128, 2) - 0.5
    t = np.clip(0.25 + 0.6 * blotch + 0.08 * grain, 0.0, 1.0)
    base = mix(hex_color((48, 41, 31)), hex_color((86, 74, 58)), t)
    # Трещины: края Voronoi, искажённые шумом, чтобы не выглядели сеткой.
    f1, f2, _, _, _ = voronoi(rng, 42)
    warp = fbm(rng, 8, 8, 3) - 0.5
    edge = (f2 - f1) + 7.0 * warp
    crack = np.exp(-(edge / 0.7) ** 2)
    crack *= np.clip((fbm(rng, 16, 16, 3) - 0.35) * 2.5, 0.0, 1.0)
    base = mix(base, hex_color((30, 25, 19)), np.clip(crack * 0.6, 0.0, 1.0))
    # Мелкие камешки: светлее и серее, без ярких пятен.
    f1p, _, _, _, _ = voronoi(rng, 260)
    pebble = np.clip(1.0 - f1p / 1.6, 0.0, 1.0) ** 2
    base = mix(base, hex_color((104, 98, 88)), pebble * 0.55)
    # Высушенные пятна, немного светлее.
    dry = np.clip((fbm(rng, 16, 16, 3) - 0.55) * 3.0, 0.0, 1.0)
    base = mix(base, hex_color((98, 84, 62)), dry * 0.25)
    return base


def ground_grass(rng):
    # Вытянутые штрихи: период по y больше, чем по x -> вертикальные «соломины».
    streak = normalize(fbm(rng, 6, 80, 3))
    streak = np.clip((streak - 0.3) * 1.6, 0.0, 1.0)
    fine = fbm(rng, 96, 96, 2)
    patch = normalize(fbm(rng, 3, 3, 4))
    t = np.clip(0.35 * streak + 0.25 * fine + 0.4 * patch, 0.0, 1.0)
    # Блёклый жёлто-серо-зелёный, не насыщенный.
    pale = mix(hex_color((92, 84, 56)), hex_color((116, 106, 72)), t)
    dead = mix(pale, hex_color((90, 80, 56)), np.clip((patch - 0.5) * 1.2, 0.0, 0.35))
    # Тёмная подстилка между травинками.
    litter = np.clip((fbm(rng, 24, 24, 3) - 0.6) * 2.0, 0.0, 1.0)
    out = mix(dead, hex_color((46, 42, 30)), litter * 0.6)
    return out


def ground_stone(rng):
    # Плиты: Voronoi-клетки, у каждой свой оттенок серо-коричневого камня.
    f1, f2, label, px, py = voronoi(rng, 26)
    tint = rng.random(len(px))
    tint_map = tint[label]
    stone_lo = hex_color((64, 57, 48))
    stone_hi = hex_color((90, 83, 72))
    slab = mix(stone_lo, stone_hi, 0.3 + 0.7 * tint_map)
    # Шов: тонкая тёмная линия на границе клеток, размытая шумом.
    warp = fbm(rng, 10, 10, 2) - 0.5
    seam = np.exp(-((f2 - f1 + 2.5 * warp) / 1.6) ** 2)
    slab = mix(slab, hex_color((34, 32, 29)), np.clip(seam * 0.6, 0.0, 1.0))
    # Потёртость: мелкий шум и светлые пятна.
    wear = fbm(rng, 64, 64, 3) - 0.5
    slab = slab * (1.0 + 0.22 * wear)[..., None]
    # Царапины: длинные тонкие штрихи.
    scratch = fbm(rng, 4, 128, 2)
    scratch = np.clip((scratch - 0.62) * 4.0, 0.0, 1.0)
    slab = mix(slab, hex_color((100, 96, 88)), scratch * 0.25)
    # Щебень в швах и мелкие крошки.
    f1c, _, _, _, _ = voronoi(rng, 180)
    crumb = np.clip(1.0 - f1c / 1.3, 0.0, 1.0) ** 2
    slab = mix(slab, hex_color((82, 78, 72)), crumb * 0.4)
    return slab


def macro_noise(rng):
    # Крупные пятна: используются шейдером для разбивки повторов тайлинга.
    m = normalize(fbm(rng, 2, 2, 4))
    return np.stack([m, m, m], axis=-1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gens = [
        ("ground_dirt.png", ground_dirt),
        ("ground_grass.png", ground_grass),
        ("ground_stone.png", ground_stone),
        ("ground_macro.png", macro_noise),
    ]
    for k, (name, fn) in enumerate(gens):
        rng = np.random.default_rng(SEED + k)
        save(name, fn(rng))
        write_import_sidecar(name)


if __name__ == "__main__":
    main()
