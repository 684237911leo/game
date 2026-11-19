#!/usr/bin/env python3
"""Нарезка 4 спрайт-листов героини на отдельные кадры.

Запуск из корня проекта:  python3 -I tools/slice_anims.py

Для каждого листа (assets/source/*.png):
  1. режет на ячейки 112x144 (ряд r, колонка f);
  2. чистит ячейку: альфа < 200 -> 0, иначе 255; оставляет крупнейшую
     8-связную компоненту и компоненты > 8% её площади; снимает светлую
     кайму (граничные пиксели, яркость > 200 и низкая насыщенность) за два
     прохода, если фигура не распадается;
  3. кладёт кадры на единый холст листа без масштабирования: по горизонтали
     центр тела (центр масс в нижних 40% bbox), подошвы на строке H-5;
  4. пишет assets/hero/anim/<anim>/r<R>_f<F>.png и _meta.json;
  5. пишет контрольные _contact.png и _rows.png;
  6. печатает сводку и аномалии.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "source"
OUT = ROOT / "assets" / "hero" / "anim"

CELL_W, CELL_H = 112, 144
ALPHA_T = 200          # альфа-порог: ниже -> 0, иначе 255
ISLAND_FRAC = 0.08     # компоненты > 8% площади крупнейшей остаются
LIGHT_MIN = 200        # порог яркости (среднее RGB) для светлой каймы
SAT_MAX = 0.2          # низкая насыщенность: (max-min) <= 0.2*max
BODY_FRAC = 0.4        # центр тела - центр масс в нижних 40% bbox
PAD = 8                # холст = max bbox + 8 (по 4 px с каждой стороны)
FEET_GAP = 4           # отступ подошв от нижнего края холста
LOSS_WARN = 0.03       # предупреждение, если очистка съела > 3% пикселей

SHEETS = [
    ("walk", "walk_5dirs_7frames.png", 5, 7),
    ("attack", "attack_5dirs_8frames.png", 5, 8),
    ("idle", "idle_3dirs_8frames.png", 3, 8),
    ("death", "death_1dir_8frames.png", 1, 8),
]

NB8 = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def components8(mask):
    """Список связных компонент (списки (y, x)) по 8-связности."""
    h, w = mask.shape
    m = mask.tolist()
    seen = [[False] * w for _ in range(h)]
    comps = []
    for y in range(h):
        for x in range(w):
            if not m[y][x] or seen[y][x]:
                continue
            seen[y][x] = True
            stack = [(y, x)]
            pts = []
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for dy, dx in NB8:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and m[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            comps.append(pts)
    return comps


def keep_main(mask):
    """Оставляет крупнейшую компоненту и компоненты > 8% её площади.
    Возвращает (маска, отсортированные размеры оставленных компонент)."""
    comps = components8(mask)
    out = np.zeros_like(mask)
    if not comps:
        return out, []
    big = max(len(c) for c in comps)
    kept = [c for c in comps if len(c) == big or len(c) > ISLAND_FRAC * big]
    for c in kept:
        ys = [p[0] for p in c]
        xs = [p[1] for p in c]
        out[ys, xs] = True
    return out, sorted((len(c) for c in kept), reverse=True)


def fringe_pass(m, rgb):
    """Один проход: граничные светлые ненасыщенные пиксели -> прозрачность."""
    h, w = m.shape
    p = np.pad(m, 1, constant_values=False)
    all_nb = np.ones((h, w), dtype=bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if (dy, dx) != (1, 1):
                all_nb &= p[dy:dy + h, dx:dx + w]
    boundary = m & ~all_nb                       # есть прозрачный сосед
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    mean = rgb.mean(axis=2)
    light = (mean > LIGHT_MIN) & ((mx - mn) <= SAT_MAX * mx)
    return m & ~(boundary & light)


def fringe_step(m1, rgb, n_before):
    """Светлая кайма: два прохода. Если фигура распалась на большее число
    компонент, шаг отменяется. Возвращает (маска, отменён?)."""
    m = m1
    for _ in range(2):
        m = fringe_pass(m, rgb)
    m2, sizes = keep_main(m)
    if len(sizes) > n_before:
        return m1, True
    return m2, False


def load_font(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def process_sheet(anim, fn, rows, cols, notes):
    arr = np.array(Image.open(SRC / fn).convert("RGBA"))
    H, W = arr.shape[:2]
    if (W, H) != (cols * CELL_W, rows * CELL_H):
        raise SystemExit(f"{fn}: размер {W}x{H} не совпадает с сеткой "
                         f"{cols}x{rows} по {CELL_W}x{CELL_H}")

    cells = {}
    for r in range(rows):
        for f in range(cols):
            tag = f"r{r}f{f}"
            cell = arr[r * CELL_H:(r + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            rgb8 = cell[..., :3].copy()
            cand = cell[..., 3] >= ALPHA_T
            n_before = int(cand.sum())
            if n_before == 0:
                raise SystemExit(f"{anim} {tag}: пустая ячейка")

            m1, sizes1 = keep_main(cand)
            m2, reverted = fringe_step(m1, rgb8.astype(np.float32), len(sizes1))
            if reverted:
                notes["fringe_reverted"].append(tag)
                m_final, sizes = m1, sizes1
            else:
                m_final, sizes = m2, keep_main(m2)[1]
            n_after = int(m_final.sum())

            if len(sizes) > 1:
                notes["multi"].append(f"{tag} (компоненты {sizes})")
            loss = 1 - n_after / n_before
            if loss > LOSS_WARN:
                notes["loss"].append(f"{tag} (-{loss * 100:.1f}%)")

            ys, xs = np.nonzero(m_final)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            if x0 == 0 or y0 == 0 or x1 == CELL_W - 1 or y1 == CELL_H - 1:
                notes["edge"].append(f"{tag} bbox=({x0},{y0})-({x1},{y1})")
            bh = y1 - y0 + 1
            sel = ys >= y0 + (1 - BODY_FRAC) * bh
            cx = float(xs[sel].mean())
            cells[(r, f)] = dict(mask=m_final, rgb=rgb8,
                                 bbox=(x0, y0, x1, y1), cx=cx)

    bws = [c["bbox"][2] - c["bbox"][0] + 1 for c in cells.values()]
    bhs = [c["bbox"][3] - c["bbox"][1] + 1 for c in cells.values()]
    CW = max(bws) + PAD
    CH = max(bhs) + PAD
    ref_h = int(np.median(bhs))
    feet_y = CH - 1 - FEET_GAP

    placed = {}
    for (r, f), c in cells.items():
        x0, y0, x1, y1 = c["bbox"]
        dx = round((CW - 1) / 2 - c["cx"])
        lo, hi = -x0, CW - 1 - x1
        if dx < lo or dx > hi:
            notes["clamped"].append(f"r{r}f{f}")
            dx = min(max(dx, lo), hi)
        dy = feet_y - y1
        ys, xs = np.nonzero(c["mask"])
        out = np.zeros((CH, CW, 4), dtype=np.uint8)
        out[ys + dy, xs + dx, :3] = c["rgb"][ys, xs]
        out[ys + dy, xs + dx, 3] = 255
        placed[(r, f)] = out

    anim_dir = OUT / anim
    anim_dir.mkdir(parents=True, exist_ok=True)
    for (r, f), out in placed.items():
        Image.fromarray(out, "RGBA").save(anim_dir / f"r{r}_f{f}.png")
    meta = {"rows": rows, "frames": cols, "canvas": [CW, CH],
            "ref_height": ref_h, "feet_y": feet_y}
    (anim_dir / "_meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                         encoding="utf-8")

    write_contact(anim_dir / "_contact.png", placed, rows, cols, CW, CH, feet_y)
    write_rows(anim_dir / "_rows.png", placed, rows, CW, CH, feet_y)
    return meta, len(placed)


def paint_red_line(bg, x, y, width, feet_y_abs):
    ImageDraw.Draw(bg).line([(x, feet_y_abs), (x + width - 1, feet_y_abs)],
                            fill=(255, 0, 0, 255))


def write_contact(path, placed, rows, cols, CW, CH, feet_y):
    GAP, LABEL_H, S = 4, 12, 2
    cell_w, cell_h = CW + GAP, CH + LABEL_H + GAP
    W1, H1 = GAP + cols * cell_w, GAP + rows * cell_h
    bg = Image.new("RGBA", (W1, H1), (32, 32, 32, 255))
    labels = []
    for r in range(rows):
        for f in range(cols):
            x = GAP + f * cell_w
            top = GAP + r * cell_h
            y = top + LABEL_H
            bg.alpha_composite(Image.fromarray(placed[(r, f)], "RGBA"), (x, y))
            paint_red_line(bg, x, y + feet_y, CW, y + feet_y)
            labels.append((x, top, f"r{r}f{f}"))
    big = bg.resize((W1 * S, H1 * S), Image.NEAREST)
    draw = ImageDraw.Draw(big)
    font = load_font(12 * S)
    for x, top, text in labels:
        draw.text((x * S + 2, top * S + 1), text, font=font,
                  fill=(210, 210, 210, 255))
    big.convert("RGB").save(path)


def write_rows(path, placed, rows, CW, CH, feet_y):
    GAP, LABEL_W, S = 4, 44, 2
    W1 = GAP + LABEL_W + CW + GAP
    H1 = GAP + rows * (CH + GAP)
    bg = Image.new("RGBA", (W1, H1), (32, 32, 32, 255))
    labels = []
    for r in range(rows):
        y = GAP + r * (CH + GAP)
        x = GAP + LABEL_W
        bg.alpha_composite(Image.fromarray(placed[(r, 0)], "RGBA"), (x, y))
        paint_red_line(bg, x, y + feet_y, CW, y + feet_y)
        labels.append((GAP, y + CH // 2, f"row {r}"))
    big = bg.resize((W1 * S, H1 * S), Image.NEAREST)
    draw = ImageDraw.Draw(big)
    font = load_font(12 * S)
    for x, y, text in labels:
        draw.text((x * S, (y - 6) * S), text, font=font,
                  fill=(210, 210, 210, 255))
    big.convert("RGB").save(path)


def main():
    all_ok = True
    for anim, fn, rows, cols in SHEETS:
        notes = {"multi": [], "edge": [], "loss": [], "clamped": [],
                 "fringe_reverted": []}
        meta, n = process_sheet(anim, fn, rows, cols, notes)
        print(f"[{anim}] {fn}: rows={rows} frames={cols} "
              f"canvas={meta['canvas'][0]}x{meta['canvas'][1]} "
              f"ref_height={meta['ref_height']} feet_y={meta['feet_y']} "
              f"кадров={n} (ожидалось {rows * cols})")
        if n != rows * cols:
            all_ok = False
        labels = [
            ("оставлено > 1 компоненты", notes["multi"]),
            ("bbox касается края ячейки (обрезан)", notes["edge"]),
            ("очистка съела > 3% пикселей", notes["loss"]),
            ("светлая кайма отменена (фигура распалась)", notes["fringe_reverted"]),
            ("центр тела вылез за холст, сдвиг ограничен", notes["clamped"]),
        ]
        for title, items in labels:
            if items:
                print(f"  аномалия - {title}: {', '.join(items)}")
        if not any(notes.values()):
            print("  аномалий нет")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
