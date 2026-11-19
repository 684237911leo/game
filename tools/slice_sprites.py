#!/usr/bin/env python3
"""Нарезка спрайт-листа героини на 8 очищенных кадров одинакового размера.

Запуск из корня проекта:  python3 -I tools/slice_sprites.py

Вход:  assets/source/hero_sheet.png (RGBA, сетка 4x2, ячейка 384x512). Не изменяется.
Выход: assets/hero/cell_0..7.png и assets/hero/_contact.png
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SHEET_PATH = ROOT / "assets" / "source" / "hero_sheet.png"
OUT_DIR = ROOT / "assets" / "hero"

COLS, ROWS = 4, 2
CELL_W, CELL_H = 384, 512
ALPHA_THRESHOLD = 200       # alpha < 200 -> 0, иначе -> 255
MIN_AREA_RATIO = 0.05       # компоненты крупнее 5% от самой большой тоже оставляем
PAD = 8                     # отступ холста/подошв
CANVAS_MARGIN = 16          # холст = max bbox + 16 по каждой оси
GREY = (128, 128, 128)
RED = (255, 0, 0)
LABEL_H = 20


def components_4(mask):
    """Связные компоненты (4-связность) булевой маски, BFS на чистом Python.

    Возвращает список компонент, каждая — список плоских индексов пикселей.
    """
    h, w = mask.shape
    flat = mask.ravel().tolist()
    seen = bytearray(h * w)
    comps = []
    for start in range(h * w):
        if not flat[start] or seen[start]:
            continue
        seen[start] = 1
        queue = deque([start])
        comp = []
        while queue:
            idx = queue.popleft()
            comp.append(idx)
            y, x = divmod(idx, w)
            if x > 0 and flat[idx - 1] and not seen[idx - 1]:
                seen[idx - 1] = 1
                queue.append(idx - 1)
            if x < w - 1 and flat[idx + 1] and not seen[idx + 1]:
                seen[idx + 1] = 1
                queue.append(idx + 1)
            if y > 0 and flat[idx - w] and not seen[idx - w]:
                seen[idx - w] = 1
                queue.append(idx - w)
            if y < h - 1 and flat[idx + w] and not seen[idx + w]:
                seen[idx + w] = 1
                queue.append(idx + w)
        comps.append(comp)
    return comps


def clean_cell(cell_rgba):
    """Убирает полупрозрачное свечение. Возвращает (очищенный RGBA, bbox, метаданные)."""
    h, w = cell_rgba.shape[:2]
    alpha = cell_rgba[:, :, 3]
    opaque = alpha >= ALPHA_THRESHOLD
    opaque_count = int(opaque.sum())

    comps = components_4(opaque)
    if not comps:
        raise RuntimeError("в ячейке нет непрозрачных пикселей")
    largest = max(len(c) for c in comps)
    kept = [c for c in comps if len(c) == largest or len(c) > MIN_AREA_RATIO * largest]

    keep_flat = np.zeros(h * w, dtype=bool)
    for c in kept:
        keep_flat[c] = True
    keep = keep_flat.reshape(h, w)

    out = np.zeros_like(cell_rgba)
    out[keep, :3] = cell_rgba[keep, :3]
    out[keep, 3] = 255

    ys, xs = np.where(keep)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    meta = {
        "opaque_before": opaque_count,
        "kept_px": int(keep.sum()),
        "kept_pct": 100.0 * int(keep.sum()) / opaque_count,
        "components_total": len(comps),
        "components_kept": len(kept),
    }
    return out, bbox, meta


def main():
    sheet = np.array(Image.open(SHEET_PATH).convert("RGBA"))
    sheet_h, sheet_w = sheet.shape[:2]
    if (sheet_w, sheet_h) != (COLS * CELL_W, ROWS * CELL_H):
        raise SystemExit(f"неожиданный размер листа: {sheet_w}x{sheet_h}")

    cells, bboxes, metas = [], [], []
    for i in range(COLS * ROWS):
        row, col = divmod(i, COLS)
        cell = sheet[row * CELL_H:(row + 1) * CELL_H, col * CELL_W:(col + 1) * CELL_W]
        out, bbox, meta = clean_cell(cell)
        cells.append(out)
        bboxes.append(bbox)
        metas.append(meta)

    bw = [b[2] - b[0] for b in bboxes]
    bh = [b[3] - b[1] for b in bboxes]
    canvas_w = max(bw) + CANVAS_MARGIN
    canvas_h = max(bh) + CANVAS_MARGIN

    frames = []
    for i, (out, bbox, w_, h_) in enumerate(zip(cells, bboxes, bw, bh)):
        x0, y0, x1, y1 = bbox
        figure = out[y0:y1, x0:x1]
        frame = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        left = (canvas_w - w_) // 2
        top = canvas_h - PAD - h_          # подошвы на одной линии: низ bbox = canvas_h - PAD
        frame[top:top + h_, left:left + w_] = figure
        # Для прозрачных пикселей обнуляем RGB, чтобы файлы были чистыми.
        frame[frame[:, :, 3] == 0] = 0
        frames.append(frame)
        Image.fromarray(frame, "RGBA").save(OUT_DIR / f"cell_{i}.png")

    # Контрольный лист: 4x2 на сером фоне, подписи индексов, красная линия подошв.
    cell_w = canvas_w + 2 * PAD
    cell_h = canvas_h + 2 * PAD + LABEL_H
    sheet_img = Image.new("RGB", (COLS * cell_w, ROWS * cell_h), GREY)
    draw = ImageDraw.Draw(sheet_img)
    for i, frame in enumerate(frames):
        row, col = divmod(i, COLS)
        ox = col * cell_w + PAD
        oy = row * cell_h + PAD
        sheet_img.paste(Image.fromarray(frame, "RGBA"), (ox, oy),
                        Image.fromarray(frame, "RGBA"))
        feet_y = oy + canvas_h - PAD - 1   # последний ряд пикселей подошв
        draw.line([(ox, feet_y), (ox + canvas_w - 1, feet_y)], fill=RED, width=1)
        draw.text((ox, oy + canvas_h + 4), f"#{i}", fill=(255, 255, 255))
    sheet_img.save(OUT_DIR / "_contact.png")

    print(f"Холст: {canvas_w}x{canvas_h} (max bbox {max(bw)}x{max(bh)} + {CANVAS_MARGIN})")
    print(f"Подошвы: отступ {PAD} px от низа холста")
    print("idx | bbox до нормализации (x0,y0,x1,y1) | размер   | оставшиеся непрозр. px | компонент (всего/оставлено)")
    for i, (bbox, m) in enumerate(zip(bboxes, metas)):
        print(f"{i:>3} | {str(bbox):<33} | {bw[i]}x{bh[i]:<5} | "
              f"{m['kept_px']}/{m['opaque_before']} = {m['kept_pct']:.1f}% | "
              f"{m['components_total']}/{m['components_kept']}")


if __name__ == "__main__":
    sys.exit(main())
