extends RefCounted
## Параметры анимаций героини. Всё зашито константами: assets/hero/anim/<anim>/_meta.json
## в игре не читается (json не попадает в экспорт APK).
## Таблицы направлений: направление -> [ряд, flip_h]. Правьте здесь.

## Масштаб walk (и база для остальных анимаций).
const SCALE_WALK := 2.0
## Рост героини в walk; остальные анимации масштабируются под него (кроме death).
const REF_HEIGHT_WALK := 84.0
const FRAME_PATH := "res://assets/hero/anim/%s/r%d_f%d.png"

## rows/frames — сетка кадров; canvas — размер холста (справочно);
## ref_height — высота тела в кадре; feet_y — строка подошв (origin героя = feet_y + 1);
## fps/loop — проигрывание; anchor: "feet" (origin = подошвы) или "bottom" (origin = низ холста);
## scale_to_ref: true -> масштаб SCALE_WALK * REF_HEIGHT_WALK / ref_height, false -> SCALE_WALK.
const ANIMS := {
	"idle": {
		"rows": 3, "frames": 8, "canvas": Vector2i(71, 123), "ref_height": 113.0, "feet_y": 118,
		"fps": 6.0, "loop": true, "anchor": "feet", "scale_to_ref": true,
	},
	"walk": {
		"rows": 5, "frames": 7, "canvas": Vector2i(70, 106), "ref_height": 84.0, "feet_y": 101,
		"fps": 10.0, "loop": true, "anchor": "feet", "scale_to_ref": true,
	},
	"attack": {
		"rows": 5, "frames": 8, "canvas": Vector2i(81, 96), "ref_height": 83.0, "feet_y": 91,
		"fps": 16.0, "loop": false, "anchor": "feet", "scale_to_ref": true,
	},
	"death": {
		"rows": 1, "frames": 8, "canvas": Vector2i(94, 89), "ref_height": 74.0, "feet_y": 84,
		"fps": 8.0, "loop": false, "anchor": "bottom", "scale_to_ref": false,
	},
}

## Ряды walk/attack: 0=спереди S, 1=вперёд-вправо SE, 2=вправо E, 3=назад-вправо NE, 4=спиной N.
const DIR_WALK := {
	"S": [0, false], "SE": [1, false], "E": [2, false], "NE": [3, false],
	"N": [4, false], "NW": [3, true], "W": [2, true], "SW": [1, true],
}
const DIR_ATTACK := {
	"S": [0, false], "SE": [1, false], "E": [2, false], "NE": [3, false],
	"N": [4, false], "NW": [3, true], "W": [2, true], "SW": [1, true],
}
const DIR_IDLE := {
	"S": [0, false], "SE": [1, false], "E": [1, false], "NE": [2, false],
	"N": [2, false], "NW": [2, true], "W": [1, true], "SW": [1, true],
}
const DIR_DEATH := {
	"S": [0, false], "SE": [0, false], "E": [0, false], "NE": [0, false],
	"N": [0, false], "NW": [0, true], "W": [0, true], "SW": [0, true],
}

const DIR_TABLES := {
	"idle": DIR_IDLE,
	"walk": DIR_WALK,
	"attack": DIR_ATTACK,
	"death": DIR_DEATH,
}


static func frame_path(anim: String, row: int, frame: int) -> String:
	return FRAME_PATH % [anim, row, frame]


## Масштаб спрайта анимации.
static func scale_for(anim: String) -> float:
	var info: Dictionary = ANIMS[anim]
	if bool(info["scale_to_ref"]):
		return SCALE_WALK * REF_HEIGHT_WALK / float(info["ref_height"])
	return SCALE_WALK


## Offset спрайта (centered = false), чтобы origin героя совпал с подошвами
## (anchor "feet") или с низом холста (anchor "bottom").
static func origin_offset(anim: String, tex: Texture2D) -> Vector2:
	var info: Dictionary = ANIMS[anim]
	var w := float(tex.get_width())
	var h := float(tex.get_height())
	if String(info["anchor"]) == "bottom":
		return Vector2(-w * 0.5, -h)
	return Vector2(-w * 0.5, -float(int(info["feet_y"]) + 1))
