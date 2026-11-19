extends RefCounted
## Таблица направлений героини. Код героя читает ТОЛЬКО эти константы.
## DIR_TO_CELL: направление -> индекс кадра assets/hero/cell_<N>.png (0..7)
## DIR_FLIP: направление -> flip_h (пока пусто)
## Значения пока приблизительные, оркестратор уточнит после просмотра кадров.

const DIR_TO_CELL = {
	"E": 1,
	"SE": 0,
	"S": 7,
	"SW": 6,
	"W": 5,
	"NW": 4,
	"N": 3,
	"NE": 2,
}

const DIR_FLIP = {}

## Порядок секторов по часовой стрелке в экранных координатах (y вниз),
## начиная с E (угол 0), шаг 45 градусов.
const DIR_NAMES = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]


static func dir_name_from_vector(v: Vector2) -> String:
	var sector: int = posmod(int(round(v.angle() / (PI / 4.0))), 8)
	return DIR_NAMES[sector]
