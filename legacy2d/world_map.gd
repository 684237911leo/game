extends Node2D
## Изометрическая карта: земля рисуется кодом, препятствия — отдельными нодами
## (они добавляются в родителя через build_props, чтобы сортироваться по y вместе с героем).

const GRID := 28
const TILE_W := 128.0
const TILE_H := 64.0
const HALF_W := TILE_W * 0.5
const HALF_H := TILE_H * 0.5
const RNG_SEED := 1337
const START_CELL := Vector2i(14, 14)
const SCATTER_COUNT := 25
const RUIN_ROW := 18
const RUIN_GAP_X := 13
const RUIN_XS := [10, 11, 12, 14, 15, 16]

const KIND_STONE := 0
const KIND_TREE := 1
const KIND_GRAVE := 2
const KIND_COLUMN := 3

const GROUND_COLORS := [
	Color(0.20, 0.19, 0.12),
	Color(0.15, 0.20, 0.12),
	Color(0.24, 0.20, 0.13),
]
const EDGE_COLOR := Color(0.07, 0.06, 0.04, 0.7)

var _kinds := {}
var _astar: AStarGrid2D = null


func _ready() -> void:
	_build_obstacles()
	_build_astar()
	queue_redraw()


## Центр ромба клетки (i, j) в мировых координатах.
func grid_to_world(c: Vector2i) -> Vector2:
	return Vector2((c.x - c.y) * HALF_W, (c.x + c.y) * HALF_H)


## Обратное преобразование: мировая точка -> клетка (округление в координатах сетки).
func world_to_grid(p: Vector2) -> Vector2i:
	var u: float = (p.x / HALF_W + p.y / HALF_H) * 0.5
	var v: float = (p.y / HALF_H - p.x / HALF_W) * 0.5
	return Vector2i(roundi(u), roundi(v))


## Ограничивающий прямоугольник всей карты в мировых координатах.
func get_bounds() -> Rect2:
	var pts := PackedVector2Array()
	for uv in [Vector2(-0.5, -0.5), Vector2(GRID - 0.5, -0.5), Vector2(-0.5, GRID - 0.5), Vector2(GRID - 0.5, GRID - 0.5)]:
		pts.append(Vector2((uv.x - uv.y) * HALF_W, (uv.x + uv.y) * HALF_H))
	var r := Rect2(pts[0], Vector2.ZERO)
	for p in pts:
		r = r.expand(p)
	return r


## Путь между клетками в мировых точках (центры клеток). Пустой массив, если пути нет.
func find_path(from_cell: Vector2i, to_cell: Vector2i) -> PackedVector2Array:
	var out := PackedVector2Array()
	var a := nearest_free_cell(from_cell)
	var b := nearest_free_cell(to_cell)
	if a.x < 0 or b.x < 0:
		return out
	var ids = _astar.get_id_path(a, b)
	for c in ids:
		out.append(grid_to_world(c))
	return out


## Ближайшая свободная клетка к c (сам c, если он свободен и в сетке).
func nearest_free_cell(c: Vector2i) -> Vector2i:
	var cc := _clamp_cell(c)
	if _is_free(cc):
		return cc
	var best := Vector2i(-1, -1)
	var best_d := 1.0e18
	for j in GRID:
		for i in GRID:
			var p := Vector2i(i, j)
			if _astar.is_point_solid(p):
				continue
			var d: float = float((p - cc).length_squared())
			if d < best_d:
				best_d = d
				best = p
	return best


## Создаёт ноды препятствий как детей parent (должен быть y_sort_enabled).
func build_props(parent: Node) -> void:
	for key in _kinds.keys():
		var c: Vector2i = key
		var kind: int = _kinds[c]
		var n := Node2D.new()
		n.name = "Prop_%d_%d" % [c.x, c.y]
		n.position = grid_to_world(c)
		parent.add_child(n)
		n.draw.connect(_on_prop_draw.bind(n, kind, c))


func _draw() -> void:
	for j in GRID:
		for i in GRID:
			var ctr := grid_to_world(Vector2i(i, j))
			var pts := PackedVector2Array([
				ctr + Vector2(0.0, -HALF_H),
				ctr + Vector2(HALF_W, 0.0),
				ctr + Vector2(0.0, HALF_H),
				ctr + Vector2(-HALF_W, 0.0),
			])
			var h: int = ((i * 73856093) ^ (j * 19349663)) & 0x7fffffff
			var col: Color = GROUND_COLORS[h % GROUND_COLORS.size()]
			draw_colored_polygon(pts, col)
			var closed := pts.duplicate()
			closed.append(pts[0])
			draw_polyline(closed, EDGE_COLOR, 1.0)


func _build_obstacles() -> void:
	_kinds.clear()
	var rng := RandomNumberGenerator.new()
	rng.seed = RNG_SEED
	# Рамка по краю карты.
	for j in GRID:
		for i in GRID:
			if i == 0 or j == 0 or i == GRID - 1 or j == GRID - 1:
				_kinds[Vector2i(i, j)] = rng.randi_range(KIND_STONE, KIND_TREE)
	# Случайный разброс, стартовая зона 5x5 свободна.
	var placed := 0
	var guard := 0
	while placed < SCATTER_COUNT and guard < 2000:
		guard += 1
		var c := Vector2i(rng.randi_range(1, GRID - 2), rng.randi_range(1, GRID - 2))
		if _in_start_zone(c) or _kinds.has(c):
			continue
		_kinds[c] = rng.randi_range(KIND_STONE, KIND_GRAVE)
		placed += 1
	# Стена-руина: 6 клеток в ряд с проходом посередине.
	for x in RUIN_XS:
		_kinds[Vector2i(x, RUIN_ROW)] = KIND_COLUMN
	_kinds.erase(Vector2i(RUIN_GAP_X, RUIN_ROW))


func _build_astar() -> void:
	_astar = AStarGrid2D.new()
	_astar.region = Rect2i(0, 0, GRID, GRID)
	_astar.cell_size = Vector2(1, 1)
	_astar.diagonal_mode = AStarGrid2D.DIAGONAL_MODE_ONLY_IF_NO_OBSTACLES
	_astar.default_compute_heuristic = AStarGrid2D.HEURISTIC_OCTILE
	_astar.default_estimate_heuristic = AStarGrid2D.HEURISTIC_OCTILE
	_astar.update()
	for key in _kinds.keys():
		var c: Vector2i = key
		_astar.set_point_solid(c, true)


func _in_start_zone(c: Vector2i) -> bool:
	return absi(c.x - START_CELL.x) <= 2 and absi(c.y - START_CELL.y) <= 2


func _clamp_cell(c: Vector2i) -> Vector2i:
	return Vector2i(clampi(c.x, 0, GRID - 1), clampi(c.y, 0, GRID - 1))


func _is_free(c: Vector2i) -> bool:
	return c.x >= 0 and c.y >= 0 and c.x < GRID and c.y < GRID and not _astar.is_point_solid(c)


func _on_prop_draw(node: Node2D, kind: int, cell: Vector2i) -> void:
	var v: int = (cell.x * 31 + cell.y * 17) % 2
	node.draw_colored_polygon(_ellipse(0.0, 4.0, 26.0, 9.0), Color(0.0, 0.0, 0.0, 0.35))
	match kind:
		KIND_STONE:
			_draw_stone(node, v)
		KIND_TREE:
			_draw_tree(node, v)
		KIND_GRAVE:
			_draw_grave(node, v)
		_:
			_draw_column(node, v)


func _ellipse(cx: float, cy: float, rx: float, ry: float) -> PackedVector2Array:
	var pts := PackedVector2Array()
	for k in 20:
		var a := TAU * float(k) / 20.0
		pts.append(Vector2(cx + cos(a) * rx, cy + sin(a) * ry))
	return pts


func _draw_stone(n: Node2D, v: int) -> void:
	var s := 1.0 if v == 0 else 0.75
	var off := Vector2(0.0, 0.0) if v == 0 else Vector2(3.0, 2.0)
	var pts := PackedVector2Array([
		Vector2(-22, 2), Vector2(-16, -12), Vector2(-4, -20), Vector2(12, -18),
		Vector2(22, -6), Vector2(16, 4), Vector2(0, 8), Vector2(-14, 6),
	])
	for k in pts.size():
		pts[k] = pts[k] * s + off
	n.draw_colored_polygon(pts, Color(0.40, 0.40, 0.42))
	var closed := pts.duplicate()
	closed.append(pts[0])
	n.draw_polyline(closed, Color(0.17, 0.17, 0.19), 1.5)


func _draw_tree(n: Node2D, v: int) -> void:
	var h := 64.0 if v == 0 else 52.0
	var bark := Color(0.20, 0.15, 0.10)
	n.draw_colored_polygon(PackedVector2Array([Vector2(-6, 2), Vector2(6, 2), Vector2(2.5, -h), Vector2(-2.5, -h)]), bark)
	n.draw_line(Vector2(0, -h * 0.55), Vector2(-24, -h * 0.55 - 16), bark, 3.0)
	n.draw_line(Vector2(0, -h * 0.40), Vector2(22, -h * 0.40 - 14), bark, 3.0)
	n.draw_line(Vector2(0, -h), Vector2(-10, -h - 16), bark, 2.5)
	n.draw_line(Vector2(0, -h), Vector2(9, -h - 20), bark, 2.5)
	n.draw_line(Vector2(-24, -h * 0.55 - 16), Vector2(-30, -h * 0.55 - 24), bark, 1.5)


func _draw_grave(n: Node2D, v: int) -> void:
	var stone := Color(0.34, 0.34, 0.32)
	var edge := Color(0.14, 0.14, 0.13)
	n.draw_colored_polygon(_ellipse(0.0, 1.0, 17.0, 5.0), Color(0.22, 0.20, 0.14))
	if v == 0:
		var pts := PackedVector2Array([
			Vector2(-11, 2), Vector2(-11, -24), Vector2(-7, -32), Vector2(0, -36),
			Vector2(7, -32), Vector2(11, -24), Vector2(11, 2),
		])
		n.draw_colored_polygon(pts, stone)
		var closed := pts.duplicate()
		closed.append(pts[0])
		n.draw_polyline(closed, edge, 1.5)
	else:
		# Крест на могиле.
		n.draw_line(Vector2(0, 2), Vector2(0, -34), stone, 5.0)
		n.draw_line(Vector2(-11, -24), Vector2(11, -24), stone, 5.0)


func _draw_column(n: Node2D, v: int) -> void:
	var h := 70.0 if v == 0 else 46.0
	var dark := Color(0.30, 0.28, 0.25)
	var shaft := Color(0.47, 0.44, 0.39)
	n.draw_colored_polygon(PackedVector2Array([Vector2(-14, 0), Vector2(14, 0), Vector2(14, -6), Vector2(-14, -6)]), dark)
	n.draw_colored_polygon(PackedVector2Array([Vector2(-9, -6), Vector2(9, -6), Vector2(8, -h), Vector2(-8, -h)]), shaft)
	if v == 0:
		n.draw_colored_polygon(PackedVector2Array([Vector2(-12, -h), Vector2(12, -h), Vector2(12, -h - 7), Vector2(-12, -h - 7)]), dark)
	else:
		# Сломанный верх.
		n.draw_colored_polygon(PackedVector2Array([Vector2(-8, -h), Vector2(-3, -h - 6), Vector2(3, -h - 2), Vector2(8, -h)]), shaft)
