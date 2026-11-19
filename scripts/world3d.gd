extends Node3D
## Карта 36x36 клеток по 1 м в плоскости XZ (y = 0), центр карты в начале координат.
## Земля: один ArrayMesh с цветами вершин. Препятствия: MultiMesh из примитивов (без коллайдеров).
## Навигация: AStarGrid2D по клеткам, 8 направлений. Маркер цели тоже здесь.

const N := 36
const CELL := 1.0
const HALF := N * CELL * 0.5
const RNG_SEED := 1337
const START_CELL := Vector2i(18, 18)
## Центральная зона 7x7 (|dx|, |dy| <= 3 от START_CELL) остаётся свободной.
const CENTER_ZONE := 3
const SCATTER_COUNT := 35
const RUIN_ROW := 9
## 8 колонн в ряд с проходом на x = 12.
const RUIN_XS := [8, 9, 10, 11, 13, 14, 15, 16]
## Пара надгробий.
const GRAVE_CELLS := [Vector2i(26, 25), Vector2i(27, 25)]

const KIND_STONE := 0
const KIND_TREE := 1
const KIND_GRAVE := 2
const KIND_COLUMN := 3

const GROUND_COLORS := [
	Color(0.36, 0.33, 0.21),
	Color(0.24, 0.34, 0.20),
	Color(0.42, 0.34, 0.22),
]
const MARKER_Y := 0.03
const MARKER_INNER := 0.34
const MARKER_OUTER := 0.46
const MARKER_PULSE_RATE := 6.0
const MARKER_PULSE_AMP := 0.1

var _kinds := {}
var _astar: AStarGrid2D = null
var _mats := {}
var _marker: MeshInstance3D = null
var _time := 0.0


func _ready() -> void:
	_build_obstacles()
	_build_astar()
	_build_ground()
	_build_props()
	_build_marker()


func _process(delta: float) -> void:
	if _marker != null and _marker.visible:
		_time += delta
		var s := 1.0 + MARKER_PULSE_AMP * sin(_time * MARKER_PULSE_RATE)
		_marker.scale = Vector3(s, 1.0, s)


## Центр клетки (c.x, c.y) в мировых координатах (y = 0).
func grid_to_world(c: Vector2i) -> Vector3:
	return Vector3((c.x + 0.5 - N * 0.5) * CELL, 0.0, (c.y + 0.5 - N * 0.5) * CELL)


## Клетка, в которую попадает мировая точка (без проверки проходимости).
func world_to_grid(p: Vector3) -> Vector2i:
	return Vector2i(floori(p.x / CELL + N * 0.5), floori(p.z / CELL + N * 0.5))


## Прямоугольник карты на плоскости XZ (x, y у Rect2 = x, z мира).
func get_ground_rect() -> Rect2:
	return Rect2(-HALF, -HALF, HALF * 2.0, HALF * 2.0)


func is_blocked(c: Vector2i) -> bool:
	return not _inside(c) or _astar.is_point_solid(c)


## Ближайшая свободная клетка к c (сам c, если свободен). Возвращает (-1, -1), если свободных нет.
func nearest_free_cell(c: Vector2i) -> Vector2i:
	var cc := Vector2i(clampi(c.x, 0, N - 1), clampi(c.y, 0, N - 1))
	if not is_blocked(cc):
		return cc
	var best := Vector2i(-1, -1)
	var best_d := 1.0e18
	for j in N:
		for i in N:
			var p := Vector2i(i, j)
			if _astar.is_point_solid(p):
				continue
			var d := float((p - cc).length_squared())
			if d < best_d:
				best_d = d
				best = p
	return best


## Путь от мировой точки до мировой точки: центры клеток. Пустой массив, если пути нет.
func find_path(from: Vector3, to: Vector3) -> PackedVector3Array:
	var out := PackedVector3Array()
	var a := nearest_free_cell(world_to_grid(from))
	var b := nearest_free_cell(world_to_grid(to))
	if a.x < 0 or b.x < 0:
		return out
	for c in _astar.get_id_path(a, b):
		out.append(grid_to_world(c))
	return out


func show_marker(p: Vector3) -> void:
	if _marker == null:
		return
	_marker.position = Vector3(p.x, MARKER_Y, p.z)
	_marker.scale = Vector3.ONE
	_marker.visible = true


func hide_marker() -> void:
	if _marker != null:
		_marker.visible = false


func _inside(c: Vector2i) -> bool:
	return c.x >= 0 and c.y >= 0 and c.x < N and c.y < N


func _hash(c: Vector2i) -> int:
	return ((c.x * 73856093) ^ (c.y * 19349663)) & 0x7fffffff


# ------------------------------------------------------------------ препятствия

func _build_obstacles() -> void:
	_kinds.clear()
	var rng := RandomNumberGenerator.new()
	rng.seed = RNG_SEED
	# Рамка по краю карты: камни и деревья.
	for j in N:
		for i in N:
			if i == 0 or j == 0 or i == N - 1 or j == N - 1:
				_kinds[Vector2i(i, j)] = rng.randi_range(KIND_STONE, KIND_TREE)
	# Руина: колонны в ряд с проходом.
	for x in RUIN_XS:
		_kinds[Vector2i(x, RUIN_ROW)] = KIND_COLUMN
	# Пара надгробий.
	for g in GRAVE_CELLS:
		_kinds[g] = KIND_GRAVE
	# Случайный разброс, центральная зона свободна.
	var placed := 0
	var guard := 0
	while placed < SCATTER_COUNT and guard < 5000:
		guard += 1
		var c := Vector2i(rng.randi_range(1, N - 2), rng.randi_range(1, N - 2))
		if _in_center_zone(c) or _kinds.has(c):
			continue
		_kinds[c] = rng.randi_range(KIND_STONE, KIND_GRAVE)
		placed += 1


func _in_center_zone(c: Vector2i) -> bool:
	return absi(c.x - START_CELL.x) <= CENTER_ZONE and absi(c.y - START_CELL.y) <= CENTER_ZONE


func _build_astar() -> void:
	_astar = AStarGrid2D.new()
	_astar.region = Rect2i(0, 0, N, N)
	_astar.cell_size = Vector2(CELL, CELL)
	_astar.diagonal_mode = AStarGrid2D.DIAGONAL_MODE_ONLY_IF_NO_OBSTACLES
	_astar.default_compute_heuristic = AStarGrid2D.HEURISTIC_OCTILE
	_astar.default_estimate_heuristic = AStarGrid2D.HEURISTIC_OCTILE
	_astar.update()
	for key in _kinds.keys():
		_astar.set_point_solid(key, true)


# ------------------------------------------------------------------ земля

func _build_ground() -> void:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for j in N:
		for i in N:
			var col: Color = GROUND_COLORS[_hash(Vector2i(i, j)) % GROUND_COLORS.size()]
			var x0 := (i - N * 0.5) * CELL
			var x1 := x0 + CELL
			var z0 := (j - N * 0.5) * CELL
			var z1 := z0 + CELL
			var a := Vector3(x0, 0.0, z0)
			var b := Vector3(x1, 0.0, z0)
			var c := Vector3(x1, 0.0, z1)
			var d := Vector3(x0, 0.0, z1)
			for v in [a, c, b, a, d, c]:
				st.set_color(col)
				st.set_normal(Vector3.UP)
				st.add_vertex(v)
	var mat := StandardMaterial3D.new()
	mat.vertex_color_use_as_albedo = true
	mat.roughness = 1.0
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	var mi := MeshInstance3D.new()
	mi.name = "Ground"
	mi.mesh = st.commit()
	mi.material_override = mat
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(mi)


# ------------------------------------------------------------------ объекты (MultiMesh)

func _build_props() -> void:
	var groups := {}
	for key in _kinds.keys():
		var c: Vector2i = key
		var kind: int = _kinds[c]
		var v := ((c.x * 31) + (c.y * 17)) % 2
		var gk := "%d_%d" % [kind, v]
		if not groups.has(gk):
			groups[gk] = {"kind": kind, "variant": v, "xforms": []}
		var rot := float((c.x * 73 + c.y * 19) % 8) * TAU / 8.0
		var xf := Transform3D(Basis(Vector3.UP, rot), grid_to_world(c))
		groups[gk]["xforms"].append(xf)
	for gk in groups.keys():
		var g: Dictionary = groups[gk]
		var mm := MultiMesh.new()
		mm.transform_format = MultiMesh.TRANSFORM_3D
		mm.mesh = _prototype(g["kind"], g["variant"])
		var xforms: Array = g["xforms"]
		mm.instance_count = xforms.size()
		for i in xforms.size():
			mm.set_instance_transform(i, xforms[i])
		var mmi := MultiMeshInstance3D.new()
		mmi.name = "Props_%s" % gk
		mmi.multimesh = mm
		mmi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		add_child(mmi)


func _prototype(kind: int, v: int) -> ArrayMesh:
	var parts: Array = []
	match kind:
		KIND_STONE:
			_parts_stone(parts, v)
		KIND_TREE:
			_parts_tree(parts, v)
		KIND_GRAVE:
			_parts_grave(parts)
		_:
			_parts_column(parts, v)
	return _combine(parts)


func _parts_stone(parts: Array, v: int) -> void:
	var stone := _mat("stone", Color(0.40, 0.40, 0.42))
	var s := SphereMesh.new()
	s.radius = 0.42 if v == 0 else 0.36
	s.height = 0.5
	s.radial_segments = 6
	s.rings = 4
	_part(parts, s, Transform3D(Basis(), Vector3(0.0, 0.25, 0.0)), stone)
	if v == 1:
		var s2 := SphereMesh.new()
		s2.radius = 0.22
		s2.height = 0.3
		s2.radial_segments = 6
		s2.rings = 4
		_part(parts, s2, Transform3D(Basis(), Vector3(0.28, 0.15, 0.12)), stone)


func _parts_tree(parts: Array, v: int) -> void:
	var wood := _mat("wood", Color(0.20, 0.14, 0.09))
	var trunk_h := 2.0 if v == 0 else 1.4
	var trunk := CylinderMesh.new()
	trunk.top_radius = 0.07
	trunk.bottom_radius = 0.12
	trunk.height = trunk_h
	trunk.radial_segments = 6
	trunk.rings = 1
	_part(parts, trunk, Transform3D(Basis(), Vector3(0.0, trunk_h * 0.5, 0.0)), wood)
	# [доля высоты ствола, азимут градусы, наклон от вертикали градусы, длина]
	var branches := [
		[0.55, 30.0, 50.0, 0.9], [0.72, 200.0, 55.0, 0.8], [0.90, 110.0, 60.0, 0.6], [0.95, 290.0, 35.0, 0.5],
	] if v == 0 else [
		[0.60, 20.0, 55.0, 0.7], [0.85, 160.0, 45.0, 0.6],
	]
	for b in branches:
		var h: float = trunk_h * float(b[0])
		var yaw := deg_to_rad(float(b[1]))
		var tilt := deg_to_rad(float(b[2]))
		var length: float = float(b[3])
		var d := Vector3(sin(tilt) * cos(yaw), cos(tilt), sin(tilt) * sin(yaw)).normalized()
		var br := CylinderMesh.new()
		br.top_radius = 0.01
		br.bottom_radius = 0.035
		br.height = length
		br.radial_segments = 5
		br.rings = 1
		var basis := Basis(Quaternion(Vector3.UP, d))
		_part(parts, br, Transform3D(basis, Vector3(0.0, h, 0.0) + d * length * 0.5), wood)


func _parts_grave(parts: Array) -> void:
	var stone := _mat("grave", Color(0.33, 0.33, 0.31))
	var cross := _mat("cross", Color(0.46, 0.46, 0.43))
	var tomb := BoxMesh.new()
	tomb.size = Vector3(0.6, 0.9, 0.16)
	_part(parts, tomb, Transform3D(Basis(), Vector3(0.0, 0.45, 0.0)), stone)
	var vert := BoxMesh.new()
	vert.size = Vector3(0.08, 0.36, 0.05)
	_part(parts, vert, Transform3D(Basis(), Vector3(0.0, 1.02, 0.11)), cross)
	var horiz := BoxMesh.new()
	horiz.size = Vector3(0.26, 0.08, 0.05)
	_part(parts, horiz, Transform3D(Basis(), Vector3(0.0, 1.04, 0.11)), cross)


func _parts_column(parts: Array, v: int) -> void:
	var stone := _mat("column", Color(0.50, 0.47, 0.42))
	var base := BoxMesh.new()
	base.size = Vector3(0.62, 0.2, 0.62)
	_part(parts, base, Transform3D(Basis(), Vector3(0.0, 0.1, 0.0)), stone)
	var shaft_h := 2.1 if v == 0 else 1.2
	var shaft := CylinderMesh.new()
	shaft.top_radius = 0.2
	shaft.bottom_radius = 0.25
	shaft.height = shaft_h
	shaft.radial_segments = 8
	shaft.rings = 1
	_part(parts, shaft, Transform3D(Basis(), Vector3(0.0, 0.2 + shaft_h * 0.5, 0.0)), stone)
	if v == 0:
		var cap := BoxMesh.new()
		cap.size = Vector3(0.66, 0.16, 0.66)
		_part(parts, cap, Transform3D(Basis(), Vector3(0.0, 0.2 + shaft_h + 0.08, 0.0)), stone)
	else:
		var rubble := BoxMesh.new()
		rubble.size = Vector3(0.34, 0.16, 0.3)
		_part(parts, rubble, Transform3D(Basis(), Vector3(0.42, 0.08, 0.25)), stone)


func _part(parts: Array, mesh: Mesh, xf: Transform3D, mat: Material) -> void:
	parts.append({"mesh": mesh, "xf": xf, "mat": mat})


## Склеивает части в один ArrayMesh: по одной поверхности на материал.
func _combine(parts: Array) -> ArrayMesh:
	var out := ArrayMesh.new()
	var by_mat := {}
	for p in parts:
		var m: Material = p["mat"]
		if not by_mat.has(m):
			by_mat[m] = []
		by_mat[m].append(p)
	for m in by_mat.keys():
		var st := SurfaceTool.new()
		st.begin(Mesh.PRIMITIVE_TRIANGLES)
		for p in by_mat[m]:
			st.append_from(p["mesh"], 0, p["xf"])
		st.commit(out)
		out.surface_set_material(out.get_surface_count() - 1, m)
	return out


func _mat(key: String, col: Color) -> StandardMaterial3D:
	if _mats.has(key):
		return _mats[key]
	var m := StandardMaterial3D.new()
	m.albedo_color = col
	m.roughness = 0.95
	_mats[key] = m
	return m


# ------------------------------------------------------------------ маркер цели

func _build_marker() -> void:
	var torus := TorusMesh.new()
	torus.inner_radius = MARKER_INNER
	torus.outer_radius = MARKER_OUTER
	torus.rings = 24
	torus.ring_segments = 12
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.albedo_color = Color(1.0, 0.85, 0.4, 0.85)
	_marker = MeshInstance3D.new()
	_marker.name = "TargetMarker"
	_marker.mesh = torus
	_marker.material_override = mat
	_marker.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_marker.visible = false
	add_child(_marker)
