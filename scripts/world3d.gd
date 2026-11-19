extends Node3D
## Карта 36x36 клеток по 1 м в плоскости XZ (y = 0), центр карты в начале координат.
## Земля: одна сетка 0.5 м (ArrayMesh), веса трёх покрытий (земля, трава, камень) лежат в вершинных
## цветах и плавно интерполируются. Шейдер смешивает три бесшовные текстуры из assets/tex/ по весам.
## Декор (трава, камни, кости, трещины, лужи) — MultiMesh, не влияет на навигацию.
## Препятствия: MultiMesh из примитивов (без коллайдеров).
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

## Шаг вершин земли, м. Меньше — плавнее переходы между покрытиями.
const GROUND_STEP := 0.5
## Одна текстура покрывает TILE_M метров; MACRO_M — крупные пятна для разбивки повторов.
const TILE_M := 3.5
const MACRO_M := 14.0
## Вокруг стартовой клетки декор не ставится (радиус, м).
const DECOR_SAFE_R := 2.0
const TEX_DIR := "res://assets/tex/"
const GROUND_GAIN := 2.0

const MARKER_Y := 0.03
const MARKER_INNER := 0.34
const MARKER_OUTER := 0.46
const MARKER_PULSE_RATE := 6.0
const MARKER_PULSE_AMP := 0.1

## Шейдер земли: fragment-only, без compute, работает в gl_compatibility.
const GROUND_SHADER := """
shader_type spatial;
render_mode cull_disabled, specular_schlick_ggx;

uniform sampler2D tex_dirt : source_color, filter_linear_mipmap, repeat_enable;
uniform sampler2D tex_grass : source_color, filter_linear_mipmap, repeat_enable;
uniform sampler2D tex_stone : source_color, filter_linear_mipmap, repeat_enable;
uniform sampler2D tex_macro : source_color, filter_linear_mipmap, repeat_enable;
uniform float tile_m = 3.5;
uniform float macro_m = 14.0;
uniform float map_half = 18.0;
uniform float edge_w = 3.0;
uniform vec3 far_color : source_color = vec3(0.05, 0.05, 0.06);
uniform float fog_near = 14.0;
uniform float fog_far = 36.0;
uniform float fog_strength = 0.35;
// Поднимает яркость после sRGB-декодирования тёмных текстур до средней.
uniform float gain = 2.4;

varying vec3 wpos;
varying vec4 wcol;

void vertex() {
	wpos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	wcol = COLOR;
}

void fragment() {
	vec2 uv = wpos.xz / tile_m;
	vec3 d = texture(tex_dirt, uv).rgb;
	vec3 g = texture(tex_grass, uv * 1.37 + vec2(0.31, 0.57)).rgb;
	vec3 s = texture(tex_stone, uv * 0.83 + vec2(0.12, 0.44)).rgb;
	float wsum = max(wcol.r + wcol.g + wcol.b, 0.001);
	vec3 col = (d * wcol.r + g * wcol.g + s * wcol.b) / wsum;
	float m = texture(tex_macro, wpos.xz / macro_m).r;
	col *= mix(0.82, 1.12, m) * gain * vec3(1.06, 0.98, 0.86);
	// Края карты уходят в темноту.
	float edge = max(abs(wpos.x), abs(wpos.z));
	col = mix(far_color, col, 1.0 - smoothstep(map_half - edge_w, map_half, edge));
	// Дальняя земля слегка темнеет (поверх общего тумана сцены).
	float dist = distance(wpos.xz, CAMERA_POSITION_WORLD.xz);
	col = mix(col, far_color, smoothstep(fog_near, fog_far, dist) * fog_strength);
	ALBEDO = col;
	ROUGHNESS = 1.0;
	SPECULAR = 0.2;
}
"""

var _kinds := {}
var _astar: AStarGrid2D = null
var _mats := {}
var _marker: MeshInstance3D = null
var _time := 0.0
var _fn_grass: FastNoiseLite = null
var _fn_stone: FastNoiseLite = null


func _ready() -> void:
	_build_obstacles()
	_build_astar()
	_build_ground()
	_build_props()
	_build_decor()
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
	_fn_grass = FastNoiseLite.new()
	_fn_grass.seed = RNG_SEED + 11
	_fn_grass.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
	_fn_grass.frequency = 0.16
	_fn_grass.fractal_octaves = 3
	_fn_stone = FastNoiseLite.new()
	_fn_stone.seed = RNG_SEED + 23
	_fn_stone.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
	_fn_stone.frequency = 0.12
	_fn_stone.fractal_octaves = 3

	var steps := int(round(N * CELL / GROUND_STEP))
	var n := steps + 1
	var verts := PackedVector3Array()
	var normals := PackedVector3Array()
	var colors := PackedColorArray()
	verts.resize(n * n)
	normals.resize(n * n)
	colors.resize(n * n)
	for j in n:
		for i in n:
			var x := -HALF + i * GROUND_STEP
			var z := -HALF + j * GROUND_STEP
			var k := j * n + i
			verts[k] = Vector3(x, 0.0, z)
			normals[k] = Vector3.UP
			colors[k] = _ground_weights(Vector2(x, z))
	var indices := PackedInt32Array()
	indices.resize(steps * steps * 6)
	var w := 0
	for j in steps:
		for i in steps:
			var a := j * n + i
			var b := a + 1
			var c := a + n
			var d := c + 1
			indices[w] = a
			indices[w + 1] = c
			indices[w + 2] = b
			indices[w + 3] = b
			indices[w + 4] = c
			indices[w + 5] = d
			w += 6

	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_COLOR] = colors
	arrays[Mesh.ARRAY_INDEX] = indices
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)

	var shader := Shader.new()
	shader.code = GROUND_SHADER
	var mat := ShaderMaterial.new()
	mat.shader = shader
	mat.set_shader_parameter("tex_dirt", _load_tex("ground_dirt.png"))
	mat.set_shader_parameter("tex_grass", _load_tex("ground_grass.png"))
	mat.set_shader_parameter("tex_stone", _load_tex("ground_stone.png"))
	mat.set_shader_parameter("tex_macro", _load_tex("ground_macro.png"))
	mat.set_shader_parameter("tile_m", TILE_M)
	mat.set_shader_parameter("macro_m", MACRO_M)
	mat.set_shader_parameter("map_half", HALF)
	mat.set_shader_parameter("gain", GROUND_GAIN)

	var mi := MeshInstance3D.new()
	mi.name = "Ground"
	mi.mesh = mesh
	mi.material_override = mat
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(mi)


## Веса покрытий в точке карты: r = земля, g = трава, b = камень. Нормируются в шейдере.
func _ground_weights(p: Vector2) -> Color:
	var grass := smoothstep(0.0, 0.3, _fn_grass.get_noise_2d(p.x, p.y))
	var stone := smoothstep(0.26, 0.5, _fn_stone.get_noise_2d(p.x, p.y))
	var dirt := (1.0 - grass) * (1.0 - stone)
	return Color(dirt, grass, stone)


func _load_tex(file: String) -> Texture2D:
	var tex := load(TEX_DIR + file) as Texture2D
	if tex == null:
		push_warning("Нет текстуры %s (запусти tools/gen_textures.py и --import)" % file)
	return tex


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
	# Тёмный сланец, сферы с большим числом сегментов (гладкая нормаль), сплюснутые по высоте.
	var stone := _mat("stone", Color(0.20, 0.20, 0.22))
	var s := SphereMesh.new()
	s.radius = 0.42 if v == 0 else 0.36
	s.height = 0.5
	s.radial_segments = 12
	s.rings = 7
	_part(parts, s, Transform3D(Basis().scaled(Vector3(1.0, 0.85, 0.9)), Vector3(0.0, 0.2, 0.0)), stone)
	if v == 1:
		var s2 := SphereMesh.new()
		s2.radius = 0.22
		s2.height = 0.3
		s2.radial_segments = 10
		s2.rings = 6
		_part(parts, s2, Transform3D(Basis(), Vector3(0.28, 0.12, 0.12)), stone)


func _parts_tree(parts: Array, v: int) -> void:
	var wood := _mat("wood", Color(0.12, 0.09, 0.06))
	var trunk_h := 2.0 if v == 0 else 1.4
	var trunk := CylinderMesh.new()
	trunk.top_radius = 0.07
	trunk.bottom_radius = 0.12
	trunk.height = trunk_h
	trunk.radial_segments = 10
	trunk.rings = 2
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
		br.radial_segments = 6
		br.rings = 1
		var basis := Basis(Quaternion(Vector3.UP, d))
		_part(parts, br, Transform3D(basis, Vector3(0.0, h, 0.0) + d * length * 0.5), wood)


func _parts_grave(parts: Array) -> void:
	var stone := _mat("grave", Color(0.22, 0.22, 0.21))
	var cross := _mat("cross", Color(0.34, 0.34, 0.32))
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
	# Светлый известняк: силуэт колонны читается на тёмной земле.
	var stone := _mat("column", Color(0.58, 0.55, 0.50))
	var base := BoxMesh.new()
	base.size = Vector3(0.62, 0.2, 0.62)
	_part(parts, base, Transform3D(Basis(), Vector3(0.0, 0.1, 0.0)), stone)
	var shaft_h := 2.1 if v == 0 else 1.2
	var shaft := CylinderMesh.new()
	shaft.top_radius = 0.2
	shaft.bottom_radius = 0.25
	shaft.height = shaft_h
	shaft.radial_segments = 16
	shaft.rings = 4
	_part(parts, shaft, Transform3D(Basis(), Vector3(0.0, 0.2 + shaft_h * 0.5, 0.0)), stone)
	if v == 0:
		var cap := BoxMesh.new()
		cap.size = Vector3(0.66, 0.16, 0.66)
		_part(parts, cap, Transform3D(Basis(), Vector3(0.0, 0.2 + shaft_h + 0.08, 0.0)), stone)
	else:
		var rubble := SphereMesh.new()
		rubble.radius = 0.2
		rubble.height = 0.22
		rubble.radial_segments = 10
		rubble.rings = 5
		_part(parts, rubble, Transform3D(Basis().scaled(Vector3(1.2, 0.7, 1.0)), Vector3(0.42, 0.08, 0.25)), stone)


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


# ------------------------------------------------------------------ декор (не блокирует путь)

func _build_decor() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = RNG_SEED + 77
	var start := grid_to_world(START_CELL)
	var tufts: Array = []
	var pebbles: Array = []
	var bones: Array = []
	var skulls: Array = []
	var cracks: Array = []
	var puddles: Array = []
	for j in N:
		for i in N:
			var c := Vector2i(i, j)
			if _kinds.has(c):
				continue
			var base := grid_to_world(c)
			if Vector2(base.x - start.x, base.z - start.z).length() < DECOR_SAFE_R:
				continue
			# Каждый тип тянет свой кусок клетки; позиция внутри клетки случайная.
			if rng.randf() < 0.3:
				tufts.append(_scatter_xform(base, rng, 0.0, rng.randf_range(0.7, 1.2)))
			if rng.randf() < 0.1:
				pebbles.append(_scatter_xform(base, rng, 0.02, rng.randf_range(0.7, 1.4)))
			if rng.randf() < 0.025:
				bones.append(_scatter_xform(base, rng, 0.02, 1.0))
			if rng.randf() < 0.012:
				skulls.append(_scatter_xform(base, rng, 0.02, rng.randf_range(0.9, 1.1)))
			if rng.randf() < 0.035:
				cracks.append(_scatter_xform(base, rng, 0.006, rng.randf_range(0.8, 1.3)))
			if rng.randf() < 0.018:
				puddles.append(_scatter_xform(base, rng, 0.004, rng.randf_range(0.8, 1.2)))
	_add_multimesh("Decor_Tufts", _proto_tuft(), tufts)
	_add_multimesh("Decor_Pebbles", _proto_pebble(), pebbles)
	_add_multimesh("Decor_Bones", _proto_bone(), bones)
	_add_multimesh("Decor_Skulls", _proto_skull(), skulls)
	_add_multimesh("Decor_Cracks", _proto_crack(), cracks)
	_add_multimesh("Decor_Puddles", _proto_puddle(), puddles)


func _scatter_xform(base: Vector3, rng: RandomNumberGenerator, y: float, s: float) -> Transform3D:
	var off := Vector3(rng.randf_range(-0.42, 0.42), y, rng.randf_range(-0.42, 0.42))
	var basis := Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(s, s, s))
	return Transform3D(basis, Vector3(base.x, 0.0, base.z) + off)


func _add_multimesh(node_name: String, proto: Mesh, xforms: Array) -> void:
	if xforms.is_empty():
		return
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.mesh = proto
	mm.instance_count = xforms.size()
	for i in xforms.size():
		mm.set_instance_transform(i, xforms[i])
	var mmi := MultiMeshInstance3D.new()
	mmi.name = node_name
	mmi.multimesh = mm
	mmi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(mmi)


## Пучок сухой травы: 4 вытянутых лезвия разного наклона, цвет от корня (тёмный) к кончику (блёклый).
func _proto_tuft() -> ArrayMesh:
	var parts: Array = []
	var mat := _vmat("tuft")
	for k in 4:
		var yaw := float(k) * TAU / 4.0 + 0.4
		var lean := 0.04 + 0.03 * float(k % 2)
		var h := 0.42 + 0.08 * float(k % 3)
		_part(parts, _blade(h * 1.5, 0.045, lean * 1.5), Transform3D(Basis(Vector3.UP, yaw), Vector3(0.0, 0.0, 0.0)), mat)
	return _combine(parts)


func _blade(h: float, w: float, lean: float) -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var root := Color(0.20, 0.18, 0.11)
	var tip := Color(0.56, 0.50, 0.32)
	var a := Vector3(-w, 0.0, 0.0)
	var b := Vector3(w, 0.0, 0.0)
	var t := Vector3(lean, h, 0.0)
	for p in [[a, root], [b, root], [t, tip]]:
		st.set_color(p[1])
		st.set_normal(Vector3.UP)
		st.add_vertex(p[0])
	var mesh := st.commit()
	return mesh


func _proto_pebble() -> ArrayMesh:
	var parts: Array = []
	var s := SphereMesh.new()
	s.radius = 0.12
	s.height = 0.14
	s.radial_segments = 8
	s.rings = 4
	_part(parts, s, Transform3D(Basis(), Vector3(0.0, 0.05, 0.0)), _mat("pebble", Color(0.30, 0.29, 0.27)))
	return _combine(parts)


func _proto_bone() -> ArrayMesh:
	var parts: Array = []
	var mat := _mat("bone", Color(0.42, 0.39, 0.33))
	var shaft := CylinderMesh.new()
	shaft.top_radius = 0.022
	shaft.bottom_radius = 0.022
	shaft.height = 0.26
	shaft.radial_segments = 6
	shaft.rings = 1
	_part(parts, shaft, Transform3D(Basis(Vector3.FORWARD, PI * 0.5), Vector3(0.0, 0.03, 0.0)), mat)
	for e in [-0.13, 0.13]:
		var knob := SphereMesh.new()
		knob.radius = 0.04
		knob.height = 0.08
		knob.radial_segments = 6
		knob.rings = 3
		_part(parts, knob, Transform3D(Basis(), Vector3(e, 0.03, 0.0)), mat)
	return _combine(parts)


func _proto_skull() -> ArrayMesh:
	var parts: Array = []
	var mat := _mat("skull", Color(0.46, 0.43, 0.37))
	var cranium := SphereMesh.new()
	cranium.radius = 0.1
	cranium.height = 0.18
	cranium.radial_segments = 10
	cranium.rings = 6
	_part(parts, cranium, Transform3D(Basis(), Vector3(0.0, 0.09, 0.0)), mat)
	var jaw := BoxMesh.new()
	jaw.size = Vector3(0.1, 0.04, 0.07)
	_part(parts, jaw, Transform3D(Basis(), Vector3(0.0, 0.02, 0.06)), mat)
	return _combine(parts)


## Трещина: зигзаг из четырёх полос тёмной краски на земле, без освещения.
func _proto_crack() -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var dark := Color(0.14, 0.11, 0.08)
	var pts := [Vector2(-0.45, 0.0), Vector2(-0.22, 0.07), Vector2(-0.02, -0.04), Vector2(0.2, 0.06), Vector2(0.45, -0.02)]
	for k in pts.size() - 1:
		var p0: Vector2 = pts[k]
		var p1: Vector2 = pts[k + 1]
		var dir := (p1 - p0).normalized()
		var nrm := Vector2(-dir.y, dir.x) * 0.011
		var q := [p0 - nrm, p0 + nrm, p1 + nrm, p1 - nrm]
		for idx in [0, 1, 2, 0, 2, 3]:
			st.set_color(dark)
			st.set_normal(Vector3.UP)
			st.add_vertex(Vector3(q[idx].x, 0.0, q[idx].y))
	var mesh := st.commit()
	var m := _vmat("crack")
	mesh.surface_set_material(0, m)
	return mesh


func _proto_puddle() -> ArrayMesh:
	var parts: Array = []
	var disc := CylinderMesh.new()
	disc.top_radius = 0.24
	disc.bottom_radius = 0.24
	disc.height = 0.004
	disc.radial_segments = 12
	disc.rings = 1
	var m := StandardMaterial3D.new()
	m.albedo_color = Color(0.08, 0.07, 0.05, 0.3)
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.roughness = 0.05
	m.metallic = 0.4
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	_part(parts, disc, Transform3D(Basis(), Vector3(0.0, 0.002, 0.0)), m)
	return _combine(parts)


## Материал с цветом из вершин (для декора, где каждая вершина своего цвета).
func _vmat(key: String) -> StandardMaterial3D:
	if _mats.has(key):
		return _mats[key]
	var m := StandardMaterial3D.new()
	m.vertex_color_use_as_albedo = true
	m.roughness = 1.0
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
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
