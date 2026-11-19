extends Node3D
## Героиня (3D). Модель и анимации: models/hero_anim.glb; шпага: models/rapier.glb в кости WeaponSocket.R.
## glTF грузится в рантайме через GLTFDocument (без .import-кэша), поэтому models/ не меняется.
## Состояния: IDLE, WALK, ATTACK, DEAD. Путь по тапу строит world3d.find_path, ходьба по центрам клеток.

signal arrived
signal attack_finished
signal died
signal death_finished
signal revived

enum State { IDLE, WALK, ATTACK, DEAD }

const HERO_GLB := "res://models/hero_dressed.glb"
const RAPIER_GLB := "res://models/rapier.glb"
const SOCKET_BONE_PREFIX := "WeaponSocket"

## Анимация Walk: один цикл = 1.0 с при корне 0.9 м/с. Скорость проигрывания = масштаб шага,
## поэтому скорость корня = 0.9 * WALK_SPEED_SCALE, и ноги не скользят.
const WALK_SPEED_SCALE := 2.5
const MOVE_SPEED := 0.9 * WALK_SPEED_SCALE
const IDLE_SPEED_SCALE := 1.0
const ATTACK_SPEED_SCALE := 1.4
const DEATH_SPEED_SCALE := 1.0
const BLEND_TIME := 0.15
const TURN_RATE := 12.0
const REVIVE_DELAY := 2.5

## Шпага относительно кости WeaponSocket.R: смещение (м) и поворот (градусы, порядок YXZ).
## Клинок в rapier.glb вдоль -Z, рукоять около начала. Ось Y кости WeaponSocket.R = направление кисти
## (вперёд и чуть вверх в idle). Клинок идёт вдоль +Y кости, гарда поперёк кулака (ось X шпаги = +Z кости),
## высота гарды (ось Y шпаги) = -X кости. Смещение -0.04 по Y кости: рукоять в ладони, у запястья.
const RAPIER_OFFSET := Vector3(0.0, -0.04, 0.0)
const RAPIER_ROT_DEG := Vector3(90.0, -90.0, 0.0)

## Лицо модели смотрит в +Z (glTF): yaw = atan2(dir.x, dir.z) + MODEL_YAW_OFFSET.
const MODEL_YAW_OFFSET := 0.0

const TORCH_POS := Vector3(0.0, 1.7, 0.0)
const TORCH_COLOR := Color(1.0, 0.72, 0.42)
const TORCH_RANGE := 7.0
const TORCH_ENERGY := 1.0
const BLOB_SIZE := 1.9
const BLOB_ALPHA := 0.8

## Назначается из main.gd (world3d.gd: find_path, grid_to_world, show_marker, hide_marker).
var world = null
## Горизонтальная скорость (м/с) — для камеры: look-ahead. Ненулевая только при ходьбе.
var velocity := Vector3.ZERO

var _state: int = State.IDLE
var _path := PackedVector3Array()
var _model: Node3D = null
var _anim: AnimationPlayer = null
var _skeleton: Skeleton3D = null
## ключ ("idle", "walk", ...) -> реальное имя анимации в AnimationPlayer
var _anim_names := {}
var _target_yaw := 0.0
var _revive_timer := -1.0


## Загрузка glTF/glb из res:// в дерево узлов (рантайм, без импорта).
static func load_gltf(path: String) -> Node3D:
	# В экспортированном APK glb лежат как импортированные ресурсы — берём их.
	if ResourceLoader.exists(path):
		var res = load(path)
		if res is PackedScene:
			return (res as PackedScene).instantiate() as Node3D
	var doc := GLTFDocument.new()
	var st := GLTFState.new()
	if doc.append_from_file(path, st) != OK:
		push_error("glTF не загрузился: " + path)
		return null
	return doc.generate_scene(st) as Node3D


func _ready() -> void:
	_model = Node3D.new()
	_model.name = "Model"
	add_child(_model)
	_load_model()
	_build_blob()
	_build_torch()
	_set_state(State.IDLE)


func _process(delta: float) -> void:
	if _state == State.WALK:
		_step(delta)
	if _revive_timer > 0.0:
		_revive_timer -= delta
		if _revive_timer <= 0.0:
			_revive_timer = -1.0
			_set_state(State.IDLE)
			revived.emit()
	var k := 1.0 - exp(-TURN_RATE * delta)
	_model.rotation.y = lerp_angle(_model.rotation.y, _target_yaw, k)


## Тап по мировой точке земли: строит путь и идёт. Во время удара путь ждёт окончания удара.
func move_to(target: Vector3) -> void:
	if world == null or _state == State.DEAD:
		return
	var pts: PackedVector3Array = world.find_path(position, target)
	if pts.is_empty():
		return
	# Первая точка (центр клетки) позади — пропускаем.
	if pts.size() > 1 and position.distance_squared_to(pts[1]) < position.distance_squared_to(pts[0]):
		pts.remove_at(0)
	_path = pts
	world.show_marker(pts[pts.size() - 1])
	if _state == State.IDLE:
		_set_state(State.WALK)


## Удар на месте. После удара героиня идёт дальше, если путь ещё есть.
func attack() -> void:
	if _state != State.IDLE and _state != State.WALK:
		return
	velocity = Vector3.ZERO
	_set_state(State.ATTACK)


## Смерть: один проигрыш Death, затем лежит REVIVE_DELAY секунд и воскресает.
func die() -> void:
	if _state == State.DEAD:
		return
	_path = PackedVector3Array()
	velocity = Vector3.ZERO
	_revive_timer = -1.0
	if world != null:
		world.hide_marker()
	_set_state(State.DEAD)
	died.emit()


## Мгновенный поворот модели (без сглаживания), например при старте.
func face_direction(dir: Vector3) -> void:
	dir.y = 0.0
	if dir.length_squared() < 0.0001:
		return
	_target_yaw = atan2(dir.x, dir.z) + MODEL_YAW_OFFSET
	_model.rotation.y = _target_yaw


func state_name() -> String:
	match _state:
		State.IDLE:
			return "idle"
		State.WALK:
			return "walk"
		State.ATTACK:
			return "attack"
		_:
			return "dead"


# ------------------------------------------------------------------ движение

func _step(delta: float) -> void:
	var remaining := MOVE_SPEED * delta
	while remaining > 0.0 and not _path.is_empty():
		var to_wp: Vector3 = _path[0] - position
		to_wp.y = 0.0
		var d := to_wp.length()
		if d <= remaining:
			position = Vector3(_path[0].x, 0.0, _path[0].z)
			remaining -= d
			_path.remove_at(0)
		else:
			var dir := to_wp / d
			position += dir * remaining
			remaining = 0.0
		if d > 0.0001:
			_face(to_wp)
	if _path.is_empty():
		_finish()
	else:
		velocity = _path_dir() * MOVE_SPEED


func _path_dir() -> Vector3:
	var to_wp: Vector3 = _path[0] - position
	to_wp.y = 0.0
	if to_wp.length_squared() < 0.0001:
		return Vector3.ZERO
	return to_wp.normalized()


func _face(dir: Vector3) -> void:
	dir.y = 0.0
	if dir.length_squared() > 0.0001:
		_target_yaw = atan2(dir.x, dir.z) + MODEL_YAW_OFFSET


func _finish() -> void:
	velocity = Vector3.ZERO
	if world != null:
		world.hide_marker()
	_set_state(State.IDLE)
	arrived.emit()


# ------------------------------------------------------------------ анимации

func _set_state(s: int) -> void:
	_state = s
	match s:
		State.IDLE:
			_play("idle", IDLE_SPEED_SCALE)
		State.WALK:
			_play("walk", WALK_SPEED_SCALE)
		State.ATTACK:
			_play("attack", ATTACK_SPEED_SCALE)
		State.DEAD:
			_play("death", DEATH_SPEED_SCALE)


func _play(key: String, speed: float) -> void:
	if _anim == null or not _anim_names.has(key):
		return
	_anim.play(_anim_names[key], BLEND_TIME, speed)


func _on_animation_finished(anim_name: StringName) -> void:
	var key := String(anim_name).to_lower()
	if key == "attack" and _state == State.ATTACK:
		_set_state(State.WALK if not _path.is_empty() else State.IDLE)
		attack_finished.emit()
	elif key == "death" and _state == State.DEAD:
		_revive_timer = REVIVE_DELAY
		death_finished.emit()


# ------------------------------------------------------------------ сборка узлов

func _load_model() -> void:
	var root := load_gltf(HERO_GLB)
	if root == null:
		return
	_model.add_child(root)
	_anim = _find_first(root, "AnimationPlayer") as AnimationPlayer
	_skeleton = _find_first(root, "Skeleton3D") as Skeleton3D
	if _anim == null or _skeleton == null:
		push_error("hero: в модели нет AnimationPlayer или Skeleton3D")
		return
	for n in _anim.get_animation_list():
		_anim_names[String(n).to_lower()] = n
	for key in ["idle", "walk"]:
		if _anim_names.has(key):
			_anim.get_animation(_anim_names[key]).loop_mode = Animation.LOOP_LINEAR
	for key in ["attack", "death"]:
		if _anim_names.has(key):
			_anim.get_animation(_anim_names[key]).loop_mode = Animation.LOOP_NONE
	_anim.animation_finished.connect(_on_animation_finished)
	_attach_rapier()


func _attach_rapier() -> void:
	var bone := ""
	for i in _skeleton.get_bone_count():
		var bn := _skeleton.get_bone_name(i)
		if bn.begins_with(SOCKET_BONE_PREFIX):
			bone = bn
			break
	if bone == "":
		push_error("hero: кость WeaponSocket не найдена")
		return
	var socket := BoneAttachment3D.new()
	socket.name = "WeaponSocket"
	socket.bone_name = bone
	_skeleton.add_child(socket)
	var rapier := load_gltf(RAPIER_GLB)
	if rapier == null:
		return
	rapier.position = RAPIER_OFFSET
	rapier.rotation_degrees = RAPIER_ROT_DEG
	socket.add_child(rapier)


func _find_first(n: Node, cls: String) -> Node:
	if n.is_class(cls):
		return n
	for c in n.get_children():
		var f := _find_first(c, cls)
		if f != null:
			return f
	return null


## Круглое тёмное пятно под героиней (плоский QuadMesh с радиальным градиентом).
func _build_blob() -> void:
	var grad := Gradient.new()
	grad.set_color(0, Color(0.0, 0.0, 0.0, BLOB_ALPHA))
	grad.set_color(1, Color(0.0, 0.0, 0.0, 0.0))
	var tex := GradientTexture2D.new()
	tex.gradient = grad
	tex.fill = GradientTexture2D.FILL_RADIAL
	tex.fill_from = Vector2(0.5, 0.5)
	tex.fill_to = Vector2(1.0, 0.5)
	tex.width = 128
	tex.height = 128
	var mat := StandardMaterial3D.new()
	mat.albedo_texture = tex
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	var quad := QuadMesh.new()
	quad.size = Vector2(BLOB_SIZE, BLOB_SIZE)
	var blob := MeshInstance3D.new()
	blob.name = "Blob"
	blob.mesh = quad
	blob.material_override = mat
	blob.rotation_degrees = Vector3(-90.0, 0.0, 0.0)
	blob.position = Vector3(0.0, 0.01, 0.0)
	blob.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(blob)


## Факел героини: тёплый свет вокруг неё (без теней).
func _build_torch() -> void:
	var torch := OmniLight3D.new()
	torch.name = "Torch"
	torch.position = TORCH_POS
	torch.light_color = TORCH_COLOR
	torch.light_energy = TORCH_ENERGY
	torch.omni_range = TORCH_RANGE
	torch.shadow_enabled = false
	add_child(torch)
