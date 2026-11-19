extends Node2D
## Героиня: ходит по пути на AStar-сетке карты (world.gd), по тапу/клику.

signal arrived

const HeroDirs = preload("res://scripts/hero_dirs.gd")

const SPEED := 260.0
const TARGET_HEIGHT := 180.0
const FRAME_PATH := "res://assets/hero/cell_%d.png"
const FRAME_COUNT := 8
const BOB_AMP := 2.5
const SQUASH_AMP := 0.03
const MARKER_HW := 22.0
const MARKER_HH := 11.0

## Назначается из main.gd (WorldMap с find_path / world_to_grid / grid_to_world).
var world = null
## Текущая скорость на экране (для камеры: look-ahead).
var velocity := Vector2.ZERO
var is_moving := false

var _path := PackedVector2Array()
var _idx := 0
var _dir_name := "S"
var _facing := Vector2(0.0, 1.0)
var _frames: Array = []
var _use_fallback := true
var _base_scale := Vector2.ONE
var _walk_t := 0.0
var _time := 0.0
var _bob := 0.0
var _sq := 0.0

var _sprite: Sprite2D
var _marker: Node2D


func _ready() -> void:
	_frames = []
	for i in FRAME_COUNT:
		_frames.append(_load_texture(FRAME_PATH % i))
	_sprite = Sprite2D.new()
	_sprite.name = "Sprite"
	_sprite.centered = false
	add_child(_sprite)
	_marker = Node2D.new()
	_marker.name = "TargetMarker"
	_marker.top_level = true
	_marker.z_index = -5
	_marker.visible = false
	add_child(_marker)
	_marker.draw.connect(_draw_marker)
	_apply_frame()


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.pressed and mb.button_index == MOUSE_BUTTON_LEFT and world != null:
			var p: Vector2 = get_global_mouse_position()
			var cell: Vector2i = world.world_to_grid(p)
			move_to_cell(cell)
			get_viewport().set_input_as_handled()


func get_cell() -> Vector2i:
	return world.world_to_grid(position)


## Строит путь до клетки и начинает движение (новая цель заменяет старую).
func move_to_cell(cell: Vector2i) -> void:
	if world == null:
		return
	var from_cell: Vector2i = world.world_to_grid(position)
	var pts: PackedVector2Array = world.find_path(from_cell, cell)
	if pts.is_empty():
		return
	# Если первая точка (центр текущей клетки) позади нас — пропускаем её.
	if pts.size() > 1 and position.distance_squared_to(pts[1]) < position.distance_squared_to(pts[0]):
		pts.remove_at(0)
	_path = pts
	_idx = 0
	is_moving = true
	_marker.global_position = pts[pts.size() - 1]
	_marker.visible = true


func _process(delta: float) -> void:
	_time += delta
	if is_moving:
		_step(delta)
	_update_visuals(delta)
	if _marker.visible:
		_marker.queue_redraw()
	queue_redraw()


func _step(delta: float) -> void:
	var remaining := SPEED * delta
	while is_moving and remaining > 0.0:
		var to_wp: Vector2 = _path[_idx] - position
		var d := to_wp.length()
		if d <= remaining:
			position = _path[_idx]
			remaining -= d
			_idx += 1
			if _idx >= _path.size():
				_finish()
				return
		else:
			var dir := to_wp / d
			position += dir * remaining
			_set_facing(dir)
			remaining = 0.0


func _finish() -> void:
	is_moving = false
	velocity = Vector2.ZERO
	_path = PackedVector2Array()
	_marker.visible = false
	arrived.emit()


func _set_facing(dir: Vector2) -> void:
	_facing = dir
	velocity = dir * SPEED
	var nm: String = HeroDirs.dir_name_from_vector(dir)
	if nm != _dir_name:
		_dir_name = nm
		_apply_frame()


func _update_visuals(delta: float) -> void:
	var target_bob := 0.0
	var target_sq := 0.0
	if is_moving:
		_walk_t += delta
		var s := sin(_walk_t * TAU * 2.0)
		target_bob = s * BOB_AMP
		target_sq = s * SQUASH_AMP
	var k := 1.0 - exp(-12.0 * delta)
	_bob = lerpf(_bob, target_bob, k)
	_sq = lerpf(_sq, target_sq, k)
	_sprite.position = Vector2(0.0, _bob)
	_sprite.scale = _base_scale * Vector2(1.0 + _sq, 1.0 - _sq)


func _apply_frame() -> void:
	var idx: int = HeroDirs.DIR_TO_CELL.get(_dir_name, 0)
	var tex = null
	if idx >= 0 and idx < _frames.size():
		tex = _frames[idx]
	if tex == null:
		_use_fallback = true
		_sprite.visible = false
		_sprite.texture = null
	else:
		_use_fallback = false
		_sprite.visible = true
		_sprite.texture = tex
		_sprite.flip_h = HeroDirs.DIR_FLIP.get(_dir_name, false)
		# Ноги (низ по центру) совпадают с origin героя.
		_sprite.offset = Vector2(-tex.get_width() * 0.5, -tex.get_height())
		var k: float = TARGET_HEIGHT / float(tex.get_height())
		_base_scale = Vector2.ONE * k
	queue_redraw()


func _draw() -> void:
	_draw_shadow()
	if _use_fallback:
		# Запасная капсула, если кадров ещё нет. Низ капсулы = origin.
		draw_set_transform(Vector2(0.0, _bob), 0.0, Vector2(1.0 + _sq, 1.0 - _sq))
		draw_rect(Rect2(-16.0, -134.0, 32.0, 134.0), Color(0.42, 0.30, 0.22))
		draw_circle(Vector2(0.0, -134.0), 16.0, Color(0.80, 0.68, 0.58))
		draw_set_transform(Vector2.ZERO, 0.0, Vector2.ONE)
		draw_line(Vector2(0.0, -60.0), Vector2(0.0, -60.0) + _facing * 28.0, Color(1.0, 0.9, 0.5, 0.8), 3.0)


func _draw_shadow() -> void:
	var pts := PackedVector2Array()
	for k in 24:
		var a := TAU * float(k) / 24.0
		pts.append(Vector2(cos(a) * 30.0, sin(a) * 11.0))
	draw_colored_polygon(pts, Color(0.0, 0.0, 0.0, 0.4))


## Колбэк draw-сигнала маркера (рисуется на узле _marker, top-level).
func _draw_marker() -> void:
	var pulse := 0.5 + 0.5 * sin(_time * 6.0)
	var k := 1.0 + 0.12 * pulse
	var pts := PackedVector2Array([
		Vector2(0.0, -MARKER_HH * k), Vector2(MARKER_HW * k, 0.0),
		Vector2(0.0, MARKER_HH * k), Vector2(-MARKER_HW * k, 0.0),
	])
	_marker.draw_colored_polygon(pts, Color(1.0, 0.85, 0.4, 0.12 + 0.12 * pulse))
	var closed := pts.duplicate()
	closed.append(pts[0])
	_marker.draw_polyline(closed, Color(1.0, 0.85, 0.4, 0.5 + 0.5 * pulse), 2.0)


func _load_texture(path: String) -> Texture2D:
	if ResourceLoader.exists(path):
		var r = load(path)
		if r is Texture2D:
			return r
	if FileAccess.file_exists(path):
		var img := Image.load_from_file(path)
		if img != null:
			return ImageTexture.create_from_image(img)
	return null
