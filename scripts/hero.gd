extends Node2D
## Героиня: ходит по пути на AStar-сетке карты (world_map.gd) по тапу/клику.
## Состояния: IDLE, WALK, ATTACK, DEAD. Кадры и параметры анимаций — scripts/hero_anim.gd.

signal arrived
signal attack_finished
signal died
signal revived

const HeroDirs = preload("res://scripts/hero_dirs.gd")
const HeroAnim = preload("res://scripts/hero_anim.gd")

enum State { IDLE, WALK, ATTACK, DEAD }

const SPEED := 260.0
const REVIVE_DELAY := 2.5
const MARKER_HW := 22.0
const MARKER_HH := 11.0

## Назначается из main.gd (WorldMap с find_path / world_to_grid / grid_to_world).
var world = null
## Текущая скорость на экране (для камеры: look-ahead). Ненулевая только при ходьбе.
var velocity := Vector2.ZERO
## true, пока есть путь (в том числе когда героиня стоит в атаке и ждёт продолжения).
var is_moving := false

var _state: int = State.IDLE
var _path := PackedVector2Array()
var _idx := 0
var _dir_name := "S"
var _facing := Vector2(0.0, 1.0)
var _anim := "idle"
var _anim_t := 0.0
var _anim_done := false
var _revive_left := 0.0
## anim -> Array (ряд) -> Array (кадр) -> Texture2D или null.
var _textures: Dictionary = {}
var _use_fallback := true
var _time := 0.0

var _sprite: Sprite2D
var _marker: Node2D


func _ready() -> void:
	_load_textures()
	_sprite = Sprite2D.new()
	_sprite.name = "Sprite"
	_sprite.centered = false
	_sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	add_child(_sprite)
	_marker = Node2D.new()
	_marker.name = "TargetMarker"
	_marker.top_level = true
	_marker.z_index = -5
	_marker.visible = false
	add_child(_marker)
	_marker.draw.connect(_draw_marker)
	_play("idle")
	_update_visuals()


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


func get_state() -> int:
	return _state


## Строит путь до клетки. Во время атаки путь ждёт окончания удара; при смерти игнорируется.
func move_to_cell(cell: Vector2i) -> void:
	if world == null or _state == State.DEAD:
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
	if _state == State.IDLE:
		_set_state(State.WALK)


## Удар в текущем направлении. Героиня стоит; после удара идёт дальше, если есть путь.
func attack() -> void:
	if _state != State.IDLE and _state != State.WALK:
		return
	velocity = Vector2.ZERO
	_set_state(State.ATTACK)


## Смерть: проигрывается один раз, затем воскрешение через REVIVE_DELAY секунд.
func die() -> void:
	if _state == State.DEAD:
		return
	_path = PackedVector2Array()
	is_moving = false
	velocity = Vector2.ZERO
	_marker.visible = false
	_revive_left = REVIVE_DELAY
	_set_state(State.DEAD)
	died.emit()


func _process(delta: float) -> void:
	_time += delta
	if _state == State.WALK:
		_step(delta)
	elif _state == State.DEAD:
		_revive_left -= delta
		if _revive_left <= 0.0:
			_revive()
	_anim_t += delta
	_check_anim_end()
	_update_visuals()
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
	_set_state(State.IDLE)
	arrived.emit()


func _set_facing(dir: Vector2) -> void:
	_facing = dir
	velocity = dir * SPEED
	_dir_name = HeroDirs.dir_name_from_vector(dir)


func _set_state(s: int) -> void:
	_state = s
	match s:
		State.IDLE:
			_play("idle")
		State.WALK:
			_play("walk")
		State.ATTACK:
			_play("attack")
		State.DEAD:
			_play("death")


func _play(anim: String) -> void:
	_anim = anim
	_anim_t = 0.0
	_anim_done = false


## Конец одноразовой анимации (attack / death).
func _check_anim_end() -> void:
	if _anim_done:
		return
	var info: Dictionary = HeroAnim.ANIMS[_anim]
	if bool(info["loop"]):
		return
	if _anim_t * float(info["fps"]) >= float(info["frames"]):
		_anim_done = true
		if _state == State.ATTACK:
			_set_state(State.WALK if is_moving else State.IDLE)
			attack_finished.emit()


func _revive() -> void:
	_set_state(State.IDLE)
	revived.emit()


func _update_visuals() -> void:
	var entry: Array = HeroAnim.DIR_TABLES[_anim][_dir_name]
	var row: int = entry[0]
	var flip: bool = entry[1]
	var tex: Texture2D = _frame_texture(row)
	if tex == null:
		_use_fallback = true
		_sprite.visible = false
		_sprite.texture = null
		return
	_use_fallback = false
	_sprite.visible = true
	_sprite.texture = tex
	_sprite.flip_h = flip
	_sprite.scale = Vector2.ONE * HeroAnim.scale_for(_anim)
	_sprite.offset = HeroAnim.origin_offset(_anim, tex)


## Текущий кадр выбранного ряда по времени анимации.
func _frame_texture(row: int) -> Texture2D:
	var rows: Array = _textures.get(_anim, [])
	if row < 0 or row >= rows.size():
		return null
	var frames: Array = rows[row]
	var info: Dictionary = HeroAnim.ANIMS[_anim]
	var n := int(info["frames"])
	var idx := int(_anim_t * float(info["fps"]))
	if bool(info["loop"]):
		idx = idx % n
	else:
		idx = mini(idx, n - 1)
	if idx >= frames.size():
		return null
	return frames[idx]


func _load_textures() -> void:
	_textures = {}
	for key in HeroAnim.ANIMS.keys():
		var anim: String = key
		var info: Dictionary = HeroAnim.ANIMS[anim]
		var rows: Array = []
		for r in int(info["rows"]):
			var frames: Array = []
			for f in int(info["frames"]):
				frames.append(_load_texture(HeroAnim.frame_path(anim, r, f)))
			rows.append(frames)
		_textures[anim] = rows


func _draw() -> void:
	_draw_shadow()
	if _use_fallback:
		# Запасная капсула, если кадров ещё нет. Низ капсулы = origin.
		draw_rect(Rect2(-16.0, -134.0, 32.0, 134.0), Color(0.42, 0.30, 0.22))
		draw_circle(Vector2(0.0, -134.0), 16.0, Color(0.80, 0.68, 0.58))
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
