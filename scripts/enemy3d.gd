class_name Enemy3D
extends Node3D
## Враг «Скелет-пират» (временный, из примитивов в коде). Один объект: HP, ИИ, анимации,
## полоска здоровья над головой, кольцо выбора, попадание лучом камеры.
## Подмена вида: set_visual_scene(scene) заменяет тело целиком (процедурные анимации выключаются,
## если в сцене есть AnimationPlayer, играет walk/attack/death/idle); полоска и кольцо остаются.
## Ось модели: лицо в +Z (как у героини), рост 1.7 м, позиция — точка на земле (y = 0).
##
## ИИ: IDLE (стоит) -> CHASE при приближении героини ближе AGGRO_RANGE; путь через world.find_path
## (пересчёт каждые REPATH_INTERVAL); в досягаемости ATTACK: взмах, урон в STRIKE_DELAY, кулдаун.

signal died(enemy)
signal hp_changed(enemy, hp, max_hp)

enum State { IDLE, CHASE, ATTACK, DEAD }

## Попадание лучом: радиус вокруг вертикальной оси врага, высота оси.
const PICK_RADIUS := 0.7
const PICK_HEIGHT := 1.8
const AGGRO_RANGE := 5.0
const STAND_REACH := 1.0
const REPATH_INTERVAL := 0.5
const STRIKE_DELAY := 0.25
const SWING_TIME := 0.5
const FALL_TIME := 0.6
const WALK_RATE := 7.0
const WALK_AMP := 0.5
const TURN_RATE := 12.0
const STRIKE_SLACK := 0.3
const FLASH_TIME := 0.12
const BAR_Y := 2.0
const BAR_W := 0.9
const BAR_H := 0.12
const BAR_SELECTED_SCALE := 1.35
const RING_INNER := 0.5
const RING_OUTER := 0.66

const COL_BONE := Color(0.87, 0.84, 0.75)
const COL_BONE_DARK := Color(0.68, 0.64, 0.55)
const COL_HOLLOW := Color(0.04, 0.03, 0.03)
const COL_RUST := Color(0.5, 0.3, 0.16)
const COL_RING := Color(0.55, 0.04, 0.05, 0.9)
const COL_BAR_BG := Color(0.06, 0.02, 0.02, 0.9)
const COL_BAR_FILL := Color(0.8, 0.08, 0.08)
const COL_BAR_FLASH := Color(1.0, 0.95, 0.9)

var hp := 30
var max_hp := 30
var speed := 1.8
var attack_range := 1.2
var attack_damage := 6
var attack_cooldown := 1.4
var state: int = State.IDLE
## Горизонтальная скорость (м/с) — для камеры (как у героини).
var velocity := Vector3.ZERO
var selected := false
## Назначаются через setup() (world3d.gd и hero3d.gd).
var world = null
var hero = null

var _path := PackedVector3Array()
var _repath := 0.0
var _cooldown := 0.0
var _strike_in := -1.0
var _swing_left := 0.0
var _moving := false
var _phase := 0.0
var _target_yaw := 0.0
var _flash := 0.0
var _procedural := true
var _anim_player: AnimationPlayer = null
var _yaw_node: Node3D = null
var _body: Node3D = null
var _torso: Node3D = null
var _arm_r: Node3D = null
var _arm_l: Node3D = null
var _leg_r: Node3D = null
var _leg_l: Node3D = null
var _ring: MeshInstance3D = null
var _bar_root: Node3D = null
var _fill: MeshInstance3D = null
var _fill_quad: QuadMesh = null
var _fill_mat: StandardMaterial3D = null
var _swing_tween: Tween = null
var _fall_tween: Tween = null


func _ready() -> void:
	_yaw_node = Node3D.new()
	_yaw_node.name = "Yaw"
	add_child(_yaw_node)
	_body = Node3D.new()
	_body.name = "Body"
	_yaw_node.add_child(_body)
	_build_skeleton()
	_build_ring()
	_build_bar()


func setup(p_world, p_hero) -> void:
	world = p_world
	hero = p_hero


func _process(delta: float) -> void:
	if _flash > 0.0:
		_flash -= delta
	if state == State.DEAD:
		return
	_update_ai(delta)
	_update_visual(delta)
	_update_bar()
	_yaw_node.rotation.y = lerp_angle(_yaw_node.rotation.y, _target_yaw, 1.0 - exp(-TURN_RATE * delta))


## Попадание лучом камеры: расстояние по лучу (>= 0) или -1, если мимо / мёртв.
## Ближайшая точка луча к оси врага; промах, если расстояние до оси больше PICK_RADIUS.
func pick_distance(origin: Vector3, dir: Vector3) -> float:
	if state == State.DEAD:
		return -1.0
	var a := Vector3(position.x, 0.0, position.z)
	var u := Vector3.UP * PICK_HEIGHT
	var w0 := origin - a
	var b := dir.dot(u)
	var c := u.dot(u)
	var d := dir.dot(w0)
	var ec := u.dot(w0)
	var denom := c - b * b
	var s := 0.5
	if denom > 0.000001:
		s = clampf((ec - b * d) / denom, 0.0, 1.0)
	var q := a + u * s
	var t := maxf(dir.dot(q - origin), 0.0)
	var closest := origin + dir * t
	if closest.distance_to(q) > PICK_RADIUS:
		return -1.0
	return t


## Точка для прицела и полоски (грудь).
func hit_point() -> Vector3:
	return position + Vector3(0.0, 1.0, 0.0)


func is_alive() -> bool:
	return state != State.DEAD


func set_selected(on: bool) -> void:
	selected = on and is_alive()
	if _ring != null:
		_ring.visible = selected


## Урон. При hp <= 0 — смерть. Удар по бездействующему врагу поднимает его в погоню.
func take_damage(amount: int) -> void:
	if state == State.DEAD or amount <= 0:
		return
	hp = maxi(hp - amount, 0)
	_flash = FLASH_TIME
	_refresh_bar()
	hp_changed.emit(self, hp, max_hp)
	if hp <= 0:
		_die()
	elif state == State.IDLE:
		state = State.CHASE


## Подмена вида: заменяет тело (всё под _body) на готовую сцену. Полоска и кольцо остаются.
func set_visual_scene(scene: Node3D) -> void:
	if _swing_tween != null:
		_swing_tween.kill()
	for c in _body.get_children():
		c.queue_free()
	_body.add_child(scene)
	_procedural = false
	_torso = null
	_arm_r = null
	_arm_l = null
	_leg_r = null
	_leg_l = null
	_anim_player = _find_anim_player(scene)


# ------------------------------------------------------------------ ИИ

func _update_ai(delta: float) -> void:
	if hero == null or world == null:
		return
	if _cooldown > 0.0:
		_cooldown -= delta
	if state == State.ATTACK:
		_moving = false
		velocity = Vector3.ZERO
		_face(hero.position)
		if _strike_in > 0.0:
			_strike_in -= delta
			if _strike_in <= 0.0:
				_strike_in = -1.0
				_strike()
		_swing_left -= delta
		if _swing_left <= 0.0:
			state = State.CHASE
		return
	if hero.state_name() == "dead":
		state = State.IDLE
		_moving = false
		velocity = Vector3.ZERO
		return
	var dist: float = Combat.hdist(position, hero.position)
	if state == State.IDLE:
		if dist > AGGRO_RANGE:
			return
		state = State.CHASE
	if dist <= attack_range:
		_moving = false
		velocity = Vector3.ZERO
		_face(hero.position)
		if _cooldown <= 0.0:
			_start_swing()
		return
	_repath -= delta
	if _repath <= 0.0:
		_repath = REPATH_INTERVAL
		_plan()
	_walk(delta)


## Путь к клетке у героини (на расстоянии удара). Пустой путь — стоим до следующего пересчёта.
func _plan() -> void:
	_path = PackedVector3Array()
	var cell: Vector2i = Combat.stand_cell(world, hero.position, STAND_REACH, position)
	if cell.x < 0:
		return
	var wp: Vector3 = world.grid_to_world(cell)
	var pts: PackedVector3Array = world.find_path(position, wp)
	pts = Combat.trim_passed(position, pts)
	_path = pts


func _walk(delta: float) -> void:
	var remaining := speed * delta
	_moving = false
	while remaining > 0.0 and not _path.is_empty():
		var to_wp: Vector3 = _path[0] - position
		to_wp.y = 0.0
		var d := to_wp.length()
		if d > 0.0001:
			_face(to_wp)
		if d <= remaining:
			position = Vector3(_path[0].x, 0.0, _path[0].z)
			remaining -= d
			_path.remove_at(0)
		else:
			position += to_wp / d * remaining
			remaining = 0.0
		_moving = true
	position.y = 0.0
	velocity = Vector3.ZERO
	if _moving and not _path.is_empty():
		var dir: Vector3 = _path[0] - position
		dir.y = 0.0
		if dir.length_squared() > 0.0001:
			velocity = dir.normalized() * speed


func _start_swing() -> void:
	state = State.ATTACK
	_cooldown = attack_cooldown
	_strike_in = STRIKE_DELAY
	_swing_left = SWING_TIME
	_path = PackedVector3Array()
	_moving = false
	velocity = Vector3.ZERO
	_face(hero.position)
	_play_anim("attack")
	if _procedural:
		_swing_procedural()


func _strike() -> void:
	if hero.state_name() == "dead":
		return
	if Combat.hdist(position, hero.position) <= attack_range + STRIKE_SLACK:
		hero.take_damage(attack_damage)


func _die() -> void:
	state = State.DEAD
	_path = PackedVector3Array()
	velocity = Vector3.ZERO
	_moving = false
	selected = false
	if _ring != null:
		_ring.visible = false
	if _bar_root != null:
		_bar_root.visible = false
	if _swing_tween != null:
		_swing_tween.kill()
	if _procedural:
		_fall_tween = create_tween()
		_fall_tween.tween_property(_body, "rotation:x", -PI * 0.5, FALL_TIME) \
			.set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_IN)
	else:
		_play_anim("death")
	died.emit(self)


func _face(p: Vector3) -> void:
	var d := p - position
	d.y = 0.0
	if d.length_squared() > 0.0001:
		_target_yaw = atan2(d.x, d.z)


# ------------------------------------------------------------------ анимации

func _swing_procedural() -> void:
	if _swing_tween != null:
		_swing_tween.kill()
	_swing_tween = create_tween()
	_swing_tween.tween_property(_arm_r, "rotation:x", -2.6, 0.15) \
		.set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_OUT)
	_swing_tween.tween_property(_arm_r, "rotation:x", -0.4, 0.1) \
		.set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_IN)
	_swing_tween.tween_property(_arm_r, "rotation:x", 0.0, 0.25) \
		.set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_OUT)


func _update_visual(delta: float) -> void:
	if not _procedural:
		if _anim_player != null:
			var key := "walk" if _moving else "idle"
			if state == State.ATTACK:
				key = "attack"
			_play_anim(key)
		return
	if _moving:
		_phase += delta * WALK_RATE
	var amp := WALK_AMP if _moving else 0.0
	_leg_r.rotation.x = sin(_phase) * amp
	_leg_l.rotation.x = -sin(_phase) * amp
	if state != State.ATTACK:
		_arm_l.rotation.x = -sin(_phase) * amp * 0.8


func _play_anim(key: String) -> void:
	if _anim_player == null or not _anim_player.has_animation(key):
		return
	if _anim_player.current_animation != key:
		_anim_player.play(key, 0.1)


func _find_anim_player(n: Node) -> AnimationPlayer:
	if n is AnimationPlayer:
		return n as AnimationPlayer
	for c in n.get_children():
		var f := _find_anim_player(c)
		if f != null:
			return f
	return null


# ------------------------------------------------------------------ полоска и кольцо

## Каждый кадр: масштаб выделения и вспышка. Ширина заполнения — в _refresh_bar (не через scale:
## billboard-шейдер Godot игнорирует масштаб самого меша).
func _update_bar() -> void:
	if _bar_root == null or not is_alive():
		return
	_bar_root.scale = Vector3.ONE * (BAR_SELECTED_SCALE if selected else 1.0)
	_fill_mat.albedo_color = COL_BAR_FLASH if _flash > 0.0 else COL_BAR_FILL


## Заполнение: левый край прижат к левому краю фона, ширина = доля HP.
## Сдвиг задаётся внутри меша (center_offset), а не позицией узла: позиция по мировой X в изометрии
## уходит по диагонали на экране, а billboard поворачивает меш к камере уже вместе со сдвигом.
func _refresh_bar() -> void:
	if _fill == null:
		return
	var frac := clampf(float(hp) / float(max_hp), 0.0, 1.0)
	_fill_quad.size = Vector2(BAR_W * maxf(frac, 0.001), BAR_H)
	_fill_quad.center_offset = Vector3(-BAR_W * 0.5 + BAR_W * frac * 0.5, 0.0, 0.0)


func _build_ring() -> void:
	var torus := TorusMesh.new()
	torus.inner_radius = RING_INNER
	torus.outer_radius = RING_OUTER
	torus.rings = 24
	torus.ring_segments = 12
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.albedo_color = COL_RING
	_ring = MeshInstance3D.new()
	_ring.name = "SelectRing"
	_ring.mesh = torus
	_ring.material_override = mat
	_ring.position = Vector3(0.0, 0.03, 0.0)
	_ring.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_ring.visible = false
	add_child(_ring)


func _build_bar() -> void:
	_bar_root = Node3D.new()
	_bar_root.name = "HealthBar"
	_bar_root.position = Vector3(0.0, BAR_Y, 0.0)
	add_child(_bar_root)
	var quad := QuadMesh.new()
	quad.size = Vector2(BAR_W, BAR_H)
	var bg := MeshInstance3D.new()
	bg.name = "Back"
	bg.mesh = quad
	bg.material_override = _bar_mat(COL_BAR_BG, 0)
	_bar_root.add_child(bg)
	_fill_mat = _bar_mat(COL_BAR_FILL, 1)
	_fill_quad = QuadMesh.new()
	_fill = MeshInstance3D.new()
	_fill.name = "Fill"
	_fill.mesh = _fill_quad
	_fill.material_override = _fill_mat
	_bar_root.add_child(_fill)
	_refresh_bar()


## Полоска всегда лицом к камере и видна сквозь тело (no_depth_test).
func _bar_mat(col: Color, prio: int) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.billboard_mode = BaseMaterial3D.BILLBOARD_ENABLED
	m.no_depth_test = true
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.albedo_color = col
	m.render_priority = prio
	return m


# ------------------------------------------------------------------ скелет из примитивов

func _build_skeleton() -> void:
	var bone := _mat(COL_BONE)
	var bone_dark := _mat(COL_BONE_DARK)
	var hollow := _mat(COL_HOLLOW)
	var rust := _mat(COL_RUST, 0.6)

	# Таз и ноги (пивоты ног — в бедрах, высота 0.85 м).
	_part(_body, _box(Vector3(0.34, 0.12, 0.2)), Vector3(0.0, 0.85, 0.0), bone_dark)
	_leg_r = _pivot(_body, Vector3(-0.11, 0.85, 0.0))
	_leg_l = _pivot(_body, Vector3(0.11, 0.85, 0.0))
	for leg in [_leg_r, _leg_l]:
		_part(leg, _cyl(0.055, 0.055, 0.85), Vector3(0.0, -0.425, 0.0), bone)
		_part(leg, _box(Vector3(0.12, 0.05, 0.22)), Vector3(0.0, -0.825, 0.04), bone_dark)

	# Торс (пивот в бедрах): позвоночник, рёбра, ключицы, шея, череп.
	_torso = _pivot(_body, Vector3(0.0, 0.85, 0.0))
	_part(_torso, _cyl(0.04, 0.04, 0.5), Vector3(0.0, 0.3, 0.0), bone)
	for y in [0.16, 0.26, 0.36, 0.46]:
		_part(_torso, _cyl(0.016, 0.016, 0.3), Vector3(0.0, y, 0.02), bone, Vector3(0.0, 0.0, 90.0))
	_part(_torso, _box(Vector3(0.44, 0.04, 0.06)), Vector3(0.0, 0.5, 0.0), bone_dark)
	_part(_torso, _cyl(0.04, 0.04, 0.08), Vector3(0.0, 0.59, 0.0), bone)
	_part(_torso, _sphere(0.14), Vector3(0.0, 0.71, 0.0), bone)
	_part(_torso, _box(Vector3(0.15, 0.05, 0.12)), Vector3(0.0, 0.57, 0.05), bone_dark)
	_part(_torso, _sphere(0.038), Vector3(-0.055, 0.73, 0.12), hollow)
	_part(_torso, _sphere(0.038), Vector3(0.055, 0.73, 0.12), hollow)
	_part(_torso, _box(Vector3(0.025, 0.04, 0.02)), Vector3(0.0, 0.68, 0.135), hollow)
	_part(_torso, _box(Vector3(0.11, 0.02, 0.03)), Vector3(0.0, 0.6, 0.1), hollow)

	# Правая рука с ржавой саблей (вниз вдоль руки, клинок 0.7 м).
	_arm_r = _pivot(_torso, Vector3(-0.24, 0.51, 0.0))
	_part(_arm_r, _cyl(0.04, 0.04, 0.58), Vector3(0.0, -0.29, 0.0), bone_dark)
	_part(_arm_r, _sphere(0.045), Vector3(0.0, -0.6, 0.0), bone)
	_part(_arm_r, _box(Vector3(0.12, 0.02, 0.035)), Vector3(0.0, -0.6, 0.0), rust)
	_part(_arm_r, _box(Vector3(0.035, 0.7, 0.012)), Vector3(0.0, -0.95, 0.0), rust, Vector3(0.0, 0.0, 6.0))

	# Левая рука.
	_arm_l = _pivot(_torso, Vector3(0.24, 0.51, 0.0))
	_part(_arm_l, _cyl(0.04, 0.04, 0.58), Vector3(0.0, -0.29, 0.0), bone_dark)
	_part(_arm_l, _sphere(0.045), Vector3(0.0, -0.6, 0.0), bone)


func _pivot(parent: Node3D, pos: Vector3) -> Node3D:
	var n := Node3D.new()
	n.position = pos
	parent.add_child(n)
	return n


func _part(parent: Node3D, mesh: Mesh, pos: Vector3, mat: Material, rot_deg: Vector3 = Vector3.ZERO) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.material_override = mat
	mi.position = pos
	mi.rotation_degrees = rot_deg
	parent.add_child(mi)
	return mi


func _mat(col: Color, rough: float = 0.9) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = col
	m.roughness = rough
	return m


func _cyl(top: float, bottom: float, h: float) -> CylinderMesh:
	var c := CylinderMesh.new()
	c.top_radius = top
	c.bottom_radius = bottom
	c.height = h
	c.radial_segments = 8
	c.rings = 1
	return c


func _box(size: Vector3) -> BoxMesh:
	var b := BoxMesh.new()
	b.size = size
	return b


func _sphere(r: float) -> SphereMesh:
	var s := SphereMesh.new()
	s.radius = r
	s.height = r * 2.0
	s.radial_segments = 8
	s.rings = 4
	return s
