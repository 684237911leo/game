extends Node3D
## Корень среза (3D): мир, героиня, камера, HUD. Тап по земле -> героиня идёт.
## Режимы (аргументы после --):
##   --autotest           тап в 6 м -> удар -> смерть; печатает AUTOTEST3D OK или AUTOTEST3D FAIL
##   --shots=DIR          серия кадров в DIR (3d_idle, 3d_walk_E/N/W/S, 3d_attack, 3d_death, 3d_close)
##   --screenshot=PATH    один кадр через секунду

const START_CELL := Vector2i(18, 18)
const SUN_ROT_DEG := Vector3(-55.0, 35.0, 0.0)
const HUD_FADE_AT := 4.0
const AUTOTEST_START_DELAY := 0.5
const AUTOTEST_TIMEOUT := 25.0
const AUTOTEST_TAP_CELLS := 6
const SHOT_IDLE_DELAY := 1.0
const SHOT_WALK_DELAY := 0.5
const SHOT_WALK_CELLS := 5
const SHOT_ATTACK_DELAY := 0.3
const SHOT_DEATH_DELAY := 1.4
const SHOT_CLOSE_ZOOM := 3.0
## Крупный план руки: видимая высота 1.6 м, фокус на груди героини.
const SHOT_HAND_ZOOM := 1.6
## Моменты удара для крупного плана (секунды от начала удара).
const SHOT_HAND_ATTACK_A := 0.2
const SHOT_HAND_ATTACK_B := 0.5
const SHOT_SETTLE := 1.2
const SHOT_WAIT_TIMEOUT := 15.0
const SINGLE_SHOT_DELAY := 1.0
## Направления для кадров ходьбы (по осям мира): E = +X, N = -Z, W = -X, S = +Z.
const SHOT_DIRS := {
	"E": Vector2i(1, 0),
	"N": Vector2i(0, -1),
	"W": Vector2i(-1, 0),
	"S": Vector2i(0, 1),
}

@onready var world = $World
@onready var hero = $Hero
@onready var camera = $FollowCamera
@onready var sun: DirectionalLight3D = $Sun
@onready var hint: Label = $HUD/Hint
@onready var attack_button: Button = $HUD/Controls/AttackButton
@onready var death_button: Button = $HUD/Controls/DeathButton

var _autotest := false
var _shots_dir := ""
var _screenshot_path := ""
var _autotest_target := Vector2i.ZERO
## 0: идём к цели, 1: удар, 2: смерть, 3: завершено.
var _autotest_phase := 0


func _ready() -> void:
	for a in OS.get_cmdline_user_args():
		if a == "--autotest":
			_autotest = true
		elif a.begins_with("--shots="):
			_shots_dir = a.trim_prefix("--shots=")
		elif a.begins_with("--screenshot="):
			_screenshot_path = a.trim_prefix("--screenshot=")

	sun.rotation_degrees = SUN_ROT_DEG

	hero.world = world
	hero.position = world.grid_to_world(START_CELL)
	# Лицом к камере (камера стоит в направлении +X +Z).
	hero.face_direction(Vector3(1.0, 0.0, 1.0))
	hero.arrived.connect(_on_hero_arrived)
	hero.attack_finished.connect(_on_attack_finished)
	hero.death_finished.connect(_on_death_finished)

	# Тестовые кнопки: Button съедает ввод, поэтому героиня по ним не идёт.
	attack_button.pressed.connect(hero.attack)
	death_button.pressed.connect(hero.die)

	camera.target = hero
	camera.set_bounds(world.get_ground_rect())

	_run_hud_fade()

	if _autotest:
		_run_autotest()
	elif _shots_dir != "":
		_run_shots(_shots_dir)
	elif _screenshot_path != "":
		_run_single_shot(_screenshot_path)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.pressed and mb.button_index == MOUSE_BUTTON_LEFT:
			var ground = _screen_to_ground(mb.position)
			if ground != null:
				hero.move_to(ground)
				get_viewport().set_input_as_handled()


## Луч от камеры через точку экрана -> пересечение с плоскостью y = 0.
func _screen_to_ground(screen_pos: Vector2):
	var origin: Vector3 = camera.project_ray_origin(screen_pos)
	var dir: Vector3 = camera.project_ray_normal(screen_pos)
	if absf(dir.y) < 0.0001:
		return null
	var t: float = -origin.y / dir.y
	if t < 0.0:
		return null
	return origin + dir * t


func _run_hud_fade() -> void:
	var tw := create_tween()
	tw.tween_interval(HUD_FADE_AT)
	tw.tween_property(hint, "modulate:a", 0.0, 0.8)
	tw.tween_callback(hint.hide)


# ------------------------------------------------------------------ --autotest

func _run_autotest() -> void:
	get_tree().create_timer(AUTOTEST_TIMEOUT).timeout.connect(_autotest_timeout)
	await get_tree().create_timer(AUTOTEST_START_DELAY).timeout
	var from_cell: Vector2i = world.world_to_grid(hero.position)
	_autotest_target = from_cell + Vector2i(AUTOTEST_TAP_CELLS, 0)
	print("AUTOTEST3D tap from=%s target=%s" % [from_cell, _autotest_target])
	hero.move_to(world.grid_to_world(_autotest_target))


func _on_hero_arrived() -> void:
	if not _autotest or _autotest_phase != 0:
		return
	var cell: Vector2i = world.world_to_grid(hero.position)
	var expected: Vector2i = world.nearest_free_cell(_autotest_target)
	var center: Vector3 = world.grid_to_world(cell)
	var on_center: bool = hero.position.distance_to(center) < 0.05
	if cell != expected or not on_center:
		_autotest_fail("arrive pos=(%.2f, %.2f) cell=%s expected=%s" % [hero.position.x, hero.position.z, cell, expected])
		return
	print("AUTOTEST3D arrived pos=(%.2f, %.2f) cell=%s -> attack" % [hero.position.x, hero.position.z, cell])
	_autotest_phase = 1
	hero.attack()


func _on_attack_finished() -> void:
	if not _autotest or _autotest_phase != 1:
		return
	print("AUTOTEST3D attack done -> die")
	_autotest_phase = 2
	hero.die()


func _on_death_finished() -> void:
	if not _autotest or _autotest_phase != 2:
		return
	_autotest_phase = 3
	print("AUTOTEST3D OK walk+attack+death pos=(%.2f, %.2f) state=%s" % [hero.position.x, hero.position.z, hero.state_name()])
	get_tree().quit(0)


func _autotest_fail(reason: String) -> void:
	if _autotest_phase == 3:
		return
	_autotest_phase = 3
	print("AUTOTEST3D FAIL %s" % reason)
	get_tree().quit(1)


func _autotest_timeout() -> void:
	_autotest_fail("timeout phase=%d pos=(%.2f, %.2f) state=%s" % [_autotest_phase, hero.position.x, hero.position.z, hero.state_name()])


# ------------------------------------------------------------------ --shots / --screenshot

func _run_shots(dir: String) -> void:
	DirAccess.make_dir_recursive_absolute(dir)
	hero.face_direction(Vector3(1.0, 0.0, 1.0))
	await _wait_seconds(SHOT_IDLE_DELAY)
	await _capture(dir.path_join("3d_idle.png"))

	for key in ["E", "N", "W", "S"]:
		var from_cell: Vector2i = world.world_to_grid(hero.position)
		hero.move_to(world.grid_to_world(from_cell + SHOT_DIRS[key] * SHOT_WALK_CELLS))
		await _wait_seconds(SHOT_WALK_DELAY)
		await _capture(dir.path_join("3d_walk_%s.png" % key))
		await _wait_until(func(): return hero.state_name() != "walk")

	hero.attack()
	await _wait_seconds(SHOT_ATTACK_DELAY)
	await _capture(dir.path_join("3d_attack.png"))
	await _wait_until(func(): return hero.state_name() != "attack")

	hero.die()
	await _wait_seconds(SHOT_DEATH_DELAY)
	await _capture(dir.path_join("3d_death.png"))
	# Ждём воскрешения (конец Death + REVIVE_DELAY), затем крупный план живой героини.
	await _wait_until(func(): return hero.state_name() == "idle")

	camera.set_zoom(SHOT_CLOSE_ZOOM)
	await _wait_seconds(SHOT_SETTLE)
	await _capture(dir.path_join("3d_close.png"))

	# Крупный план руки со шпагой: покой и два момента удара.
	camera.set_zoom(SHOT_HAND_ZOOM)
	await _wait_seconds(SHOT_SETTLE)
	await _capture(dir.path_join("3d_hand_idle.png"))
	hero.attack()
	await _wait_seconds(SHOT_HAND_ATTACK_A)
	await _capture(dir.path_join("3d_hand_attack_a.png"))
	await _wait_seconds(SHOT_HAND_ATTACK_B - SHOT_HAND_ATTACK_A)
	await _capture(dir.path_join("3d_hand_attack_b.png"))
	await _wait_until(func(): return hero.state_name() != "attack")
	print("SHOTS DONE dir=%s" % dir)
	get_tree().quit(0)


func _run_single_shot(path: String) -> void:
	await _wait_seconds(SINGLE_SHOT_DELAY)
	await _capture(path)
	get_tree().quit(0)


func _wait_seconds(s: float) -> void:
	await get_tree().create_timer(s).timeout


func _wait_until(cond: Callable) -> void:
	var waited := 0.0
	while not cond.call():
		if waited > SHOT_WAIT_TIMEOUT:
			print("SHOT WAIT TIMEOUT")
			return
		await get_tree().process_frame
		waited += get_process_delta_time()


## Сохраняет кадр корневого viewport в PNG (после отрисовки кадра).
func _capture(path: String) -> void:
	await RenderingServer.frame_post_draw
	var img := get_viewport().get_texture().get_image()
	if img == null:
		print("SCREENSHOT FAIL: no image (renderer unavailable?)")
		return
	DirAccess.make_dir_recursive_absolute(path.get_base_dir())
	var err: int = img.save_png(path)
	print("SCREENSHOT %s err=%d size=%dx%d" % [path, err, img.get_width(), img.get_height()])
