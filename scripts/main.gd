extends Node3D
## Корень среза (3D): мир, героиня, враги, бой, камера, HUD.
## Тап по врагу -> цель (бой). Тап по земле -> цель сбрасывается, героиня идёт.
## Режимы (аргументы после --):
##   --autotest           бой: тап по ближайшему врагу, авто-переключение, навык; AUTOTEST3D OK / FAIL
##   --shots=DIR          кадры боя (3d_combat_approach/hit/skill, 3d_enemy_close), затем 3d_idle ... 3d_hand_attack_b
##   --screenshot=PATH    один кадр через секунду

const START_CELL := Vector2i(18, 18)
const SUN_ROT_DEG := Vector3(-55.0, 35.0, 0.0)
const HUD_FADE_AT := 4.0
const AUTOTEST_START_DELAY := 0.5
const AUTOTEST_TIMEOUT := 40.0
const AUTOTEST_SKILL_APPROACH := 2.5
const SHOT_IDLE_DELAY := 1.0
const SHOT_WALK_DELAY := 0.5
const SHOT_WALK_CELLS := 5
const SHOT_ATTACK_DELAY := 0.3
const SHOT_DEATH_DELAY := 1.4
const SHOT_CLOSE_ZOOM := 3.0
const SHOT_DEFAULT_ZOOM := 9.0
## Крупный план руки: видимая высота 1.6 м, фокус на груди героини.
const SHOT_HAND_ZOOM := 1.6
## Крупный план врага (видимая высота, м).
const SHOT_ENEMY_ZOOM := 2.6
## Кадр подхода: герой идёт, цель в этом радиусе.
const SHOT_APPROACH_DIST := 4.0
const SHOT_HIT_DELAY := 0.08
## Моменты удара для крупного плана (секунды от начала удара).
const SHOT_HAND_ATTACK_A := 0.2
const SHOT_HAND_ATTACK_B := 0.5
const SHOT_SETTLE := 1.2
const SHOT_WAIT_TIMEOUT := 15.0
const SINGLE_SHOT_DELAY := 1.0
const SKILL_BASE_MODULATE := Color(1.0, 1.0, 1.0, 0.85)
const SKILL_FAIL_MODULATE := Color(1.0, 0.45, 0.45, 0.85)
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
@onready var combat: Combat = $Combat
@onready var sun: DirectionalLight3D = $Sun
@onready var hint: Label = $HUD/Hint
@onready var hero_bar: ProgressBar = $HUD/HeroHP
@onready var kills_label: Label = $HUD/Kills
@onready var skill_button: Button = $HUD/Controls/SkillButton
@onready var skill_bar: ProgressBar = $HUD/Controls/SkillButton/CooldownBar

var _autotest := false
var _shots_dir := ""
var _screenshot_path := ""
var _autotest_done := false
var _at_kill_seen := false
var _at_switch_ok := false
var _at_expected: Enemy3D = null


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
	hero.hp_changed.connect(_on_hero_hp)
	_on_hero_hp(hero.hp, hero.max_hp)

	camera.target = hero
	camera.set_bounds(world.get_ground_rect())

	combat.setup(world, hero)
	combat.spawn_enemies(hero.position)
	combat.enemy_killed.connect(_on_enemy_killed)
	_on_enemy_killed(null, null)

	skill_button.pressed.connect(_on_skill_pressed)

	_run_hud_fade()

	if _autotest:
		_run_autotest()
	elif _shots_dir != "":
		_run_shots(_shots_dir)
	elif _screenshot_path != "":
		_run_single_shot(_screenshot_path)


func _process(_delta: float) -> void:
	skill_bar.value = combat.skill_ready_fraction() * 100.0


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.pressed and mb.button_index == MOUSE_BUTTON_LEFT:
			if _handle_tap(mb.position):
				get_viewport().set_input_as_handled()


## Один путь для тапа (и для касания, и для автотеста): враг под лучом -> цель;
## иначе земля -> цель сбрасывается, героиня идёт. true, если тап что-то выбрал.
func _handle_tap(screen_pos: Vector2) -> bool:
	var origin: Vector3 = camera.project_ray_origin(screen_pos)
	var dir: Vector3 = camera.project_ray_normal(screen_pos)
	var enemy: Enemy3D = combat.pick_enemy(origin, dir)
	if enemy != null:
		combat.select_target(enemy)
		return true
	var ground = _screen_to_ground(screen_pos)
	if ground == null:
		return false
	combat.clear_target()
	hero.move_to(ground)
	return true


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


# ------------------------------------------------------------------ HUD

func _on_hero_hp(cur: int, mx: int) -> void:
	hero_bar.max_value = mx
	hero_bar.value = cur


func _on_enemy_killed(_e, _next) -> void:
	kills_label.text = "Убито: %d" % combat.kills


func _on_skill_pressed() -> void:
	if not combat.use_skill():
		_blink_skill()


## Не сработал (нет цели в 3 м или кулдаун): кнопка мигает красным.
func _blink_skill() -> void:
	var tw := create_tween()
	tw.tween_property(skill_button, "modulate", SKILL_FAIL_MODULATE, 0.1)
	tw.tween_property(skill_button, "modulate", SKILL_BASE_MODULATE, 0.2)


# ------------------------------------------------------------------ --autotest

func _run_autotest() -> void:
	get_tree().create_timer(AUTOTEST_TIMEOUT).timeout.connect(_autotest_timeout)
	await _wait_seconds(AUTOTEST_START_DELAY)

	# 1. Тап по ближайшему врагу в его точке прицела: проверяем луч и выбор цели.
	var first: Enemy3D = combat.nearest_enemy(hero.position)
	if first == null:
		_autotest_fail("no enemies")
		return
	var sp: Vector2 = camera.unproject_position(first.hit_point())
	_handle_tap(sp)
	if combat.target != first:
		_autotest_fail("tap on enemy did not select it (ray pick)")
		return
	print("AUTOTEST3D tap enemy dist=%.2f selected" % Combat.hdist(hero.position, first.position))

	# 2. Герои подходит и убивает цель авто-ударами; при смерти — авто-переключение.
	combat.enemy_killed.connect(_at_on_kill)
	await _wait_until(func(): return _at_kill_seen)
	if not _at_kill_seen:
		_autotest_fail("first enemy not killed")
		return
	if not _at_switch_ok:
		_autotest_fail("auto-switch wrong: target=%s expected=%s" % [combat.target, _at_expected])
		return
	print("AUTOTEST3D kill1 ok auto-switch=%s" % ("yes" if _at_expected != null else "none nearby"))

	if combat.target != null:
		var hits0 := combat.hero_hits
		await _wait_until(func(): return combat.hero_hits > hits0)
		if combat.hero_hits <= hits0:
			_autotest_fail("no hit on the auto-selected target")
			return
		print("AUTOTEST3D hit auto-target ok")

	# 3. Навык: подойти к ближайшему врагу на 2.5 м и применить «Выпад».
	var n = await _approach_nearest(AUTOTEST_SKILL_APPROACH)
	if n == null:
		_autotest_fail("no enemy to use skill on")
		return
	if not combat.use_skill():
		_autotest_fail("skill not used (cooldown or no enemy in 3 m)")
		return
	var kills0 := combat.kills
	await _wait_until(func(): return not combat.skill_pending)
	if combat.kills <= kills0:
		_autotest_fail("skill strike did not kill the target")
		return
	print("AUTOTEST3D skill ok dmg=%d" % Combat.SKILL_DAMAGE)

	_autotest_done = true
	print("AUTOTEST3D OK combat kills=%d hero_hp=%d" % [combat.kills, hero.hp])
	get_tree().quit(0)


## Проверка в момент смерти первой цели (сигнал идёт уже после переключения).
func _at_on_kill(_e, _next) -> void:
	if _at_kill_seen:
		return
	_at_kill_seen = true
	_at_expected = combat.nearest_enemy(hero.position, Combat.AUTO_SWITCH_RADIUS)
	if _at_expected == null:
		_at_switch_ok = combat.target == null
	else:
		_at_switch_ok = combat.target == _at_expected


func _autotest_fail(reason: String) -> void:
	if _autotest_done:
		return
	_autotest_done = true
	print("AUTOTEST3D FAIL %s" % reason)
	get_tree().quit(1)


func _autotest_timeout() -> void:
	_autotest_fail("timeout hero=(%.2f, %.2f) state=%s kills=%d" % [hero.position.x, hero.position.z, hero.state_name(), combat.kills])


## Тап по ближайшему врагу, ждём подхода на dist_max. Возвращает врага или null.
func _approach_nearest(dist_max: float) -> Enemy3D:
	for _i in 10:
		var n: Enemy3D = combat.nearest_enemy(hero.position)
		if n == null:
			return null
		_handle_tap(camera.unproject_position(n.hit_point()))
		var nn: Enemy3D = n
		await _wait_until(func(): return not nn.is_alive() or Combat.hdist(hero.position, nn.position) <= dist_max)
		if nn.is_alive() and Combat.hdist(hero.position, nn.position) <= dist_max:
			return nn
	return null


# ------------------------------------------------------------------ --shots / --screenshot

func _run_shots(dir: String) -> void:
	DirAccess.make_dir_recursive_absolute(dir)
	hero.face_direction(Vector3(1.0, 0.0, 1.0))
	await _run_combat_shots(dir)
	combat.cancel()
	hero.stop()
	await _wait_until(func(): return hero.state_name() != "attack")

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


## Кадры боя: подход, удар, навык, крупный план врага с полоской и кольцом.
func _run_combat_shots(dir: String) -> void:
	await _wait_seconds(SHOT_IDLE_DELAY)
	var first: Enemy3D = combat.nearest_enemy(hero.position)
	if first == null:
		print("SHOT combat: no enemies")
		return
	_handle_tap(camera.unproject_position(first.hit_point()))
	var f: Enemy3D = first
	await _wait_until(func(): return not f.is_alive() or (hero.state_name() == "walk" and Combat.hdist(hero.position, f.position) <= SHOT_APPROACH_DIST))
	await _capture(dir.path_join("3d_combat_approach.png"))

	var hits0 := combat.hero_hits
	await _wait_until(func(): return combat.hero_hits > hits0)
	await _wait_seconds(SHOT_HIT_DELAY)
	await _capture(dir.path_join("3d_combat_hit.png"))

	var n: Enemy3D = await _approach_nearest(AUTOTEST_SKILL_APPROACH)
	if n != null and combat.use_skill():
		await _wait_until(func(): return not combat.skill_pending)
		await _wait_seconds(SHOT_HIT_DELAY)
		await _capture(dir.path_join("3d_combat_skill.png"))
	else:
		print("SHOT combat: skill not used")

	var close: Enemy3D = combat.nearest_enemy(hero.position)
	if close != null:
		combat.select_target(close)
		camera.target = close
		camera.set_zoom(SHOT_ENEMY_ZOOM)
		await _wait_seconds(SHOT_SETTLE)
		await _capture(dir.path_join("3d_enemy_close.png"))
		camera.target = hero
		camera.set_zoom(SHOT_DEFAULT_ZOOM)
		await _wait_seconds(SHOT_SETTLE)


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
