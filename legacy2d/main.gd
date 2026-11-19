extends Node2D
## Корень среза: собирает карту, героиню, камеру, HUD; режим --autotest и --screenshot=путь.

const START_CELL := Vector2i(14, 14)
const BG_COLOR := Color(0.05, 0.05, 0.04)
const HUD_FADE_AT := 4.0
const AUTOTEST_START_DELAY := 0.5
const AUTOTEST_TIMEOUT := 20.0
const AUTOTEST_STEP := 6
## Скриншот в --autotest снимается через это время после удара (середина атаки).
const AUTOTEST_SHOT_DELAY := 0.25

@onready var world = $WorldMap
@onready var hero = $Hero
@onready var camera = $FollowCamera
@onready var hint: Label = $HUD/Hint
@onready var attack_button: Button = $HUD/Controls/AttackButton
@onready var death_button: Button = $HUD/Controls/DeathButton

var _autotest := false
var _screenshot_path := ""
var _autotest_target := Vector2i.ZERO
## 0: идём к цели, 1: удар, 2: смерть и воскрешение, 3: завершено.
var _autotest_phase := 0


func _ready() -> void:
	RenderingServer.set_default_clear_color(BG_COLOR)
	for a in OS.get_cmdline_user_args():
		if a == "--autotest":
			_autotest = true
		elif a.begins_with("--screenshot="):
			_screenshot_path = a.trim_prefix("--screenshot=")

	# Препятствия становятся детьми main, чтобы y-сортировка работала вместе с героем.
	world.build_props(self)

	hero.world = world
	hero.position = world.grid_to_world(START_CELL)
	hero.arrived.connect(_on_hero_arrived)
	hero.attack_finished.connect(_on_hero_attack_finished)
	hero.revived.connect(_on_hero_revived)

	# Тестовые кнопки: нажатие съедается Button, поэтому герой не двигается.
	attack_button.pressed.connect(hero.attack)
	death_button.pressed.connect(hero.die)

	camera.target = hero
	camera.set_bounds(world.get_bounds())

	_run_hud_fade()

	if _autotest:
		get_tree().create_timer(AUTOTEST_START_DELAY).timeout.connect(_autotest_tap)
		get_tree().create_timer(AUTOTEST_TIMEOUT).timeout.connect(_autotest_timeout)
	elif _screenshot_path != "":
		get_tree().create_timer(1.0).timeout.connect(_on_screenshot_timer)


func _run_hud_fade() -> void:
	var tw := create_tween()
	tw.tween_interval(HUD_FADE_AT)
	tw.tween_property(hint, "modulate:a", 0.0, 0.8)
	tw.tween_callback(hint.hide)


func _autotest_tap() -> void:
	var from_cell: Vector2i = world.world_to_grid(hero.position)
	_autotest_target = from_cell + Vector2i(AUTOTEST_STEP, 0)
	print("AUTOTEST tap from=%s target=%s" % [from_cell, _autotest_target])
	hero.move_to_cell(_autotest_target)


func _on_hero_arrived() -> void:
	if not _autotest or _autotest_phase != 0:
		return
	var cell: Vector2i = world.world_to_grid(hero.position)
	var expected: Vector2i = world.nearest_free_cell(_autotest_target)
	var center: Vector2 = world.grid_to_world(cell)
	var on_center: bool = hero.position.distance_to(center) < 1.0
	if not (cell == expected and on_center):
		print("AUTOTEST FAIL pos=(%.1f, %.1f) cell=%s expected=%s" % [hero.position.x, hero.position.y, cell, expected])
		_autotest_phase = 3
		get_tree().quit(1)
		return
	print("AUTOTEST arrived pos=(%.1f, %.1f) cell=%s -> attack" % [hero.position.x, hero.position.y, cell])
	_autotest_phase = 1
	hero.attack()
	if _screenshot_path != "":
		get_tree().create_timer(AUTOTEST_SHOT_DELAY).timeout.connect(_on_attack_shot)


func _on_attack_shot() -> void:
	if _autotest and _autotest_phase == 1:
		await _capture(_screenshot_path)


func _on_hero_attack_finished() -> void:
	if not _autotest or _autotest_phase != 1:
		return
	print("AUTOTEST attack done -> die")
	_autotest_phase = 2
	hero.die()


func _on_hero_revived() -> void:
	if not _autotest or _autotest_phase != 2:
		return
	_autotest_phase = 3
	print("AUTOTEST OK walk+attack+death+revive pos=(%.1f, %.1f)" % [hero.position.x, hero.position.y])
	get_tree().quit(0)


func _autotest_timeout() -> void:
	if _autotest_phase == 3:
		return
	var phase := _autotest_phase
	_autotest_phase = 3
	print("AUTOTEST FAIL timeout phase=%d pos=(%.1f, %.1f)" % [phase, hero.position.x, hero.position.y])
	get_tree().quit(1)


func _on_screenshot_timer() -> void:
	await _capture(_screenshot_path)
	get_tree().quit()


## Сохраняет кадр viewport в PNG (после отрисовки кадра).
func _capture(path: String) -> void:
	await RenderingServer.frame_post_draw
	var img := get_viewport().get_texture().get_image()
	if img == null:
		print("SCREENSHOT FAIL: no image (renderer unavailable?)")
		return
	DirAccess.make_dir_recursive_absolute(path.get_base_dir())
	var err: int = img.save_png(path)
	print("SCREENSHOT %s err=%d" % [path, err])
