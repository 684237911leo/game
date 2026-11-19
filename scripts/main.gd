extends Node2D
## Корень среза: собирает карту, героиню, камеру, HUD; режим --autotest и --screenshot=путь.

const START_CELL := Vector2i(14, 14)
const BG_COLOR := Color(0.05, 0.05, 0.04)
const HUD_FADE_AT := 4.0
const AUTOTEST_START_DELAY := 0.5
const AUTOTEST_TIMEOUT := 15.0
const AUTOTEST_STEP := 6

@onready var world = $WorldMap
@onready var hero = $Hero
@onready var camera = $FollowCamera
@onready var hint: Label = $HUD/Hint

var _autotest := false
var _screenshot_path := ""
var _autotest_target := Vector2i.ZERO
var _finished := false


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
	if not _autotest or _finished:
		return
	_finished = true
	var cell: Vector2i = world.world_to_grid(hero.position)
	var expected: Vector2i = world.nearest_free_cell(_autotest_target)
	var center: Vector2 = world.grid_to_world(cell)
	var on_center: bool = hero.position.distance_to(center) < 1.0
	var ok: bool = cell == expected and on_center
	if ok:
		print("AUTOTEST OK pos=(%.1f, %.1f) cell=%s target=%s" % [hero.position.x, hero.position.y, cell, expected])
	else:
		print("AUTOTEST FAIL pos=(%.1f, %.1f) cell=%s expected=%s" % [hero.position.x, hero.position.y, cell, expected])
	if _screenshot_path != "":
		await _capture(_screenshot_path)
	get_tree().quit(0 if ok else 1)


func _autotest_timeout() -> void:
	if _finished:
		return
	_finished = true
	print("AUTOTEST FAIL timeout pos=(%.1f, %.1f)" % [hero.position.x, hero.position.y])
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
