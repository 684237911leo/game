class_name Combat
extends Node
## Полуавтоматический бой (point & click). Владеет врагами, выбранной целью и кулдаунами.
## Тап по врагу -> select_target: героиня идёт к клетке на дистанции удара (world.find_path),
## в досягаемости сама бьёт по кулдауну; путь пересчитывается каждые REPATH_INTERVAL.
## Смерть цели -> авто-переключение на ближайшего живого врага в AUTO_SWITCH_RADIUS (от героини).
## Навык «Выпад»: SKILL_DAMAGE по цели в SKILL_RANGE (или по ближайшему в SKILL_RANGE), кулдаун SKILL_COOLDOWN.
## Враги разводятся (SEPARATION, HERO_SPACE) и не заходят в блокированные клетки.

signal target_changed(enemy)
## Враг погиб. next_target — автоцель (или null), она уже выбрана в момент сигнала.
signal enemy_killed(enemy, next_target)
signal struck(enemy, damage, skill)

const SPAWN_SEED := 4242
const SPAWN_GROUPS := 3
const SPAWN_PER_GROUP := 2
const SPAWN_MIN_DIST := 8.0
const SPAWN_GROUP_GAP := 8.0
const HERO_REACH := 1.7
const HERO_STAND_REACH := 1.4
const HERO_STRIKE_DELAY := 0.35
const HERO_STRIKE_SLACK := 0.2
const HERO_ATTACK_CD := 1.2
const HERO_DAMAGE := 12
const SKILL_DAMAGE := 40
const SKILL_COOLDOWN := 5.0
const SKILL_RANGE := 3.0
const AUTO_SWITCH_RADIUS := 6.0
const REPATH_INTERVAL := 0.3
const UNREACHABLE_GIVE_UP := 1.5
const SEPARATION := 0.9
const HERO_SPACE := 0.7

var world = null
var hero = null
var enemies: Array = []
var target: Enemy3D = null
var kills := 0
## Сколько ударов героини прошло (для автотеста и кадров).
var hero_hits := 0
## Навык выбран и ждёт удара (при подходе к цели).
var skill_pending := false

var _rng := RandomNumberGenerator.new()
var _attack_cd := 0.0
var _skill_cd := 0.0
var _strike_in := -1.0
var _strike_target: Enemy3D = null
var _strike_skill := false
var _repath := 0.0
var _stuck := 0.0


func setup(p_world, p_hero) -> void:
	world = p_world
	hero = p_hero
	hero.died.connect(_on_hero_died)


## Расстояние в плане (XZ).
static func hdist(a: Vector3, b: Vector3) -> float:
	return Vector2(a.x - b.x, a.z - b.z).length()


## Убрать из пути точки, которые уже пройдены: первая точка отбрасывается, если позиция
## лежит впереди неё по направлению к второй (проекция на сегмент). Дистанция тут не годится:
## герой за центром клетки по диагонали ближе к центру, и путь откатывал бы его назад.
static func trim_passed(pos: Vector3, pts: PackedVector3Array) -> PackedVector3Array:
	while pts.size() > 1:
		var seg: Vector3 = pts[1] - pts[0]
		seg.y = 0.0
		var rel: Vector3 = pos - pts[0]
		rel.y = 0.0
		if seg.length_squared() < 0.0001 or rel.dot(seg) < 0.0:
			break
		pts.remove_at(0)
	return pts


## Клетка с центром не дальше reach от target_pos (свободная), ближайшая к from. (-1, -1), если нет.
## Общая для героини (досягаемость удара) и врагов (ближний бой).
static func stand_cell(world, target_pos: Vector3, reach: float, from: Vector3) -> Vector2i:
	var tc: Vector2i = world.world_to_grid(target_pos)
	var best := Vector2i(-1, -1)
	var best_d := INF
	for dy in range(-2, 3):
		for dx in range(-2, 3):
			var c := tc + Vector2i(dx, dy)
			var blocked: bool = world.is_blocked(c)
			if blocked:
				continue
			var p: Vector3 = world.grid_to_world(c)
			if hdist(p, target_pos) > reach:
				continue
			var d := hdist(p, from)
			if d < best_d:
				best_d = d
				best = c
	return best


# ------------------------------------------------------------------ расстановка

## 3 группы по 2 скелета: вне центра, не в блоках, минимум SPAWN_MIN_DIST от старта, достижимы.
func spawn_enemies(start_pos: Vector3) -> void:
	_rng.seed = SPAWN_SEED
	var anchors: Array[Vector2i] = []
	var guard := 0
	while anchors.size() < SPAWN_GROUPS and guard < 4000:
		guard += 1
		var c := Vector2i(_rng.randi_range(2, 33), _rng.randi_range(2, 33))
		var blocked: bool = world.is_blocked(c)
		if blocked:
			continue
		var p: Vector3 = world.grid_to_world(c)
		if hdist(p, start_pos) < SPAWN_MIN_DIST:
			continue
		var far := true
		for a in anchors:
			if Vector2(a - c).length() < SPAWN_GROUP_GAP:
				far = false
				break
		if not far:
			continue
		var pts: PackedVector3Array = world.find_path(start_pos, p)
		if pts.is_empty():
			continue
		anchors.append(c)
	for a in anchors:
		var free: Array[Vector2i] = []
		for dy in range(-1, 2):
			for dx in range(-1, 2):
				if dx == 0 and dy == 0:
					continue
				var n := a + Vector2i(dx, dy)
				var nb: bool = world.is_blocked(n)
				if not nb:
					free.append(n)
		var cells: Array[Vector2i] = [a]
		while cells.size() < SPAWN_PER_GROUP and not free.is_empty():
			var k := _rng.randi_range(0, free.size() - 1)
			cells.append(free[k])
			free.remove_at(k)
		for cell in cells:
			var wp: Vector3 = world.grid_to_world(cell)
			_spawn_one(wp)


func _spawn_one(p: Vector3) -> void:
	var e := Enemy3D.new()
	e.name = "Skeleton%d" % enemies.size()
	add_child(e)
	e.position = Vector3(p.x, 0.0, p.z)
	e.setup(world, hero)
	e.died.connect(_on_enemy_died)
	enemies.append(e)


# ------------------------------------------------------------------ выбор и запросы

## Враг под лучом камеры (ближайший к камере), или null.
func pick_enemy(origin: Vector3, dir: Vector3) -> Enemy3D:
	var best: Enemy3D = null
	var best_t := INF
	for e in enemies:
		var t: float = e.pick_distance(origin, dir)
		if t >= 0.0 and t < best_t:
			best_t = t
			best = e
	return best


## Ближайший живой враг к точке в радиусе radius (исключая exclude), или null.
func nearest_enemy(from: Vector3, radius: float = INF, exclude: Enemy3D = null) -> Enemy3D:
	var best: Enemy3D = null
	var best_d := INF
	for e in enemies:
		if not e.is_alive() or e == exclude:
			continue
		var d := hdist(e.position, from)
		if d <= radius and d < best_d:
			best_d = d
			best = e
	return best


func select_target(e: Enemy3D) -> void:
	if e == null or not e.is_alive() or e == target:
		return
	if target != null:
		target.set_selected(false)
	target = e
	e.set_selected(true)
	_repath = 0.0
	_stuck = 0.0
	target_changed.emit(e)


## Сбросить цель (тап по земле). Героиня не останавливается: её ведёт тап по земле.
func clear_target() -> void:
	if target == null:
		return
	target.set_selected(false)
	target = null
	target_changed.emit(null)


## Полный сброс боя (для кадров и смены режима): цель, замах, навык.
func cancel() -> void:
	clear_target()
	_strike_in = -1.0
	_strike_target = null
	skill_pending = false


func skill_ready() -> bool:
	return _skill_cd <= 0.0


func skill_ready_fraction() -> float:
	return clampf(1.0 - _skill_cd / SKILL_COOLDOWN, 0.0, 1.0)


## Навык «Выпад». Цель: текущая, если в SKILL_RANGE, иначе ближайший в SKILL_RANGE. false — не сработал.
func use_skill() -> bool:
	if hero == null or not skill_ready() or hero.state_name() == "dead":
		return false
	var cand: Enemy3D = null
	if target != null and target.is_alive() and hdist(hero.position, target.position) <= SKILL_RANGE:
		cand = target
	else:
		cand = nearest_enemy(hero.position, SKILL_RANGE)
	if cand == null:
		return false
	select_target(cand)
	skill_pending = true
	_skill_cd = SKILL_COOLDOWN
	_attack_cd = 0.0
	return true


# ------------------------------------------------------------------ процесс

func _process(delta: float) -> void:
	if hero == null:
		return
	if _attack_cd > 0.0:
		_attack_cd -= delta
	if _skill_cd > 0.0:
		_skill_cd -= delta
	_separate()
	if _strike_in > 0.0:
		_strike_in -= delta
		if _strike_in <= 0.0:
			_strike_in = -1.0
			_apply_strike()
		return
	if target == null or not target.is_alive():
		return
	var hs: String = hero.state_name()
	if hs == "dead" or hs == "attack":
		return
	var dist := hdist(hero.position, target.position)
	if dist <= HERO_REACH:
		if hs == "walk":
			hero.stop()
		hero.face_point(target.position)
		if _attack_cd <= 0.0:
			_begin_swing()
		return
	_repath -= delta
	if _repath <= 0.0:
		_repath = REPATH_INTERVAL
		_approach()


## Путь к клетке на дистанции удара от цели. Не нашли путь UNREACHABLE_GIVE_UP секунд — сброс цели.
func _approach() -> void:
	var cell: Vector2i = stand_cell(world, target.position, HERO_STAND_REACH, hero.position)
	var pts := PackedVector3Array()
	if cell.x >= 0:
		var wp: Vector3 = world.grid_to_world(cell)
		pts = world.find_path(hero.position, wp)
	if pts.is_empty():
		_stuck += REPATH_INTERVAL
		if _stuck >= UNREACHABLE_GIVE_UP:
			hero.stop()
			clear_target()
		return
	_stuck = 0.0
	hero.move_to(world.grid_to_world(cell))


func _begin_swing() -> void:
	hero.stop()
	hero.face_point(target.position)
	hero.attack()
	if hero.state_name() != "attack":
		return
	_strike_target = target
	_strike_skill = skill_pending
	_strike_in = HERO_STRIKE_DELAY
	_attack_cd = HERO_ATTACK_CD


## Урон в момент выпада. Если цель уже вне досягаемости — промах (навык остаётся ждать).
func _apply_strike() -> void:
	var t := _strike_target
	var skill := _strike_skill
	_strike_target = null
	_strike_skill = false
	if t == null or not t.is_alive() or hero.state_name() == "dead":
		return
	if hdist(hero.position, t.position) > HERO_REACH + HERO_STRIKE_SLACK:
		return
	var dmg := SKILL_DAMAGE if skill else HERO_DAMAGE
	if skill:
		skill_pending = false
	hero_hits += 1
	struck.emit(t, dmg, skill)
	t.take_damage(dmg)


func _on_enemy_died(e: Enemy3D) -> void:
	kills += 1
	var next: Enemy3D = null
	if e == target:
		target = null
		next = nearest_enemy(hero.position, AUTO_SWITCH_RADIUS)
		if next != null:
			target = next
			next.set_selected(true)
			_repath = 0.0
			_stuck = 0.0
		else:
			hero.stop()
		target_changed.emit(target)
	enemy_killed.emit(e, next)


func _on_hero_died() -> void:
	clear_target()
	_strike_in = -1.0
	_strike_target = null
	_strike_skill = false


## Разведение: живые враги не ближе SEPARATION друг к другу и HERO_SPACE к герою.
## Сдвиг, попадающий в блок, отменяется.
func _separate() -> void:
	var live: Array = []
	for e in enemies:
		if e.is_alive():
			live.append(e)
	for i in live.size():
		var a: Enemy3D = live[i]
		var dh := hdist(a.position, hero.position)
		if dh < HERO_SPACE:
			var away := Vector3(a.position.x - hero.position.x, 0.0, a.position.z - hero.position.z)
			if away.length_squared() < 0.0001:
				away = Vector3.RIGHT
			_try_move(a, a.position + away.normalized() * (HERO_SPACE - dh))
		for j in range(i + 1, live.size()):
			var b: Enemy3D = live[j]
			var d := hdist(a.position, b.position)
			if d >= SEPARATION:
				continue
			var dir := Vector3(a.position.x - b.position.x, 0.0, a.position.z - b.position.z)
			if dir.length_squared() < 0.0001:
				dir = Vector3.RIGHT
			var push := dir.normalized() * (SEPARATION - d) * 0.5
			_try_move(a, a.position + push)
			_try_move(b, b.position - push)


func _try_move(e: Enemy3D, p: Vector3) -> void:
	p.y = 0.0
	var blocked: bool = world.is_blocked(world.world_to_grid(p))
	if not blocked:
		e.position = p
