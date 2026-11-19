extends Camera3D
## Изометрическая камера: ортографическая, поворот 45° по Y, наклон ~50° вниз, дистанция DISTANCE.
## Мягко следует за героиней с look-ahead по направлению движения (до LOOK_AHEAD м, при остановке
## плавно возвращается). Позиция сглаживается экспонентой. Фокус зажат границами карты с запасом
## по половине видимой области; если карта меньше видимой области, фокус — по центру.

const YAW_DEG := 45.0
const PITCH_DEG := 50.0
const DISTANCE := 25.0
## Видимая высота области в метрах (size при KEEP_HEIGHT). Портрет 720x1280: ширина = size * 9/16.
const ZOOM_SIZE := 9.0
## Точка фокуса по высоте: уровень груди героини.
const FOCUS_HEIGHT := 0.9
const LOOK_AHEAD := 2.5
const LOOK_RATE := 3.0
const SMOOTH_SPEED := 5.0
const CLAMP_MARGIN := 1.0

## Герой (Node3D с полем velocity: Vector3), назначается из main.gd.
var target = null

var _bounds := Rect2(-18.0, -18.0, 36.0, 36.0)
var _zoom := ZOOM_SIZE
## Ед. вектор от фокуса к камере.
var _dir := Vector3.ZERO
## Базис экрана в мировых координатах (вправо и вверх для камеры).
var _right := Vector3.RIGHT
var _up := Vector3.UP
## Половины области, видимой на земле, относительно фокуса (x, z).
var _ext := Vector2.ZERO
var _focus := Vector3.ZERO
var _look := Vector3.ZERO
var _inited := false


func _ready() -> void:
	projection = Camera3D.PROJECTION_ORTHOGONAL
	keep_aspect = Camera3D.KEEP_HEIGHT
	near = 0.5
	far = 200.0
	var yaw := deg_to_rad(YAW_DEG)
	var pitch := deg_to_rad(PITCH_DEG)
	_dir = Vector3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)).normalized()
	var fwd := -_dir
	_right = fwd.cross(Vector3.UP).normalized()
	_up = _right.cross(fwd).normalized()
	make_current()
	set_zoom(ZOOM_SIZE)


## Границы карты на плоскости XZ (Rect2: x = мировой x, y = мировой z).
func set_bounds(r: Rect2) -> void:
	_bounds = r


## Видимая высота (size) в метрах; пересчитывает запас границ.
func set_zoom(z: float) -> void:
	_zoom = z
	size = z
	_ext = _footprint_half_extents()


func _process(delta: float) -> void:
	if target == null:
		return
	var want := Vector3.ZERO
	var vel: Vector3 = target.velocity
	if vel.length_squared() > 0.01:
		want = vel.normalized() * LOOK_AHEAD
	_look = _look.lerp(want, 1.0 - exp(-LOOK_RATE * delta))
	var goal: Vector3 = target.global_position + _look
	goal.y = FOCUS_HEIGHT
	goal = _clamp_focus(goal)
	if not _inited:
		_focus = goal
		_inited = true
	else:
		_focus = _focus.lerp(goal, 1.0 - exp(-SMOOTH_SPEED * delta))
	global_position = _focus + _dir * DISTANCE
	look_at(_focus, Vector3.UP)


## Половины прямоугольника, который видит камера на плоскости y = 0, относительно фокуса (x, z).
## Фокус считается на земле в начале координат; проекция угла экрана пересекается с землёй.
func _footprint_half_extents() -> Vector2:
	var vp := get_viewport().get_visible_rect().size
	var aspect := vp.x / maxf(vp.y, 1.0)
	var hh := _zoom * 0.5
	var hw := hh * aspect
	var fwd := -_dir
	var cam_pos := _dir * DISTANCE
	var ext := Vector2.ZERO
	for sx in [-1.0, 1.0]:
		for sy in [-1.0, 1.0]:
			var o: Vector3 = cam_pos + _right * (sx * hw) + _up * (sy * hh)
			var t: float = -o.y / fwd.y
			var hit: Vector3 = o + fwd * t
			ext.x = maxf(ext.x, absf(hit.x))
			ext.y = maxf(ext.y, absf(hit.z))
	return ext


func _clamp_focus(p: Vector3) -> Vector3:
	var lo_x := _bounds.position.x + _ext.x + CLAMP_MARGIN
	var hi_x := _bounds.end.x - _ext.x - CLAMP_MARGIN
	var lo_z := _bounds.position.y + _ext.y + CLAMP_MARGIN
	var hi_z := _bounds.end.y - _ext.y - CLAMP_MARGIN
	if lo_x > hi_x:
		p.x = _bounds.get_center().x
	else:
		p.x = clampf(p.x, lo_x, hi_x)
	if lo_z > hi_z:
		p.z = _bounds.get_center().y
	else:
		p.z = clampf(p.z, lo_z, hi_z)
	return p
