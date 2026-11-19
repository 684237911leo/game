extends Camera2D
## Мягкое сопровождение героини с look-ahead по направлению движения.

const ZOOM := Vector2(1.0, 1.0)
const LOOK_AHEAD := 140.0
const LOOK_RATE := 3.0
const SMOOTH_SPEED := 5.0

## Герой (Node2D с полем velocity), назначается из main.gd.
var target = null
var _look := Vector2.ZERO


func _ready() -> void:
	zoom = ZOOM
	position_smoothing_enabled = true
	position_smoothing_speed = SMOOTH_SPEED
	limit_smoothed = true
	make_current()


## Границы камеры по прямоугольнику карты (в мировых координатах).
func set_bounds(r: Rect2) -> void:
	limit_left = int(floor(r.position.x))
	limit_top = int(floor(r.position.y))
	limit_right = int(ceil(r.end.x))
	limit_bottom = int(ceil(r.end.y))


func _process(delta: float) -> void:
	if target == null:
		return
	var want := Vector2.ZERO
	var vel: Vector2 = target.velocity
	if vel.length_squared() > 0.01:
		want = vel.normalized() * LOOK_AHEAD
	_look = _look.lerp(want, 1.0 - exp(-LOOK_RATE * delta))
	global_position = target.global_position + _look
