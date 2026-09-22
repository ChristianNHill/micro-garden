# The music box plays the mp3s in the player's music folder (~/.cache/micro-garden/music, or --music=DIR),
# shuffled, while the box is down in the garden. The files are the player's, never part of this project.
# A click steps the volume; it fades rather than cuts. The garden decides what the ducks hear: to them the
# box is a sound source with a loudness.
extends Node

const LEVELS := [0.0, 0.25, 0.5, 0.75, 1.0]
const FADE_S := 1.2

var player := AudioStreamPlayer.new()
var tracks: PackedStringArray = []
var queue: Array = []
var level := 3  # which of LEVELS
var on := false
var gain := 0.0  # eases towards the level
var title := ""


func setup(folder: String) -> void:
	add_child(player)
	player.finished.connect(_next)
	var dir := DirAccess.open(folder)
	if dir == null:
		return
	for file in dir.get_files():
		if file.to_lower().ends_with(".mp3"):
			tracks.append(folder.path_join(file))


func _next() -> void:
	if tracks.is_empty():
		return
	if queue.is_empty():  # reshuffle once every track has played
		queue = Array(tracks)
		queue.shuffle()
	var path: String = queue.pop_back()
	var stream := AudioStreamMP3.new()
	stream.data = FileAccess.get_file_as_bytes(path)
	player.stream = stream
	player.play()
	title = path.get_file().get_basename()
	var bitrate := title.rfind(" (")  # strip a ripped file's " - uploader (128k)" suffix
	if bitrate > 0 and title.ends_with("k)"):
		title = title.substr(0, bitrate)
	if title.count(" - ") >= 2:
		title = title.substr(0, title.rfind(" - "))


func set_on(down: bool) -> void:
	if down and not on and not player.playing:
		_next()
	on = down


func step_volume() -> float:
	# The next level, to send to the garden. The garden owns the volume, so off is off for the ducks too,
	# and this player follows it (`follow`).
	return LEVELS[(level + 1) % LEVELS.size()]


func follow(volume: float) -> void:
	for k in LEVELS.size():
		if abs(LEVELS[k] - volume) < 0.13:
			level = k


func _process(dt: float) -> void:
	var want: float = LEVELS[level] if on else 0.0
	gain = move_toward(gain, want, dt / FADE_S)
	player.volume_db = linear_to_db(max(gain, 0.0001))
	if not on and gain <= 0.0 and player.playing:
		player.stop()
