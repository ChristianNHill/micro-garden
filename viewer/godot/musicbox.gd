# The music box's own music: whatever mp3s are in the player's music folder (~/.cache/micro-garden/music by
# default, --music=DIR for another), shuffled, playing while the box is down in the garden and stopping when it
# is picked up. The files are the player's and are never part of this project. A click on the box steps its
# volume round, and it fades in and out rather than cutting. What the ducks hear is the garden's business, not
# this: to them the box is a sound source with a loudness, whatever is on.
extends Node

const LEVELS := [0.0, 0.25, 0.5, 0.75, 1.0]
const FADE_S := 1.2

var player := AudioStreamPlayer.new()
var tracks: PackedStringArray = []
var queue: Array = []
var level := 3  # which of LEVELS
var on := false
var gain := 0.0  # what is heard now, easing towards the level
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
	if queue.is_empty():  # a fresh shuffle each time round, every track once
		queue = Array(tracks)
		queue.shuffle()
	var path: String = queue.pop_back()
	var stream := AudioStreamMP3.new()
	stream.data = FileAccess.get_file_as_bytes(path)
	player.stream = stream
	player.play()
	title = path.get_file().get_basename()
	var bitrate := title.rfind(" (")  # a ripped file's name ends " - uploader (128k)": neither is the song's
	if bitrate > 0 and title.ends_with("k)"):
		title = title.substr(0, bitrate)
	if title.count(" - ") >= 2:
		title = title.substr(0, title.rfind(" - "))


func set_on(down: bool) -> void:
	if down and not on and not player.playing:
		_next()
	on = down


func step_volume() -> float:
	# The next level round, for the garden to be told: the volume is the garden's, so that off is off for
	# the ducks as well, and this player follows what the garden says (`follow`).
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
