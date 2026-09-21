# Micro Garden

Micro Garden is five small ducks in a garden, and each duck runs on its own live copy of a real fruit fly brain. It asks whether a wiring diagram, a body with needs, and a place to live are enough to get behaviour nobody scripted. The brain is the FlyWire connectome, about 139,000 neurons and 2.7 million connections, simulated as spiking neurons. Each duck smells, sees, hears, gets hungry, thirsty and sleepy, learns, and has a personality.

The ducks are modelled on the [microduck](https://github.com/pollen-robotics/microduck), a real open-source robot, and the brain talks to the body through that robot's own command protocol. There are two bodies today. One is a 2D stub in Python. The other is the microduck's own MuJoCo simulation, where the real robot daemon runs the real walking policy and five simulated ducks live a garden day on five fly brains. A low-poly garden in Godot draws either body. The project is simulation only, so nobody plans real ducks on a floor.

## What the brain does and what is explicit

This project tries to be exact here, because "fly brain drives a duck" is an easy claim to inflate.

The senses go in through the fly's real sensory neurons: food and danger smells, humidity, heat and cold, touch, taste, sound and wind on the antennae, and light through a model of the optic lobe ([flyvis](https://github.com/TuragaLab/flyvis)). Movement comes out through real descending neurons. A screen of the whole brain found them, and nobody picked their names from papers. The clearest one is DNge091. It fires on the side the wind comes from, it does the same on a held-out seed, and it stays silent when the wiring is shuffled. A duck that smells food turns into the wind and walks up the plume, which is how a real fly finds food. With the real wiring a duck finds a dish in 20 of 20 trials. With the wiring shuffled it finds it in 1 of 20.

Learning is the fly's too. The only synapses that change are the ones between Kenyon cells and mushroom body output neurons, and dopamine gates them, reward through one set of neurons and punishment through another.

Some behaviour is explicit code and not the brain, and the code says so where it happens. The body decides how much attention each sense gets, so a hungry duck listens to the wind and a fed one does not. Turning toward music, toward other ducks (companionship), and acting on what a duck has learned about a smell (fondness) are explicit readouts. They exist because measurement showed the brain's own left and right signals for a smell sit below its steering noise. Personalities are presets over continuous dials, in the style of the Chao from Sonic Adventure.

## Run it

You need macOS on Apple Silicon or any machine PyTorch supports, Python 3.12, and [uv](https://docs.astral.sh/uv/). The repo does not redistribute the FlyWire data, so download two files into `data/`:

- `proofread_connections_783.feather` from the [FlyWire v783 release on Zenodo](https://doi.org/10.5281/zenodo.10676866)
- `Supplemental_file1_neuron_annotations.tsv` from [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)

Then start the garden:

```
uv run python -m body.stub2d.stub --view --brain
```

Click a duck to follow it and click the tree to shake fruit down. `P` pets the selected duck, `C` claps, `F` drops food at the mouse, `M` picks the music up or puts it down, and `H` puts a hat on a duck or takes it off. The garden saves when you close it. When you come back it ages the ducks by however long you were away. Add `--fresh` to hatch new ducks, `--labels Bully,Napper,Carefree,Chatty,Scaredy` to choose who hatches, and `--blind` to hide who is who until you quit. Five ducks with their eyes open run at about real time on an M-series Mac.

To run the ducks as simulated robots, check out [microduck](https://github.com/pollen-robotics/microduck) and [microduck_rl](https://github.com/pollen-robotics/microduck_rl), start five ducks with `DUCK_SIM_DUCKS=5 scripts/duck-sim`, and then start the garden around them:

```
uv run python -m body.mujoco.adapter --ducks 5 --view --brain
```

`PLAN.md` under Gate 10 has the setup, which runs natively on a Mac. The walking policy upstream ships does not yet track a velocity, so these ducks march and pivot. Tab takes the wheel of the selected duck in either body, and W, A, S and D drive it while its brain keeps watching.

## The Godot garden

`viewer/godot/` is the garden to look at. It is laid out after the Chao gardens of Sonic Adventure 2: a lawn in a bowl of rock, a pond under a waterfall, a fruit tree, a rail fence over the sea. The ducks are the real microduck, built from Pollen Robotics' own meshes and posed joint by joint. A halftone screen of ink shades everything, and night is a wash of blue. It needs [Godot 4](https://godotengine.org). Start either body with `--godot`, and then open the project:

```
uv run python -m body.stub2d.stub --brain --godot
/Applications/Godot.app/Contents/MacOS/Godot --path viewer/godot
```

Godot only draws. The garden publishes its world over UDP, and what you do goes back as the same control calls the 2D window uses.

- Click a duck to select it. The camera follows it, a panel shows its needs, moods and wants as bars, and another shows its fly brain: 12,000 of its 139,000 neurons where FlyWire has them, each flaring as it fires.
- `Tab` rides the selected duck. W, A, S and D steer it, and you see its two hex retinas and what its descending neurons ask of its legs.
- `F` shakes fruit down from the tree, and so does a click on it. `H` drops a hat at the mouse, `B` drops a ball, `M` puts the music box down or picks it up, `C` claps, and `P` pets the selected duck.
- A duck that likes the music dances to it while the box is down and nothing presses on it.
- The music box plays your own music. Put mp3s in `~/.cache/micro-garden/music/` and it shuffles them while it is down in the garden. Click the box to step its volume. No music ships with the project.
- Hats are for the ducks. A duck that finds one decides whether to wear it, and a vain duck usually does. A duck that shakes one off leaves it on the lawn for the next.

The ducks emote every so often, like a Chao or a Sim. Each acts out its strongest feeling with its head and its voice, a chatty duck more often. Strong characters have a trick of their own: an aggressive duck stomps, a sleepy one yawns, a water lover splashes, a chatty one sings, and a timid one cowers. A flag on the near cliff shows the breeze, which is what the ducks find food by.

## How it is checked

The build goes in gates. One gate is one runnable check that exits non-zero when it fails. `PLAN.md` records what each one found, including the times a gate showed that an earlier pass was luck.

```
uv run python -m gates # all of them, which takes hours
uv run python -m gates --fast # the ones that take about a minute
uv run python -m gates 4 8 # just those
```

Most gates pass. One check fails today and `PLAN.md` says so: in Gate 5 a fed duck that is too hot does not go and cool off, water lover or not, because nothing but a need or boredom gets a duck walking. Two other results stand because they are true. The five demo personalities are hard to tell apart by numbers alone, so Gate 5 prints that comparison and does not assert it. The Bully's aggression costs it food, which is character and not a bug.

## Where things are

- `brain/` loads the connectome and holds the spiking model, the sensory encoder, the motor decoder, learning, bodily needs, and the personality presets.
- `body/` holds the robot command contract, the sensory frame, and the 2D stub body with its retina.
- `world/` holds the garden: smells that drift on the wind, the pond, temperature, day and night, music.
- `viewer/` holds the debug window, the world snapshot, and the Godot garden with the script that builds its robot.
- `gates/` holds one check per gate.

`RESEARCH.md` has the goal and the reading behind it. `ARCHITECTURE.md` has the design and the decisions. `PLAN.md` is the build log. `ATTRIBUTION.md` lists the data, the papers to cite, and the projects this leans on.

## Credit

The connectome is the work of the FlyWire Consortium (Dorkenwald et al. 2024, Schlegel et al. 2024, Matsliah et al. 2024). The visual front end is flyvis from the Turaga Lab. The body contract follows Pollen Robotics' microduck. Full details and licenses are in `ATTRIBUTION.md`.

I direct the project and make its decisions. Claude Code wrote most of the code, as the commit history shows. The code is MIT licensed. The FlyWire data has its own terms, which `ATTRIBUTION.md` lists.
