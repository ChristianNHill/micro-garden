# Micro Garden

![Five microducks in the Godot garden: one sings, one yawns, one takes fright, and a ball, a drum, hats and fruit lie on the lawn](docs/garden.gif)

Micro Garden is five small ducks in a garden, and each duck runs on its own live copy of a real fruit fly brain. I built it to ask whether a wiring diagram, a body with needs, and a place to live are enough to get behaviour nobody scripted. The brain is the FlyWire connectome, about 139,000 neurons and 2.7 million connections, simulated as spiking neurons. Each duck smells, sees and hears. It gets hungry, thirsty, hot and sleepy. It learns, makes friends and holds grudges, and each one has a personality.

The ducks are modelled on the [microduck](https://github.com/pollen-robotics/microduck), a real open-source robot, and the brain talks to the body through that robot's own command protocol. The ducks can be simulated in two ways: a simple 2D simulation in Python, or the microduck's own MuJoCo physics simulation, where the robot's real control software runs its real walking policy. Either way, a low-poly 3D garden in Godot shows what the ducks are doing. Everything runs in simulation, and I have no plans to build the physical robots.

## Run it

You need macOS on Apple Silicon or any machine PyTorch supports, Python 3.12, [uv](https://docs.astral.sh/uv/), and [Godot 4](https://godotengine.org). The garden looks for Godot at the path in `GODOT`, then for `godot` on your path, then in `/Applications`. The repo does not redistribute the FlyWire data, so download two files into `data/`:

- `proofread_connections_783.feather` from the [FlyWire v783 release on Zenodo](https://doi.org/10.5281/zenodo.10676866)
- `Supplemental_file1_neuron_annotations.tsv` from [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)

Then open the garden with one command:

```
uv run python -m garden
```

That starts five ducks, each with its own brain, and opens the Godot window to show them. Closing the window saves the garden and stops it. The next time you open it, the ducks are older by however long you were away. Add `--fresh` to start with new ducks, `--labels Bully,Napper,Carefree,Chatty,Scaredy` to choose their personalities, and `--blind` to hide which duck has which personality until you quit. With vision on, five ducks run at about real time on an M-series Mac.

## The garden

The garden is modelled on the Chao gardens of Sonic Adventure 2. It is a lawn surrounded by rock walls, with a pond under a waterfall, a fruit tree, and a fence along a cliff above the sea. Each duck is a 3D model of the microduck, built from Pollen Robotics' own meshes and animated joint by joint. The ducks sit, walk, kick and get up using the robot's own recorded motions. The shading uses a halftone dot pattern, and the night is tinted blue. A flag on the nearest cliff shows which way the wind blows, which matters because the ducks find food by smelling it on the wind.

![The garden from the open front of the lawn, with ducks at the pond, hats on two of them, a ball and a drum on the grass](docs/garden.webp)

Select a duck and you see who it is, what it needs and feels, and its fly brain firing:

![The Bully selected, sorrowful and cold, with its needs, moods and wants as bars and its brain in the corner](docs/selected-duck.webp)

![The Carefree duck selected under the fruit tree, happy, with its favourite fruit and its skills on its card](docs/under-the-tree.webp)

Godot only draws the garden. The simulation runs in Python and sends the state of the world to Godot over UDP. Godot sends your clicks and key presses back as the same commands the 2D window uses. Press `/` to see every control:

- Click a duck to select it, and the camera follows it. One panel shows its needs, moods and wants as bars. Another shows 12,000 of its 139,000 neurons at their real positions in the FlyWire brain, each lighting up as it fires.
- `Tab` lets you ride the selected duck and steer it with W, A, S and D. An overlay shows what its two compound eyes see and how strongly its descending neurons are telling its legs to move. `O` hides the overlay.
- `H` swaps the cursor for a hand. Hold the left button to pick up fruit, a hat, a ball, the drum, the music box or a duck, and release it to put the thing down. Release while the mouse is moving to throw it. To turn the camera, drag with the left button, or with the right button while the hand is out.
- Press `F` or click the tree to shake fruit down. The tree also drops fruit on its own until there are four pieces on the lawn, and shaking it works until there are twelve.
- `G` gives the selected duck a fruit, `P` pets it, and `C` claps.
- `T` drops a hat at the mouse and `B` drops a ball. `D` puts a small drum down at the mouse or picks it back up, and `M` does the same for the music box.

The music box plays your own music. Put mp3s in `~/.cache/micro-garden/music/` and the box plays them in shuffle while it is in the garden. Click the box to change its volume. When you turn it off, the ducks stop hearing it too. No music ships with the project.

## What the ducks do

A duck eats when it is hungry, drinks when it is thirsty, and sleeps when it is tired. A hungry duck follows the smell of fruit upwind, and a thirsty one follows the damp air to the pond. A duck that gets too hot goes to cool off. If it likes water it heads for the pond, and if it does not it heads for the shade of the tree. When it needs nothing, a duck sits or stands around until it gets bored and wanders off.

The ducks form relationships with each other. A duck holds a grudge against a duck that shoves it and makes friends with ducks it spends time with. It keeps track of this separately for every other duck. Each duck can also smell the others from across the garden. Friends seek each other out but stop about half a metre apart, and a duck that gets crowded steps aside. A kind duck goes to comfort one that is crying, and alarm spreads from one duck to the next. A duck's singing pleases sociable ducks and annoys solitary ones.

The ducks also learn how they feel about you. Pet a duck or give it a fruit and it learns to come to you. A duck that trusts you likes being carried, and throwing a duck makes it trust you less. Each duck has a fruit it likes best.

A playful duck chases the ball and taps the drum. A duck that likes music dances while the box plays, and a bored duck may sing or dance even without music, for the ducks nearby to watch. When a duck finds a hat it decides whether to put it on, and vain ducks usually do. A duck that takes its hat off leaves it on the lawn for another duck to find.

Every so often a duck shows its strongest feeling with head movements and sounds, the way a Chao or a Sim does. Chatty ducks do this more often. Ducks with strong personalities also have a signature move: an aggressive duck stomps, a sleepy one yawns, a water lover splashes, a chatty one sings, and a timid one cowers.

## What the brain does and what is explicit

I try to be exact here, because "fly brain drives a duck" is an easy claim to inflate.

Sensory input enters the brain through the fly's real sensory neurons: food and danger smells, humidity, heat and cold, touch, taste, sound and wind on the antennae, and light through a model of the optic lobe ([flyvis](https://github.com/TuragaLab/flyvis)). Movement comes out through real descending neurons, which carry commands from the brain to the body. I found them by screening the whole brain rather than picking names from papers. The clearest one is DNge091. It fires on the side the wind comes from. It does the same on a random seed it was not chosen on, and it stays silent when the wiring is shuffled. A duck that smells food turns into the wind and walks up the odour plume, which is how a real fly finds food. With the real wiring a duck finds a dish in 20 of 20 trials. With the wiring shuffled it finds it in 1 of 20.

Learning also uses the fly's own circuitry. The only synapses that change are the ones between Kenyon cells and mushroom body output neurons. Dopamine controls when they change: one set of dopamine neurons signals reward and another signals punishment.

Some behaviour comes from explicit code rather than the brain, and the code marks each place where that happens. The body code decides how much weight each sense gets, so a hungry duck pays attention to smells on the wind and a fed one does not. The brain's own left-right difference for a smell is weaker than the noise in its steering. So explicit code turns a duck toward music, a ball, other ducks, a person it trusts, and the pond or the shade when it is hot. All the social behaviour between ducks is explicit code as well. Friendships, grudges, comfort, trust and skills are values that change slowly as things happen, and they steer a duck the same way. Personalities are preset combinations of continuous traits, like the Chao's.

## Other ways to run it

The 2D window is a simpler view of the garden:

```
uv run python -m body.stub2d.stub --view --brain
```

Click a duck to follow it and click the tree to shake fruit down. `P` pets the selected duck, `C` claps, `F` drops food at the mouse, `M` adds or removes the music, and `H` puts a hat on a duck or takes it off. `Tab` lets you steer the selected duck with W, A, S and D while its brain keeps running.

To run the ducks as simulated robots, check out [microduck](https://github.com/pollen-robotics/microduck) and [microduck_rl](https://github.com/pollen-robotics/microduck_rl), start five ducks with `DUCK_SIM_DUCKS=5 scripts/duck-sim`, and then start the garden connected to them:

```
uv run python -m body.mujoco.adapter --ducks 5 --brain --godot
```

The simulator runs natively on a Mac. The walking policy that ships upstream does not yet track a commanded velocity ([microduck_rl issue 46](https://github.com/pollen-robotics/microduck_rl/issues/46)), so these ducks walk at one pace and turn on the spot. Either body takes `--godot` to open the Godot garden. Add `--no-window` to skip opening Godot, and then open `viewer/godot` in Godot yourself.

## How it is checked

I check the project with gates. Each gate is a runnable check that exits non-zero when it fails.

```
uv run python -m gates # all of them, which takes hours
uv run python -m gates --fast # the ones that take about a minute
uv run python -m gates 4 8 # just those
```

The fast gates pass. The last full run passed 14 of 15. The one failure was Gate 5's check that a fed duck that is too hot goes to cool off. I have since changed heat to send a duck to the pond or the shade, and I have not rerun Gate 5. Two other results look like problems, but I report them as they are. The five demo personalities are hard to tell apart by numbers alone, so Gate 5 prints that comparison and does not assert it. The Bully's aggression costs it food, and I count that as character, not a bug.

## Where things are

- `garden.py` opens the garden with one command.
- `brain/` loads the connectome and holds the spiking model, the sensory encoder, the motor decoder, learning, bodily needs, the ducks' social life, emotes, and the personality presets.
- `body/` holds the robot command contract, the sensory frame, the 2D body with its retina, and the adapter for the MuJoCo robots.
- `world/` holds the garden: smells carried on the wind, the pond, rocks, temperature, day and night, fruit, balls and music.
- `viewer/` holds the debug window, the world snapshot, and the Godot garden with the scripts that build its robot and record its motions.
- `gates/` holds one check per gate.

## Credit

The connectome is the work of the FlyWire Consortium. The repo does not redistribute its data: you download it into `data/`, which git ignores. The connectivity (`proofread_connections_783.feather`, [Zenodo 10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866)) is CC BY 4.0. The annotations come from [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations), which has no licence file and asks users to cite its papers. Please cite:

- Dorkenwald et al. (2024), "Neuronal wiring diagram of an adult brain", *Nature*.
- Schlegel et al. (2024), "Whole-brain annotation and multi-connectome cell typing of *Drosophila*", *Nature*.
- Matsliah et al. (2024), "Neuronal parts list and wiring diagram for a visual system", *Nature*.
- Berg et al. (2025), male CNS connectome and annotation updates.

This project also builds on:

- [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) (Apache-2.0). The body contract follows robotd's JSON-RPC method and parameter names, and no code is copied.
- [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (Apache-2.0). `viewer/godot/robot.json` is a derived work of the microduck's own meshes, posed from its MuJoCo description and simplified to about 10,000 triangles by `viewer/godot/build_robot.py`, and `viewer/godot/clips.json` records its policies' motions. Its licence is in `viewer/godot/LICENSE-microduck`.
- [TuragaLab/flyvis](https://github.com/TuragaLab/flyvis) (MIT) is the visual front end, installed as a dependency.
- [snedea/flybrain](https://github.com/snedea/flybrain) (MIT) supplied starting LIF parameters and the neuron group map as a reference, and no code is copied.
- [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) (MIT, Shiu et al. 2024) gave the idea of treating glutamate as inhibitory and of reading sugar responses from proboscis motor neurons, and no code is copied.

The model also draws on Gaudry et al. (2013) on asymmetric ORN transmitter release, and on Schretter et al. (2020) and Deutsch et al. (2020) on the aIPg and pC1d/e aggression circuits. Where the model departs from them, the code says so.

The code is MIT licensed, and the FlyWire data has its own terms, listed above.
