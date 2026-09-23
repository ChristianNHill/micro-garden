# Micro Garden

![Five microducks in the Godot garden eating, swimming, playing a bell and laughing at each other, with the news of what they do at the top](docs/garden.gif)

Micro Garden is five small ducks in a garden, and each duck runs on its own live copy of a real fruit fly brain. I built it to ask whether a wiring diagram, a body with needs, and a place to live are enough to get behaviour nobody scripted. The brain is the FlyWire connectome, about 139,000 neurons and 2.7 million connections, simulated as spiking neurons. Each duck smells, sees and hears. It gets hungry, thirsty, hot and sleepy. It learns, makes friends and holds grudges, and each one has a personality.

The ducks are modelled on the [microduck](https://github.com/pollen-robotics/microduck), a real open-source robot, and the brain talks to the body through that robot's own command protocol. The body can be a simple 2D simulation in Python or the microduck's own MuJoCo simulation. Either way, a low-poly 3D garden in Godot shows what the ducks are doing. Everything runs in simulation, and I have no plans to build the physical robots.

## Run it

You need Python 3.12, [uv](https://docs.astral.sh/uv/), [Godot 4](https://godotengine.org), and a machine PyTorch supports. I use an M4 Pro. The garden looks for Godot at the path in `GODOT`, then on your path, then in `/Applications`. The repo does not redistribute the FlyWire data, so download two files into `data/`:

- `proofread_connections_783.feather` from the [FlyWire v783 release on Zenodo](https://doi.org/10.5281/zenodo.10676866)
- `Supplemental_file1_neuron_annotations.tsv` from [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)

Then open the garden:

```
uv run python -m garden
```

Closing the window saves the garden, and the ducks are older by however long you were away when you come back. Each new duck gets a random name and personality. Add `--fresh` to start with new ducks, `--labels Bully,Napper,Carefree,Chatty,Scaredy` to choose the personalities, and `--blind` to hide which duck has which until you quit. Five ducks run at about 0.6 times real time on an M4 Pro, and anything else on the GPU slows them more.

## The garden

The garden is modelled on the Chao gardens of Sonic Adventure 2: a lawn inside rock walls, a pond under a waterfall, a fruit tree, and a cliff above the sea. The wind turns slowly over the day, and flags and reeds show which way it blows. That matters because the ducks find food by smelling it on the wind.

![Fennel the Musician selected, with its card, its needs, moods and wants as bars, and its brain firing in the corner](docs/selected-duck.webp)

Click a duck to follow it and see its needs, moods and 12,000 of its neurons firing at their real positions. Press `/` to see every control. The main ones are:

- `Tab` rides the selected duck with W, A, S and D, and shows what its eyes see.
- `H` gives you a hand to pick up and throw fruit, hats, balls, instruments or ducks.
- `F` shakes the tree, `G` gives the selected duck a fruit, `P` pets it, and `C` claps.
- `T` drops a hat, `B` a ball, `D` a drum, `M` the music box, and `I` another instrument.

The music box plays mp3s you put in `~/.cache/micro-garden/music/`. No music ships with the project.

## What the ducks do

A duck eats when it is hungry, drinks when it is thirsty, sleeps when it is tired, and cools off in the pond or the shade when it is hot. It makes friends with ducks it spends time with and holds grudges against ducks that shove it. It learns to trust you if you pet and feed it. It plays with balls and instruments, dances to music, and tries on hats.

Ducks get better at swimming, walking, eating, fighting, dancing, fashion and music by doing them, and a clumsy duck trips, misses or hits a sour note now and then. Other ducks may laugh, comfort, or pile on, and one reaction can set off another. Personality makes each of these likelier or rarer but never rules it out.

## What the brain does and what is explicit

"Fly brain drives a duck" is an easy claim to inflate, so I try to be exact.

Senses enter the brain through the fly's real sensory neurons, with vision through a model of the optic lobe ([flyvis](https://github.com/TuragaLab/flyvis)). Movement comes out through real descending neurons, which I found by screening the whole brain. The clearest is DNge091, which fires on the side the wind comes from. A duck that smells food walks up the odour plume the way a real fly does. With the real wiring a duck finds a dish in 20 of 20 trials, and with the wiring shuffled it finds it in 1 of 20. Learning changes only the synapses between Kenyon cells and mushroom body output neurons, gated by reward and punishment dopamine neurons.

Some behaviour is explicit code, and the code marks each place. The body code decides how much weight each sense gets. The brain's own left-right signal for most smells is weaker than its steering noise, so explicit code turns a duck toward music, balls, other ducks, you, and the pond or shade. All social behaviour, trust and skills are explicit too. Personalities are preset combinations of continuous traits, and a garden draws five of 19.

## Other ways to run it

The 2D window is a simpler view:

```
uv run python -m body.stub2d.stub --view --brain
```

To run the ducks as simulated robots, check out [microduck](https://github.com/pollen-robotics/microduck) and [microduck_rl](https://github.com/pollen-robotics/microduck_rl), start five with `DUCK_SIM_DUCKS=5 scripts/duck-sim`, then run:

```
uv run python -m body.mujoco.adapter --ducks 5 --brain --godot
```

The upstream walking policy does not yet track a commanded velocity ([microduck_rl issue 46](https://github.com/pollen-robotics/microduck_rl/issues/46)), so these ducks walk at one pace and turn on the spot.

## How it is checked

Each gate is a runnable check that exits non-zero when it fails.

```
uv run python -m gates # all of them, which takes hours
uv run python -m gates --fast # about a minute
uv run python -m gates 4 8 # just those
```

The fast gates pass. The last full run passed 14 of 15, failing Gate 5's check that a hot duck goes to cool off. I have since changed how heat, walking speed and skills work and have not rerun the slow gates.

## Where things are

- `garden.py` opens the garden.
- `brain/` holds the connectome loader, the spiking model, senses, motor output, learning, needs, social life and personalities.
- `body/` holds the robot command contract, the 2D body and the MuJoCo adapter.
- `world/` holds the garden itself: wind, smells, water, weather, fruit and toys.
- `viewer/` holds the debug window and the Godot garden.
- `gates/` holds one check per gate.

## Credit

The connectome is the work of the FlyWire Consortium. The repo does not redistribute its data: you download it into `data/`, which git ignores. The connectivity ([Zenodo 10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866)) is CC BY 4.0. The annotations come from [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations), which has no licence file and asks users to cite its papers. Please cite:

- Dorkenwald et al. (2024), "Neuronal wiring diagram of an adult brain", *Nature*.
- Schlegel et al. (2024), "Whole-brain annotation and multi-connectome cell typing of *Drosophila*", *Nature*.
- Matsliah et al. (2024), "Neuronal parts list and wiring diagram for a visual system", *Nature*.
- Berg et al. (2025), male CNS connectome and annotation updates.

This project also builds on:

- [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) (Apache-2.0). The body contract follows robotd's JSON-RPC names, and no code is copied.
- [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (Apache-2.0). `viewer/godot/robot.json` is derived from the microduck's meshes and `viewer/godot/clips.json` records its policies' motions. Its licence is in `viewer/godot/LICENSE-microduck`.
- [TuragaLab/flyvis](https://github.com/TuragaLab/flyvis) (MIT) is the visual front end.
- [snedea/flybrain](https://github.com/snedea/flybrain) (MIT) supplied starting LIF parameters and the neuron group map as a reference, and no code is copied.
- [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) (MIT, Shiu et al. 2024) gave the idea of treating glutamate as inhibitory and reading sugar responses from proboscis motor neurons, and no code is copied.

The model also draws on Gaudry et al. (2013), Schretter et al. (2020), Deutsch et al. (2020), and Bailey and Katchabaw's stimulus-and-response framework. Where the model departs from them, the code says so.

The code is MIT licensed, and the FlyWire data has its own terms, listed above.
