# Attribution and licenses

Micro Garden builds on the following work. Data files are downloaded into `data/` (gitignored) and are not redistributed here.

## Data

| Source | What we use | License / terms |
|---|---|---|
| FlyWire whole-brain connectome, v783 connectivity ([Zenodo 10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866)), FlyWire Consortium | `proofread_connections_783.feather`: the synapse-count edge list | CC BY 4.0 |
| [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations) | `Supplemental_file1_neuron_annotations.tsv`: cell classes, types, sides, transmitters, positions | No license file in the repository. The README asks users to cite the papers below. **Open question for a shipped app.** |
| FlyWire Codex downloads (optional) | `connections.csv.gz`, `neurons.csv.gz`, `classification.csv.gz` if present | Codex terms of use (sign-in required); not used so far |

Please cite:
- Dorkenwald et al. (2024), "Neuronal wiring diagram of an adult brain", *Nature*.
- Schlegel et al. (2024), "Whole-brain annotation and multi-connectome cell typing of *Drosophila*", *Nature*.
- Matsliah et al. (2024), "Neuronal parts list and wiring diagram for a visual system", *Nature*.
- Berg et al. (2025), male CNS connectome and annotation updates.

## Software and models

| Project | How it is used | License |
|---|---|---|
| [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) | The body contract follows robotd's JSON-RPC method and parameter names (`robot.move`, `robot.head`, `robot.do`, `robot.sound`, ...); no code copied | Apache-2.0 |
| [TuragaLab/flyvis](https://github.com/TuragaLab/flyvis) | Visual front end (Gate 6), installed as a dependency | MIT |
| [snedea/flybrain](https://github.com/snedea/flybrain) | Starting LIF parameters and the neuron group map read as a reference; no code copied | MIT |
| [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) (Shiu et al. 2024) | The idea of treating glutamate as inhibitory and of reading sugar responses from proboscis motor neurons; no code copied | MIT |

Published findings the model leans on: Gaudry et al. (2013) on asymmetric ORN transmitter release, and Schretter et al. (2020) and Deutsch et al. (2020) on aIPg and pC1d/e aggression circuits. Where the model departs from them, the code says so.
