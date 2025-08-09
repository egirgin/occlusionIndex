# Occlusion Index

This repository provides the official implementation for our papers, which introduce novel metrics for quantifying occlusion in multi-person 3D scenes. We provide the code to compute our proposed metrics—**Occlusion Index (OI)**, **Silhouette Occlusion Index (SOI)**, and **Crowd Index (CI)**—on the 3DPW, AGORA, and OCHuman datasets.

---

## 📚 Citation

If you use this code or our metrics in your research, please cite the relevant papers:

```bibtex
@inproceedings{girgin2024detection,
  title={Detection and Quantification of Occlusion for 3D Human Pose and Shape Estimation},
  author={Girgin, Emre and G{\"o}kberk, Berk and Akarun, Lale},
  booktitle={International Conference on Pattern Recognition},
  pages={368--382},
  year={2024},
  organization={Springer}
}

@inproceedings{girgin2023novel,
  title={A novel occlusion index},
  author={Girgin, Emre and G{\"o}kberk, Berk and Akarun, Lale},
  booktitle={2023 31st Signal Processing and Communications Applications Conference (SIU)},
  pages={1--4},
  year={2023},
  organization={IEEE}
}
```

-----

## Getting Started

### 1\. Prerequisites

  - Python 3.x
  - A Python 3 compatible version of SMPL. We recommend [egirgin/smpl\_basics](https://github.com/egirgin/smpl_basics).

### 2\. Installation & Setup

Install the dependencies.

```bash
pip install -r requirements.txt
```

### 3\. Download Datasets

Download the following datasets and `dataset_config` variable of the corresponding dataset processing script.

  - **3DPW:** [Link](https://virtualhumans.mpi-inf.mpg.de/3DPW/evaluation.html)
  - **AGORA:** [Link](https://agora.is.tue.mpg.de/index.html)
  - **OCHuman:** [Link](https://github.com/liruilong940607/OCHumanApi)

**Important:** Before running, ensure all file paths in the scripts are updated to point to your dataset and SMPL model locations.

-----

## Reproducing Results

The code is organized by dataset.

### 3DPW

1.  **Generate SMPL models:** `python 3dpw/generate_smpl/generate-smpl.py`
2.  **Calculate SOI:** `python 3dpw/render_smpl/silhouette_occlusion_index.py`
3.  **Calculate OI & CI:** `python 3dpw/src/new_main.py`
4.  **Run Experiments:** `python 3dpw/src/experiments/experiments.py`

### AGORA

1.  **Preprocess:** `python agora/preprocess.py`
2.  **Calculate SOI:** `python agora/regional_occlusion.py`
3.  **Calculate OI & CI:** `python agora/new_main.py`
4.  **Run Experiments:** `python agora/experiments/experiments.py`

### OCHuman

1.  **Unzip Data:** `unzip ochuman/gtSMPL.zip`
2.  **Calculate OI & CI:** `python ochuman/src/new_main.py` *(SOI is not applicable)*
3.  **Run Experiments:** `python ochuman/src/experiments/experiments.py`

-----

## Pre-computed Results

Pre-computed results are available in the `/results` folder for direct comparison.

-----

## License

This project is licensed under the [Your License] License. See the `LICENSE.md` file for details.
