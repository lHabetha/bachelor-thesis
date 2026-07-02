# Bachelor Thesis — Public Release

**Surrogate-Based Assemblability Prediction and Optimization for Parametric Mechanical Assemblies**

Linus Habetha · ETH Zurich · IDEAL Lab · Spring Semester 2026

This repository is the public release accompanying the thesis: the submitted
PDF, the code, datasets, frozen results, and three localhost viewers needed to
inspect or rerun what the thesis reports.

Start with [`thesis/main.pdf`](thesis/main.pdf) for the full narrative. Use this
README for navigation, reproduction order, and launch commands.

## Quick start — inspect thesis results

1. Read the PDF and supplementary videos (see below).
2. Create the recommended environment:

   ```bash
   conda env create -f environment.yml
   conda activate bachelor-thesis
   ```

3. Launch a viewer (metrics work with plain Python 3; 3D render needs CadQuery):

   ```bash
   ./scripts/view_ch5.sh   # http://127.0.0.1:8090
   ./scripts/view_ch6.sh   # http://127.0.0.1:8091
   ./scripts/view_ch7.sh   # http://127.0.0.1:8092
   ```

4. Each viewer dropdown is driven by an allowlist in
   [`release_manifests/`](release_manifests/) — only thesis-table runs appear.

Frozen figures and tables under [`results/`](results/) match the PDF without
rerunning anything.

## Repository layout

| Path | Purpose |
|------|---------|
| [`thesis/main.pdf`](thesis/main.pdf) | Submitted thesis with signed declaration appended |
| [`thesis/declaration-originality-signed.pdf`](thesis/declaration-originality-signed.pdf) | Signed Declaration of Originality source PDF |
| [`Supplementary Material to written thesis/videos/`](Supplementary%20Material%20to%20written%20thesis/videos/) | Four Chapter 5 MP4s linked from the PDF |
| [`code/`](code/) | Public Python code by chapter (`chapter3_*` … `chapter7_*`) |
| [`datasets/`](datasets/) | Released pools, labels, and benchmarks |
| [`results/`](results/) | Frozen CSVs, figures, checkpoints, and summaries |
| [`viewers/`](viewers/) | Three allowlist-gated localhost viewers (Ch. 5–7) |
| [`release_manifests/`](release_manifests/) | Viewer allowlists (which runs the viewers serve) |
| [`docs/`](docs/) | [Environment](docs/environment.md) and [provenance](docs/provenance.md) notes |
| [`scripts/`](scripts/) | Viewer launchers (`view_ch5.sh`, `view_ch6.sh`, `view_ch7.sh`) |
| [`environment.yml`](environment.yml) | Recommended conda environment for full CAD/viewer reproducibility |
| [`requirements.txt`](requirements.txt) | Pip reference/fallback; conda is preferred for CadQuery |
| [`CITATION.cff`](CITATION.cff) | GitHub citation metadata |
| [`LICENSE`](LICENSE) | MIT license for **released code** in this repository |

## Reproduction order (by chapter)

Work through chapters in order when rerunning pipelines. Each step lists the
chapter README for commands; frozen outputs already ship under `results/`.

| Step | Chapter | Code | Datasets | Frozen results |
|------|---------|------|----------|----------------|
| 1 | Setup & labels | [`code/chapter3_clevis_setup/`](code/chapter3_clevis_setup/) | — | [`results/chapter3_clevis_setup/`](results/chapter3_clevis_setup/) |
| 2 | Label efficiency | [`code/chapter4_learning/`](code/chapter4_learning/) | [`datasets/chapter4_clevis_pools/`](datasets/chapter4_clevis_pools/) | [`results/chapter4_label_efficiency/`](results/chapter4_label_efficiency/) |
| 3 | Optimization | [`code/chapter5_optimization/`](code/chapter5_optimization/) | [`datasets/chapter5_blocked200/`](datasets/chapter5_blocked200/) | [`results/chapter5_optimization/`](results/chapter5_optimization/) |
| 4 | Overlap extension | [`code/chapter6_overlap/`](code/chapter6_overlap/) | [`datasets/chapter6_overlap_clevis/`](datasets/chapter6_overlap_clevis/) | [`results/chapter6_overlap/`](results/chapter6_overlap/) |
| 5 | AeroForge transfer | [`code/chapter7_aeroforge/`](code/chapter7_aeroforge/) | [`datasets/chapter7_aeroforge_adv/`](datasets/chapter7_aeroforge_adv/) | [`results/chapter7_aeroforge/`](results/chapter7_aeroforge/) |

**Typical rerun pattern:** activate the conda environment (see
[`docs/environment.md`](docs/environment.md)), `cd` to `bachelor-thesis/`,
`export PYTHONPATH=code`, then run the module documented in the chapter README.
New outputs write under `results/<chapter>/`.

**Fastest path to a figure:** many chapters ship plot scripts that read frozen
CSVs under `results/` (for example
`python -m chapter7_aeroforge.ml.plot_100k_model_grid`).

## Viewers

| Viewer | Allowlist | Runs | Port |
|--------|-----------|------|------|
| [`viewers/chapter5_clevis_optimization/`](viewers/chapter5_clevis_optimization/) | [`ch5_allowlist.json`](release_manifests/ch5_allowlist.json) | 88 entries / 80 folders | 8090 |
| [`viewers/chapter6_overlap_repair/`](viewers/chapter6_overlap_repair/) | [`ch6_allowlist.json`](release_manifests/ch6_allowlist.json) | 10 entries | 8091 |
| [`viewers/chapter7_aeroforge_repair/`](viewers/chapter7_aeroforge_repair/) | [`ch7_allowlist.json`](release_manifests/ch7_allowlist.json) | 49 entries | 8092 |

See [`viewers/README.md`](viewers/README.md) for scope and dependencies.

## Supplementary videos

Four Chapter 5 recordings live under
`Supplementary Material to written thesis/videos/`. Link from the PDF, for example:

```
https://github.com/lHabetha/bachelor-thesis/blob/main/Supplementary%20Material%20to%20written%20thesis/videos/<filename>.mp4
```

## License and redistribution

- **Released code** in this repository: [MIT License](LICENSE).
- **Thesis PDF**: © Linus Habetha; included for reading and citation. Confirm ETH /
  supervisor requirements before redistributing outside this repository.
- **Datasets and checkpoints**: released for thesis reproducibility. Chapter 7
  `labels_sim/` and viewer `sim_eval.json` files embed VLM quick-sim payloads
  derived from the IDEAL Lab AeroForge workflow and AeroSandbox; they are
  included for thesis reproducibility with attribution documented in
  [`docs/provenance.md`](docs/provenance.md).
- **Large artifacts**: the two Chapter 7 label JSONL files ship gzip-compressed
  (`labels.jsonl.gz`) so no Git LFS is needed. Chapter 7 viewer JSON has been
  compacted to keep it in normal Git; meshes are rendered locally on demand.

## Large Files And External Dependencies

- **Chapter 7 labels:** the two large label files are stored as plain-Git gzip
  archives (`labels_cloud/labels.jsonl.gz`, `labels_sim/labels.jsonl.gz`).
  Decompress them once before running Chapter 7 code:
  `gunzip -k datasets/chapter7_aeroforge_adv/labels_{cloud,sim}/labels.jsonl.gz`
  (details in
  [`datasets/chapter7_aeroforge_adv/README.md`](datasets/chapter7_aeroforge_adv/README.md)).
- **Clone size:** Chapter 5 viewer JSON is intentionally included in normal Git
  so thesis-table runs can be inspected offline; this makes the repository large.
- **AeroForge:** Chapter 7 CAD rendering and full quick-sim reruns need a sibling
  [`IDEALLab/aeroforge`](https://github.com/IDEALLab/aeroforge) checkout or an
  `AEROFORGE_ROOT` override. Metrics-only viewer use does not require it.

## Citation

If you use this release, please cite the thesis PDF and this repository:

`https://github.com/lHabetha/bachelor-thesis`

GitHub also reads [`CITATION.cff`](CITATION.cff). A formal DOI can be added later
if one is minted.
