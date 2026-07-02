# Provenance, Integrity, and Attribution

This document records integrity hashes for the key release artifacts and the
attribution for third-party-derived content.

## Thesis documents

| File | SHA-256 |
|------|---------|
| `thesis/main.pdf` | `ab6a97d6ca3ac689d84c2e0505f44ec4f55656a809e1955c5c35afb11dffbc82` |
| `thesis/declaration-originality-signed.pdf` | `05e1843093bff0e2d752c96c3bee824a7e592a2c8c9f72e37e3e8791a8683ff2` |

The signed Declaration of Originality is appended as the final pages of
`thesis/main.pdf`.

## Supplementary videos

The four Chapter 5 MP4 recordings under
`Supplementary Material to written thesis/videos/` are the videos linked from
the thesis PDF. The clevis recordings were rotated 90° clockwise before MP4
export for upright viewing.

## Key dataset hashes (Chapter 7)

The two large Chapter 7 label files ship gzip-compressed in plain Git
(no Git LFS). Decompress once with `gunzip -k` before running Chapter 7 code.

| File | SHA-256 |
|------|---------|
| `datasets/chapter7_aeroforge_adv/labels_cloud/labels.jsonl.gz` (shipped) | `6cce88df6f2ee48e4cbece47de76d4c8693ccc82df3eba121ed9c430146f32b5` |
| `datasets/chapter7_aeroforge_adv/labels_sim/labels.jsonl.gz` (shipped) | `3c0c70cff27a06cd374dd0bca588cdc5ecc8b99c92bc3fdf96cf784426fc0263` |
| `labels_cloud/labels.jsonl` (after gunzip) | `6204e15565758787aa5fa8b03699d61b838712bb268d78ddcc371ee3b7d1e002` |
| `labels_sim/labels.jsonl` (after gunzip) | `047443a9b35140705ceb6ce91336f738298ab1d4e128a82351206c68293821fa` |

The ADV pool under `datasets/chapter7_aeroforge_adv/advs/` contains exactly
100,000 JSON files.

## Attribution

- **AeroForge / IDEAL Lab.** Chapter 7 builds on the AeroForge
  aircraft-generation pipeline from the IDEAL Lab
  ([`IDEALLab/aeroforge`](https://github.com/IDEALLab/aeroforge)). The upstream
  AeroForge repository is not bundled here; Chapter 7 CAD rendering and full
  quick-sim reruns require a local AeroForge checkout (see
  `code/chapter7_aeroforge/README.md`).
- **AeroSandbox / VLM quick-sim payloads.** `labels_sim/` and the viewer
  `sim_eval.json` files embed vortex-lattice quick-report outputs derived from
  the AeroForge workflow and AeroSandbox-based quick-sim tooling. They are
  included for thesis reproducibility with attribution.

## License summary

| Artifact class | Status |
|----------------|--------|
| Released Python code (`code/`, `viewers/*.py`, `scripts/`) | MIT — see [`LICENSE`](../LICENSE) |
| Thesis PDF (`thesis/main.pdf`) | © Linus Habetha; included for reading and citation |
| Supplementary videos | Included as thesis supplementary material |
| Clevis datasets (Ch. 4–6) | Released for reproducibility; synthetic parametric designs generated in-repo |
| AeroForge ADV pool + labels (Ch. 7) | Released for reproducibility with attribution (see above) |
