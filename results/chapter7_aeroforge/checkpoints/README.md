# Chapter 7 Checkpoints

Thesis-facing overlap surrogate and performance MLP weights for AeroForge repair
(Chapter 7, §7.3–§7.5).

---

## Checkpoints

| File | Size | Role |
|------|------|------|
| `eng_multitask_gate_strong_100k.pt` | ~229 KB | **Headline overlap brain** (variant A); `tab:aeroforge-overlap-repair` |
| `eng_log_huber_100k.pt` | ~228 KB | Grid column B; `tab:aeroforge-surrogate-results` |
| `eng_gate_aug_strong_100k.pt` | ~228 KB | Best gate F1 in surrogate CV table |
| `raw_log_huber_100k.pt` | ~221 KB | Grid column E (raw features) |
| `performance_mlp_90k.pt` | ~222 KB | Performance-preserving repair (`sec:aeroforge-performance-preserving`) |
| `registry.json` | ~3 KB | Index of overlap surrogates |

Training labels: `datasets/chapter7_aeroforge_adv/labels_cloud/labels.jsonl` (overlap)
and `labels_sim/labels.jsonl` (performance MLP) — decompress the shipped
`.jsonl.gz` archives first (see the dataset README).

Verify integrity after cloning with `shasum -a 256 *.pt` against the hashes in
`docs/provenance.md` and `results/chapter7_aeroforge/README.md`.
