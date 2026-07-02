# Hybrid-lite (MLP + bounded direct polish)

Chapter 6.4 overlap-only strict repair (`hybrid_lite__v3__tauvol0`) on the frozen 50-start benchmark `strict_overlap_repair_ch64_v1`.

- Strict success (analytic overlap <= 1e-6): **50/50**
- Mean overlap-volume reduction (incl. failures): **100.0%**
- Mean verifier (analytic overlap) calls: **148**

Each frame shows the clevis at one accepted repair step; the overlap panel reports the analytic overlap volume, the strict `overlap_ok` flag, and the per-pair residuals. Geometry is generated on demand from the frame parameters.
