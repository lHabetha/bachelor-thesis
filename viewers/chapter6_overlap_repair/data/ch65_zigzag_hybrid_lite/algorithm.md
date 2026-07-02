# 6.5 zig-zag (hybrid-lite overlap + trust-region asm)

Chapter 6.5 strict alternating overlap/assemblability pipeline (v2, #5g), overlap variant `hybrid_lite`, on `strict_overlap_blocked_v2`.

- Successful pipelines shown here: **50** (of 50 starts).

Each successful run is animated as a sequence of **segments** (overlap repair, then assemblability repair, then any further overlap pass). Within a segment the viewer crossfades between the segment's start and end geometry; the on-screen label shows the current cycle, stage, and method. Geometry is generated on demand from each frame's clevis parameters.
