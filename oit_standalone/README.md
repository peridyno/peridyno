# oit_standalone — isolated OIT transparency harness (macOS-friendly)

A self-contained reproduction of PeriDyno's **Order-Independent Transparency (OIT)**
linked-list renderer, with **no dependency on Core / Framework / Topology / CUDA**, so it
builds and runs on macOS. It exists to reproduce and fix the "stacked semi-transparent
boxes lose their front/back order when rotating" bug.

## Why this is separate
The real renderer (`src/Rendering/Engine/OpenGL`) requests an **OpenGL 4.6** context and uses
SSBO + atomic counters + image load/store + SPIR-V — none of which exist on Apple's native GL
(capped at 4.1). This harness reuses the *same* OIT algorithm and shaders but runs against
**Mesa `llvmpipe` software GL (4.6)** through an **EGL surfaceless** context, rendering
offscreen to an FBO and writing a PNG. No window, no GPU, no CUDA.

## Build & run (macOS)
```sh
brew install mesa cmake          # provides libEGL + llvmpipe (in libgallium)
cd oit_standalone
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless
./build/oit_demo out.png --boxes 100 --nodes 40 --yaw 30 --pitch 18           # buggy resolve
./build/oit_demo out.png --boxes 100 --nodes 40 --yaw 30 --pitch 18 --fixed   # fixed resolve
```

### Options
| flag | meaning |
|------|---------|
| `--boxes N`   | number of stacked translucent slabs (default 12) |
| `--yaw / --pitch deg` | orbit camera angles |
| `--fixed`     | use the corrected resolver (`shaders/oit_blend_fixed.frag`) |
| `--reverse`   | draw near→far instead of far→near (exposes the order-dependence) |
| `--nodes M`   | linked-list node budget in millions (default 8). Heavy scenes need more — the tool prints `fragments stored / budget` and warns on overflow |
| `--size WxH`  | output resolution |
| `--checkfrag <transparency.glsl> <x.frag>` | compile-check an engine shader with the Mesa GLSL compiler |

## The bug (and fix)
`blend.frag` walked only the **first `MAX_FRAGMENTS` (128)** nodes of each pixel's list. The
list is head-insertion ordered, so that subset is the **last 128 fragments drawn** — a
*draw-order-dependent* set, not the nearest. Once a pixel has >128 transparent layers (easy when
many boxes overlap), the kept subset — and therefore the composite — changes with draw order and
view, so boxes appear to swap front/back.

**Fix:** walk the whole list and keep the **nearest 128** fragments by depth. Nearest layers
dominate a back-to-front composite, so the result is stable and order-independent. The fragile
`if (depth == previous) continue;` skip was also removed. See `shaders/oit_blend_fixed.frag`.

The same fix is ported back to `src/Rendering/Engine/OpenGL/shader/blend.frag`, and a missing
`glMemoryBarrier` between the OIT build and blend passes was added in
`src/Rendering/Engine/OpenGL/GLRenderEngine.cpp`.

`out/comparison_sweep.png` shows buggy (unstable red center) vs fixed (stable blue center) across
rotation angles.
