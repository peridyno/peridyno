# PeriDyno — Project Notes

Peridynamics / particle physics simulation engine. GPU backend is **CUDA** by default
(`PERIDYNO_GPU_BACKEND` = `CUDA` | `Vulkan` | `NoGPU`). Core libs: `Core;Framework;Topology`.

## src/Rendering — module map

```
src/Rendering/
├── Core/          RenderCore lib: OrbitCamera, TrackballCamera, RenderWindow, RenderParams, RenderEngine.h (iface)
├── Engine/
│   ├── OpenGL/    GLRenderEngine — the real-time renderer (the one in use)
│   │   ├── GraphicsObject/   Pure-GL wrappers: Shader/Program, Buffer, VertexArray, Framebuffer, Texture, Mesh, GPUBuffer
│   │   ├── Backend/Cuda|Vulkan   CUDA/VK ↔ GL interop (buffer mapping from sim → GL); selected by GPU backend
│   │   └── shader/           GLSL (compiled to SPIR-V at build via glslangValidator, embedded as *.h)
│   ├── VTK/       Alternative VTK-based engine
│   └── VkRenderEngine/  Vulkan engine
└── GUI/           GlfwGUI, ImGUI, QtGUI, WtGUI, UbiGUI — window/host layers
```

### Dependency / portability facts
- `RenderCore` links `Core;Framework`. `GLRenderEngine` links `Core;Framework;Topology;RenderCore;glad;imgui`.
- `GraphicsObject/*` is **almost pure OpenGL** but `#include "Object.h"` and `"Vector.h"` (dyno Core types) + `glad` + `glm`.
- Sim data reaches GL through `Backend/Cuda` (CUDA-GL interop). This is the **only hard CUDA tie** in the render path.
- Bundled deps in `external/`: `glad-4.6`, `glfw-3.3.0`, `glm-0.9.9.7`, `imgui`, `glslang`.

### ⚠️ macOS hard constraint
- GLFW context requested is **OpenGL 4.6 core** (`GlfwRenderWindow.cpp:105`).
- Shaders use `#version 440/450/460`, **SSBO (std430)**, **atomic_uint counters**, **image load/store
  (`imageAtomicExchange`)**, `early_fragment_tests`, and **SPIR-V program loading**.
- **Apple native OpenGL caps at 4.1** → none of the above are available. The real renderer
  **cannot run on Apple's GL driver**. Options to run on Mac: **Mesa `llvmpipe` software GL (4.5, supports
  SSBO/atomics/images)** [pragmatic], Zink (GL-on-Vulkan via MoltenVK), or a Metal port [large].

## Transparency = Order-Independent Transparency (per-pixel linked list)
Driver: `GLRenderEngine.cpp` Step 4 (`setupTransparencyPass`, ~L113, render ~L402-443).
Shaders: `shader/transparency.glsl` (structs/bindings), `surface.frag::TransparencyLinkedList` (build),
`shader/blend.frag` (sort + composite full-screen pass).

Flow: opaque pass writes depth → transparent pass (`glDepthMask(false)`, depth-test ON) appends each
fragment `{color, depth=gl_FragCoord.z, nextIndex, geometryID, instanceID}` to a per-pixel singly-linked
list (head-index image `r32ui` + atomic free counter + node SSBO) → `blend.frag` walks the list, insertion-sorts
by depth (far→near), composites back-to-front (`mix(dst, src, a)`, transmittance `factor *= 1-a`).

### OIT ordering bug — ROOT CAUSE CONFIRMED + FIXED (the "can't tell box front/back when rotating" report)
Reproduced in `oit_standalone/` (see below) and fixed in both the harness and the engine.
1. **PRIMARY: `blend.frag MAX_FRAGMENTS = 128` truncation kept the wrong subset.** The list is head-insertion
   ordered, so "first 128 walked" = *last 128 fragments drawn* — a draw-order-dependent subset, NOT the
   nearest. Once a pixel has >128 transparent layers (easy with many stacked boxes) the composite depends on
   draw order/view → boxes appear to swap front/back. **Fixed:** walk the whole list, keep the **nearest 128**
   by depth (nearest layers dominate a back-to-front composite → stable, order-independent). Verified: buggy
   forward-vs-reverse draw differs mean 11 / max 185; fixed differs max 16 (imperceptible).
   Also removed the fragile `if (nodes[idx].depth == depth) continue;` skip.
2. **Missing `glMemoryBarrier` between the build pass and the blend read pass** (none existed anywhere in the
   engine) → UB reads of the head-index image + node SSBO. **Fixed:** added
   `glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
   GL_ATOMIC_COUNTER_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT)` in `GLRenderEngine.cpp` after the build loop.
3. **SECONDARY (capacity, not patched): node budget `MAX_OIT_NODES = 8M` (`GLRenderEngine.h:115`).** A heavy
   100-slab scene needed ~28M fragment-nodes; on overflow the global atomic counter drops fragments in a
   draw-order-dependent way → additional artifacts. Bump it (memory = 32B × nodes; 8M = 256MB) if scenes are
   very dense. The harness prints `fragments stored / budget` and warns on overflow.

Files changed: `shader/blend.frag` (nearest-K resolve), `GLRenderEngine.cpp` (memory barrier).
Note: `surface.frag` and `car.frag` both feed the linked list but share the single `blend.frag` resolve, so
the one shader fix covers all transparent objects.

## oit_standalone/ — isolated, macOS-runnable OIT harness
Self-contained C++/glad app (no Core/Framework/CUDA) reusing the OIT algorithm + shaders. Runs on **Mesa
`llvmpipe` software GL via EGL surfaceless**, renders stacked translucent boxes offscreen → PNG. Build:
`brew install mesa`, then cmake; run with `LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless`.
See `oit_standalone/README.md`. `--fixed` toggles the corrected resolver; `--reverse` exposes the
order-dependence; `--checkfrag` compile-checks engine shaders with the Mesa GLSL compiler.

## Building / running
- Top-level build is CUDA (won't configure without CUDA). For isolated Mac render work, build a standalone
  OIT harness reusing `shader/*` + the GraphicsObject classes against Mesa `llvmpipe` (offscreen FBO → PNG).
- Toolchain present: Homebrew `/opt/homebrew`, `cmake`, `glfw`/`glew` installed, `mesa` installable (26.x).
