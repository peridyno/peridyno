// Standalone reproduction of PeriDyno's Order-Independent-Transparency (OIT)
// linked-list renderer, isolated from Core/Framework/CUDA so it builds & runs on
// macOS via Mesa software GL (OSMesa, OpenGL 4.3+). Renders a stack of
// semi-transparent boxes offscreen and writes a PNG.
//
// Usage: oit_demo <out.png> [--boxes N] [--yaw deg] [--pitch deg] [--fixed] [--size WxH]

#include <glad/glad.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

// ---------------------------------------------------------------- node layout
// std430 struct: vec4(16) + float(4) + uint(4) + int(4) + int(4) = 32 bytes
struct Node { float c[4]; float depth; unsigned next; int geom; int inst; };
static const size_t NODE_SIZE = 32;
static unsigned MAX_NODES = 8u * 1024u * 1024u;   // node budget (overridable via --nodes M)

static std::string g_shaderDir = "shaders";
static float g_radius = 14.f;   // camera orbit distance (overridable via --radius)

static std::string readFile(const std::string& p) {
    std::ifstream f(p);
    if (!f) { fprintf(stderr, "cannot open shader %s\n", p.c_str()); exit(1); }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

static GLuint compile(GLenum type, const std::string& src, const char* tag) {
    GLuint s = glCreateShader(type);
    const char* p = src.c_str();
    glShaderSource(s, 1, &p, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096]; glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "shader compile failed (%s):\n%s\n", tag, log); exit(1);
    }
    return s;
}

static GLuint program(const std::string& vsf, const std::string& fsf) {
    GLuint vs = compile(GL_VERTEX_SHADER,   readFile(g_shaderDir + "/" + vsf), vsf.c_str());
    GLuint fs = compile(GL_FRAGMENT_SHADER, readFile(g_shaderDir + "/" + fsf), fsf.c_str());
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[4096]; glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        fprintf(stderr, "link failed (%s/%s):\n%s\n", vsf.c_str(), fsf.c_str(), log); exit(1); }
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

// unit cube, 36 verts (pos + normal), no index, both faces kept (no culling)
static void buildCube(GLuint& vao, GLuint& vbo) {
    const float f[] = {
        // pos               normal
        -1,-1,-1,  0,0,-1,  1,1,-1, 0,0,-1,  1,-1,-1, 0,0,-1,  -1,-1,-1,0,0,-1,  -1,1,-1,0,0,-1,  1,1,-1,0,0,-1,
        -1,-1, 1, 0,0,1,    1,-1,1, 0,0,1,   1,1,1,  0,0,1,    -1,-1,1, 0,0,1,    1,1,1, 0,0,1,    -1,1,1, 0,0,1,
        -1,1,1, -1,0,0,     -1,1,-1,-1,0,0,  -1,-1,-1,-1,0,0,  -1,-1,-1,-1,0,0,   -1,-1,1,-1,0,0,  -1,1,1,-1,0,0,
         1,1,1,  1,0,0,      1,-1,-1,1,0,0,   1,1,-1, 1,0,0,    1,-1,-1, 1,0,0,    1,1,1, 1,0,0,    1,-1,1, 1,0,0,
        -1,-1,-1,0,-1,0,     1,-1,-1,0,-1,0,  1,-1,1, 0,-1,0,  -1,-1,-1,0,-1,0,   1,-1,1, 0,-1,0,  -1,-1,1,0,-1,0,
        -1,1,-1, 0,1,0,      1,1,1,  0,1,0,   1,1,-1, 0,1,0,   -1,1,-1, 0,1,0,    -1,1,1, 0,1,0,    1,1,1, 0,1,0,
    };
    glGenVertexArrays(1, &vao); glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(f), f, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

// deterministic pseudo-random in [-1,1] from an integer + salt (no rng needed,
// reproducible across runs so the "physics pile" view is stable)
static float jit(int i, int salt) {
    float v = std::sin((float)i * 12.9898f + (float)salt * 78.233f) * 43758.5453f;
    return 2.f * (v - std::floor(v)) - 1.f;
}

// rainbow color for box i of n
static glm::vec3 hue(float t) {
    float r = 0.5f + 0.5f * std::cos(6.2831853f * (t + 0.00f));
    float g = 0.5f + 0.5f * std::cos(6.2831853f * (t + 0.33f));
    float b = 0.5f + 0.5f * std::cos(6.2831853f * (t + 0.66f));
    return glm::vec3(r, g, b);
}

int main(int argc, char** argv) {
    const char* out = "out.png";
    int   nBoxes = 12;
    float yaw = 35.f, pitch = 20.f;
    bool  fixed = false;
    bool  reverse = false;   // draw far->near (default) or near->far (--reverse)
    int   W = 900, H = 700;
    std::string scene = "slabs"; // "slabs" | "nested" | "cluster" | "packed"
    float boxAlpha = 0.35f;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--boxes" && i + 1 < argc) nBoxes = atoi(argv[++i]);
        else if (a == "--yaw"   && i + 1 < argc) yaw = atof(argv[++i]);
        else if (a == "--pitch" && i + 1 < argc) pitch = atof(argv[++i]);
        else if (a == "--fixed") fixed = true;
        else if (a == "--reverse") reverse = true;
        else if (a == "--scene" && i + 1 < argc) scene = argv[++i];
        else if (a == "--alpha" && i + 1 < argc) boxAlpha = atof(argv[++i]);
        else if (a == "--radius" && i + 1 < argc) g_radius = atof(argv[++i]);
        else if (a == "--shaders" && i + 1 < argc) g_shaderDir = argv[++i];
        else if (a == "--nodes" && i + 1 < argc) MAX_NODES = (unsigned)atoi(argv[++i]) * 1024u * 1024u;
        else if (a == "--size" && i + 1 < argc) { sscanf(argv[++i], "%dx%d", &W, &H); }
        else if (a[0] != '-') out = argv[i];
    }

    // ---- EGL surfaceless software GL context (Mesa llvmpipe, OpenGL 4.5 core) ----
    // We render entirely to our own FBO, so no window-system surface is needed.
    auto eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC)
        eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay dpy = EGL_NO_DISPLAY;
    if (eglGetPlatformDisplayEXT)
        dpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, nullptr, nullptr);
    if (dpy == EGL_NO_DISPLAY)
        dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY) { fprintf(stderr, "eglGetDisplay failed\n"); return 1; }

    EGLint major = 0, minor = 0;
    if (!eglInitialize(dpy, &major, &minor)) { fprintf(stderr, "eglInitialize failed\n"); return 1; }
    if (!eglBindAPI(EGL_OPENGL_API)) { fprintf(stderr, "eglBindAPI(OpenGL) failed\n"); return 1; }

    const EGLint cfgAttribs[] = {
        EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_NONE
    };
    EGLConfig cfg; EGLint nCfg = 0;
    if (!eglChooseConfig(dpy, cfgAttribs, &cfg, 1, &nCfg) || nCfg < 1) {
        fprintf(stderr, "eglChooseConfig failed\n"); return 1; }

    const EGLint ctxAttribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 5,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE
    };
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctxAttribs);
    if (ctx == EGL_NO_CONTEXT) { fprintf(stderr, "eglCreateContext failed (0x%x)\n", eglGetError()); return 1; }
    if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
        fprintf(stderr, "eglMakeCurrent failed (0x%x)\n", eglGetError()); return 1; }

    if (!gladLoadGLLoader((GLADloadproc)eglGetProcAddress)) {
        fprintf(stderr, "gladLoadGLLoader failed\n"); return 1; }
    printf("GL %s | %s\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));

    // --checkfrag <transparency.glsl> <shader.frag> : compile-check the engine's
    // real blend.frag (which #includes transparency.glsl) using the Mesa compiler.
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--checkfrag" && i + 2 < argc) {
            std::string inc = readFile(argv[i + 1]);
            std::string body = readFile(argv[i + 2]);
            // strip "#version", "#extension ... include", and the "#include" line
            std::stringstream in(body), outss; std::string line; std::string ver = "#version 440\n";
            while (std::getline(in, line)) {
                if (line.find("#version") != std::string::npos) { ver = line + "\n"; continue; }
                if (line.find("#include") != std::string::npos) continue;
                if (line.find("GL_GOOGLE_include_directive") != std::string::npos) continue;
                outss << line << "\n";
            }
            std::string merged = ver + inc + "\n" + outss.str();
            GLuint s = compile(GL_FRAGMENT_SHADER, merged, argv[i + 2]); // exits on failure
            printf("OK: %s compiles cleanly under %s\n", argv[i + 2], glGetString(GL_VERSION));
            glDeleteShader(s);
            eglTerminate(dpy);
            return 0;
        }
    }

    // ---- our own FBO (so we control formats; OSMesa backbuf is just to satisfy MakeCurrent) ----
    GLuint colorTex, depthRb, fbo;
    glGenTextures(1, &colorTex);
    glBindTexture(GL_TEXTURE_2D, colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glGenRenderbuffers(1, &depthRb);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, W, H);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRb);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "FBO incomplete\n"); return 1; }

    // ---- OIT resources ----
    GLuint headTex;            // r32ui head-index image
    glGenTextures(1, &headTex);
    glBindTexture(GL_TEXTURE_2D, headTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, W, H, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

    GLuint atomicBuf;          // free-node counter
    glGenBuffers(1, &atomicBuf);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicBuf);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);

    GLuint nodeBuf;            // linked-list node SSBO
    glGenBuffers(1, &nodeBuf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, nodeBuf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)NODE_SIZE * MAX_NODES, nullptr, GL_DYNAMIC_DRAW);

    GLuint cubeVao, cubeVbo;  buildCube(cubeVao, cubeVbo);
    GLuint emptyVao;          glGenVertexArrays(1, &emptyVao);

    GLuint buildProg = program("box.vert", "oit_build.frag");
    GLuint blendProg = program("fullscreen.vert", fixed ? "oit_blend_fixed.frag" : "oit_blend.frag");

    // ---- camera ----
    glm::vec3 target(0, 0, 0);
    float radius = g_radius;
    float yr = glm::radians(yaw), pr = glm::radians(pitch);
    glm::vec3 eye = target + radius * glm::vec3(std::cos(pr) * std::sin(yr),
                                               std::sin(pr),
                                               std::cos(pr) * std::cos(yr));
    glm::mat4 view = glm::lookAt(eye, target, glm::vec3(0, 1, 0));
    glm::mat4 proj = glm::perspective(glm::radians(45.f), (float)W / H, 0.5f, 100.f);

    // ---- per-box model transforms: a diagonal staircase of overlapping slabs ----
    std::vector<glm::mat4> models;
    std::vector<glm::vec3> colors;
    for (int i = 0; i < nBoxes; i++) {
        float t = (nBoxes > 1) ? (float)i / (nBoxes - 1) : 0.f;
        glm::mat4 m(1.f);
        if (scene == "nested") {
            // concentric SOLID cubes of growing size, all centered: looks like real
            // boxes one-inside-another; every pixel through the center pierces all of
            // them (front + back face) -> ~2*nBoxes overlapping layers, clearly boxes.
            float s = 1.0f + t * 5.0f;           // 1 .. 6 half-extent
            m = glm::scale(glm::mat4(1.f), glm::vec3(s));
        } else if (scene == "cluster") {
            // a settled "physics pile": independent solid cubes that DO NOT
            // interpenetrate in 3D (grid spacing > cube size + jitter + rotation
            // slack) but still overlap in SCREEN space, so OIT must correctly
            // composite a box seen through another box. Some stacked, some scattered.
            const float half    = 1.1f;          // cube half-extent (size 2.2)
            const float spacing = 4.3f;          // > size*sqrt(2)+jitter (no interpenetration)
            int perLayer = 9;                    // 3x3 footprint per layer
            int idx = i % perLayer, layer = i / perLayer;
            int gx = idx % 3, gz = idx / 3;
            // each higher layer rests on the one below, with a lateral drift like
            // cubes that came to rest slightly off-center
            glm::vec3 pos(
                (gx - 1) * spacing + jit(i, 1) * 0.35f + layer * 0.8f,
                layer * (2.f * half + 0.4f) - 1.5f,                  // stacked, clear vertical gap
                (gz - 1) * spacing + jit(i, 2) * 0.35f - layer * 0.5f);
            m = glm::translate(glm::mat4(1.f), pos);
            // small settle rotation; kept within the spacing slack so cubes never touch
            m = glm::rotate(m, glm::radians(jit(i, 3) * 12.f), glm::vec3(0, 1, 0));
            m = glm::rotate(m, glm::radians(jit(i, 4) * 6.f),  glm::vec3(1, 0, 0));
            m = glm::scale(m, glm::vec3(half));
        } else if (scene == "packed") {
            // dense 3D grid of solid cubes packed face-to-face with a hair of gap so
            // they read as separate boxes and never interpenetrate. Increasing
            // --boxes packs more layers along every view ray (heavier OIT load).
            int side = (int)std::ceil(std::cbrt((double)nBoxes));
            int gx = i % side, gy = (i / side) % side, gz = i / (side * side);
            const float half = 1.0f;                 // cube half-extent (size 2.0)
            const float step = 2.0f * half * 1.05f;  // 5% gap -> visibly separate, no overlap
            float c = (side - 1) * 0.5f;             // center the grid on origin
            glm::vec3 pos((gx - c) * step, (gy - c) * step, (gz - c) * step);
            m = glm::translate(glm::mat4(1.f), pos);
            m = glm::scale(m, glm::vec3(half));
        } else {
            // default "slabs": thin plates stacked along depth with lateral drift ->
            // heavy single-pixel layer count for OIT stress testing.
            glm::vec3 pos((t - 0.5f) * 3.0f, (t - 0.5f) * 3.0f, (t - 0.5f) * 13.f);
            m = glm::translate(glm::mat4(1.f), pos);
            m = glm::scale(m, glm::vec3(3.0f, 3.0f, 0.12f));
        }
        models.push_back(m);
        colors.push_back(hue(t));
    }

    const float alpha = boxAlpha;
    glm::vec3 bg(0.12f, 0.13f, 0.16f);

    GLint uMVP   = glGetUniformLocation(buildProg, "uMVP");
    GLint uModel = glGetUniformLocation(buildProg, "uModel");
    GLint uColor = glGetUniformLocation(buildProg, "uColor");
    GLint uAlpha = glGetUniformLocation(buildProg, "uAlpha");
    GLint uGeom  = glGetUniformLocation(buildProg, "uGeomID");
    GLint uMaxN  = glGetUniformLocation(buildProg, "uMaxNodes");
    GLint uBg    = glGetUniformLocation(blendProg, "uBackground");

    glViewport(0, 0, W, H);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    GLenum drawbuf = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawbuf);
    glClearColor(bg.r, bg.g, bg.b, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // ===== OIT pass 1: build linked list =====
    // reset head index to 0xffffffff (upload, robust across GL versions)
    std::vector<GLuint> headClear(W * H, 0xffffffffu);
    glBindTexture(GL_TEXTURE_2D, headTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RED_INTEGER, GL_UNSIGNED_INT, headClear.data());
    // reset free counter
    GLuint zero = 0;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicBuf);
    glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &zero);

    glBindImageTexture(0, headTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, atomicBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, nodeBuf);

    glUseProgram(buildProg);
    glUniform1ui(uMaxN, MAX_NODES);
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);          // don't write depth, like the engine
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE); // no color output in build pass

    glBindVertexArray(cubeVao);
    for (int k = 0; k < nBoxes; k++) {
        int i = reverse ? (nBoxes - 1 - k) : k;
        glm::mat4 mv  = view * models[i];
        glm::mat4 mvp = proj * mv;
        glUniformMatrix4fv(uMVP,   1, GL_FALSE, glm::value_ptr(mvp));
        glUniformMatrix4fv(uModel, 1, GL_FALSE, glm::value_ptr(mv));
        glUniform3fv(uColor, 1, glm::value_ptr(colors[i]));
        glUniform1f(uAlpha, alpha);
        glUniform1i(uGeom, i);
        glDrawArrays(GL_TRIANGLES, 0, 36);
    }
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);

    // ===== make linked-list writes visible to the blend pass =====
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                    GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                    GL_TEXTURE_FETCH_BARRIER_BIT |
                    GL_ATOMIC_COUNTER_BARRIER_BIT);

    // report how many fragments were actually stored vs the node budget
    {
        GLuint used = 0;
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicBuf);
        glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &used);
        printf("fragments stored: %u / budget %u %s\n", used, MAX_NODES,
               used >= MAX_NODES ? "  <-- NODE BUFFER OVERFLOW (fragments dropped!)" : "");
    }

    // ===== OIT pass 2: resolve =====
    glUseProgram(blendProg);
    glUniform3fv(uBg, 1, glm::value_ptr(bg));
    glDisable(GL_DEPTH_TEST);
    glBindVertexArray(emptyVao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEnable(GL_DEPTH_TEST);

    // ---- read back & write PNG ----
    glFinish();
    std::vector<unsigned char> pix(W * H * 4);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, pix.data());
    // GL origin is bottom-left; flip vertically for PNG (top-left origin)
    std::vector<unsigned char> flipped(W * H * 4);
    for (int y = 0; y < H; y++)
        memcpy(&flipped[(size_t)(H - 1 - y) * W * 4], &pix[(size_t)y * W * 4], (size_t)W * 4);
    if (!stbi_write_png(out, W, H, 4, flipped.data(), W * 4)) {
        fprintf(stderr, "png write failed\n"); return 1; }
    printf("wrote %s  (%d boxes, yaw %.0f, pitch %.0f, %s)\n",
           out, nBoxes, yaw, pitch, fixed ? "FIXED" : "buggy");

    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(dpy, ctx);
    eglTerminate(dpy);
    return 0;
}
