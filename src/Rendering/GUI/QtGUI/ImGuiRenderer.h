// The MIT License (MIT)

// Copyright (c) 2017 Ryohei Ikegami

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <imgui.h>
#include <memory>

#include <QOpenGLExtraFunctions>
#include <QObject>
#include <QPoint>
#include <imgui.h>
#include <memory>

class QMouseEvent;
class QWheelEvent;
class QKeyEvent;

namespace QtImGui {

class WindowWrapper {
public:
    virtual ~WindowWrapper() {}
    virtual void installEventFilter(QObject *object) = 0;
    virtual QSize size() const = 0;
    virtual qreal devicePixelRatio() const = 0;
    virtual bool isActive() const = 0;
    virtual QPoint mapFromGlobal(const QPoint &p) const = 0;
    virtual QObject* object() = 0;
    
    virtual void setCursorShape(Qt::CursorShape shape) = 0;
    virtual void setCursorPos(const QPoint& local_pos) = 0;
};

class ImGuiRenderer : public QObject, QOpenGLExtraFunctions {
    Q_OBJECT
public:
    void initialize(WindowWrapper *window);
    void newFrame();
    void render();
    bool eventFilter(QObject *watched, QEvent *event);

    static ImGuiRenderer *instance();

public:
    ImGuiRenderer();
    ~ImGuiRenderer();

private:
    void onMousePressedChange(QMouseEvent *event);
    void onWheel(QWheelEvent *event);
    void onKeyPressRelease(QKeyEvent *event);
    
    void updateCursorShape(const ImGuiIO &io);
    void setCursorPos(const ImGuiIO &io);

    void renderDrawList(ImDrawData *draw_data);
    bool createFontsTexture();
    bool createDeviceObjects();

    std::unique_ptr<WindowWrapper> m_window;
    double       g_Time = 0.0f;
    bool         g_MousePressed[3] = { false, false, false };
    float        g_MouseWheel;
    float        g_MouseWheelH;
    GLuint       g_FontTexture = 0;
    int          g_ShaderHandle = 0, g_VertHandle = 0, g_FragHandle = 0;
    int          g_AttribLocationTex = 0, g_AttribLocationProjMtx = 0;
    int          g_AttribLocationPosition = 0, g_AttribLocationUV = 0, g_AttribLocationColor = 0;
    unsigned int g_VboHandle = 0, g_VaoHandle = 0, g_ElementsHandle = 0;

    ImGuiContext* g_ctx = nullptr;
};

} // namespace QtImGui
