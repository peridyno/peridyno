#pragma once
#include "../PyCommon.h"

#include "SceneGraph.h"
#include "GlfwGUI/GlfwApp.h"
#include "GlfwGUI/GlfwRenderWindow.h"
#include "GlfwGUI/imgui_impl_glfw.h"

void declare_gltf_app(py::module& m);

void declare_gltf_render_window(py::module& m);

void pybind_glfw_gui(py::module& m);