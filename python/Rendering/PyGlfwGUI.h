#pragma once
#include "../PyCommon.h"

#include "SceneGraph.h"
#include "GlfwGUI/GlfwApp.h"
#include "GlfwGUI/GlfwRenderWindow.h"
#include "GlfwGUI/imgui_impl_glfw.h"

#include "RenderWindow.h"
#include "Camera.h"
void declare_rednder_window(py::module& m) {
	using Class = dyno::RenderWindow;
	std::string pyclass_name = std::string("RenderWindow");
	py::class_<Class, std::shared_ptr<Class>>RW(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	RW.def(py::init<>())
		.def("initialize", &dyno::RenderWindow::initialize)
		.def("main_loop", &dyno::RenderWindow::mainLoop)
		.def("get_render_engine", &dyno::RenderWindow::getRenderEngine)
		.def("set_render_engine", &dyno::RenderWindow::setRenderEngine)
		.def("get_camera", &dyno::RenderWindow::getCamera)
		.def("set_camera", &dyno::RenderWindow::setCamera)
		.def("get_render_params", &dyno::RenderWindow::getRenderParams)
		.def("set_render_params", &dyno::RenderWindow::setRenderParams)
		.def("set_window_size", &dyno::RenderWindow::setWindowSize)
		.def("get_selection_mode", &dyno::RenderWindow::getSelectionMode)
		.def("set_selection_mode", &dyno::RenderWindow::setSelectionMode)
		.def("toggle_im_gui", &dyno::RenderWindow::toggleImGUI)
		.def("show_im_gui", &dyno::RenderWindow::showImGUI)
		.def("is_screen_recording_on", &dyno::RenderWindow::isScreenRecordingOn)
		.def("screen_recording_interval", &dyno::RenderWindow::screenRecordingInterval)
		.def("set_screen_recording_path", &dyno::RenderWindow::setScreenRecordingPath)
		.def("save_screen", &dyno::RenderWindow::saveScreen)
		.def("set_main_light_direction", &dyno::RenderWindow::setMainLightDirection)

		.def("select", py::overload_cast<int, int, int, int>(&dyno::RenderWindow::select))
		.def("select", py::overload_cast<std::shared_ptr<dyno::Node>, int, int>(&dyno::RenderWindow::select))
		.def("get_current_selected_node", &dyno::RenderWindow::getCurrentSelectedNode);

	py::enum_<Class::SelectionMode>(RW, "SelectionMode")
		.value("OBJECT_MODE", Class::SelectionMode::OBJECT_MODE)
		.value("PRIMITIVE_MODE", Class::SelectionMode::PRIMITIVE_MODE);
}


void declare_gltf_app(py::module& m);

void declare_gltf_render_window(py::module& m);

void pybind_glfw_gui(py::module& m);