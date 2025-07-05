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
		.def("mainLoop", &dyno::RenderWindow::mainLoop)
		.def("getRenderEngine", &dyno::RenderWindow::getRenderEngine)
		.def("setRenderEngine", &dyno::RenderWindow::setRenderEngine)
		.def("getCamera", &dyno::RenderWindow::getCamera)
		.def("setCamera", &dyno::RenderWindow::setCamera)
		.def("getRenderParams", &dyno::RenderWindow::getRenderParams)
		.def("setRenderParams", &dyno::RenderWindow::setRenderParams)
		.def("setWindowSize", &dyno::RenderWindow::setWindowSize)
		.def("getSelectionMode", &dyno::RenderWindow::getSelectionMode)
		.def("setSelectionMode", &dyno::RenderWindow::setSelectionMode)
		.def("toggleImGUI", &dyno::RenderWindow::toggleImGUI)
		.def("showImGUI", &dyno::RenderWindow::showImGUI)
		.def("isScreenRecordingOn", &dyno::RenderWindow::isScreenRecordingOn)
		.def("screenRecordingInterval", &dyno::RenderWindow::screenRecordingInterval)
		.def("setScreenRecordingPath", &dyno::RenderWindow::setScreenRecordingPath)
		.def("saveScreen", &dyno::RenderWindow::saveScreen)
		.def("setMainLightDirection", &dyno::RenderWindow::setMainLightDirection)

		.def("select", py::overload_cast<int, int, int, int>(&dyno::RenderWindow::select))
		.def("select", py::overload_cast<std::shared_ptr<dyno::Node>, int, int>(&dyno::RenderWindow::select))
		.def("getCurrentSelectedNode", &dyno::RenderWindow::getCurrentSelectedNode);

	py::enum_<Class::SelectionMode>(RW, "SelectionMode")
		.value("OBJECT_MODE", Class::SelectionMode::OBJECT_MODE)
		.value("PRIMITIVE_MODE", Class::SelectionMode::PRIMITIVE_MODE);
}

void declare_gltf_app(py::module& m);

void declare_gltf_render_window(py::module& m);

void pybind_glfw_gui(py::module& m);