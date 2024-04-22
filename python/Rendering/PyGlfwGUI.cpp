#include "PyGlfwGUI.h"

#include "SceneGraph.h"
#include "GlfwGUI/GlfwApp.h"

void pybind_glfw_gui(py::module& m)
{
	py::class_<dyno::GlfwApp>(m, "GLfwApp")
		.def(py::init())
		.def("set_scenegraph", &dyno::GlfwApp::setSceneGraph)
		.def("initialize", &dyno::GlfwApp::initialize)
		.def("main_loop", &dyno::GlfwApp::mainLoop)
		.def("render_window", &dyno::GlfwApp::renderWindow)
		.def("set_window_title", &dyno::GlfwApp::setWindowTitle);
}