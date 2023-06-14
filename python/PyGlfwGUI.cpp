#include "PyGlfwGUI.h"

#include "SceneGraph.h"
#include "GlfwGUI/GlfwApp.h"

void pybind_glfw_gui(py::module& m)
{
	py::class_<dyno::GlfwApp>(m, "GLApp")
		.def(py::init())
		.def("set_scenegraph", &dyno::GlfwApp::setSceneGraph)
		.def("initialize", &dyno::GlfwApp::initialize)
		.def("main_loop", &dyno::GlfwApp::mainLoop);
}
