#include "PyGlfwGUI.h"

#include "SceneGraph.h"
#include "GlfwGUI/GlfwApp.h"

void pybind_glfw_gui(py::module& m)
{
	py::class_<dyno::GlfwApp>(m, "GLApp")
		.def(py::init())
		.def("set_scenegraph", &dyno::GlfwApp::setSceneGraph)
		.def("create_window", &dyno::GlfwApp::createWindow)
		.def("main_loop", &dyno::GlfwApp::mainLoop)
		.def("name", &dyno::GlfwApp::name)
		.def("get_width", &dyno::GlfwApp::getWidth)
		.def("get_height", &dyno::GlfwApp::getHeight)
		.def("save_screen", (bool (dyno::GlfwApp::*)()) &dyno::GlfwApp::saveScreen)
		.def("save_screen", (bool (dyno::GlfwApp::*)(const std::string &) const) &dyno::GlfwApp::saveScreen);
}
