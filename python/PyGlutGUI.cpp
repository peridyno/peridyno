#include "PyGlutGUI.h"

#include "GlutGUI/GLApp.h"

void pybind_glut_gui(py::module& m)
{
	py::class_<dyno::GLApp>(m, "GLApp")
		.def(py::init())
		.def("create_window", &dyno::GLApp::createWindow)
		.def("main_loop", &dyno::GLApp::mainLoop)
		.def("name", &dyno::GLApp::name)
		.def("get_width", &dyno::GLApp::getWidth)
		.def("get_height", &dyno::GLApp::getHeight)
		.def("save_screen", (bool (dyno::GLApp::*)()) &dyno::GLApp::saveScreen)
		.def("save_screen", (bool (dyno::GLApp::*)(const std::string &) const) &dyno::GLApp::saveScreen);
}
