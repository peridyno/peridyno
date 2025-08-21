#include "PyQt.h"

//#include "AppBase.h"

void declare_qt_app(py::module& m)
{
	using Class = dyno::QtApp;
	using Parent = dyno::AppBase;
	std::string pyclass_name = std::string("QtApp");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init())
		.def("setSceneGraph", &Class::setSceneGraph)
		.def("initialize", &Class::initialize)
		.def("mainLoop", &Class::mainLoop)
		.def("renderWindow", &Class::renderWindow)
		.def("setWindowTitle", &Class::setWindowTitle);
}

void pybind_qt_gui(py::module& m)
{
	declare_qt_app(m);
}