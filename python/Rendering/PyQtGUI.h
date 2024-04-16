#pragma once
#ifndef __PYQTGUI__
#define __PYQTGUI__
#include "../PyCommon.h"

#include "ImColorbar.h"
#include "ImWidget.h"


//template<typename T>
//void declare_instance(py::module& m, std::string typestr) {
//	using Class = dyno::FInstance<T>;
//	using Parent = InstanceBase;
//	std::string pyclass_name = std::string("Instance") + typestr;
//	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
//		.def(py::init<>())
//		.def("connect", &Class::connect)
//		.def("disconnect", &Class::disconnect);
//}

void declare_im_colorbar(py::module& m, std::string typestr);

void declare_im_widget(py::module& m, std::string typestr);

void pybind_qt_gui(py::module& m);

#endif