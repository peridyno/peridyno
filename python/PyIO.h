#pragma once
#include "PyCommon.h"

#include "EigenValueWriter.h"
template <typename TDataType>
void declare_EigenValueWriter(py::module& m, std::string typestr) {
	using Class = dyno::EigenValueWriter<TDataType>;
	using Parent = dyno::OutputModule;
	std::string pyclass_name = std::string("EigenValueWriter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("boundingBox", &Class::boundingBox)
		.def("var_center", &Class::varCenter, py::return_value_policy::reference)
		.def("var_radius", &Class::varRadius, py::return_value_policy::reference)
		.def("var_latitude", &Class::varLatitude, py::return_value_policy::reference)
		.def("var_longitude", &Class::varLongitude, py::return_value_policy::reference)
		.def("var_height", &Class::varHeight, py::return_value_policy::reference)
		.def("var_height_segment", &Class::varHeightSegment, py::return_value_policy::reference)
		.def("state_polygon_set", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("out_capsule", &Class::outCapsule, py::return_value_policy::reference);
}

void declare_gmsh(py::module& m);

void declare_smesh(py::module& m);

void declare_io_initializer(py::module& m);

void pybind_io(py::module& m);