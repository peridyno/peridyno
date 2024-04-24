#pragma once
#include "PyCommon.h"

#include "Mapping/HeightFieldToTriangleSet.h"
template <typename TDataType>
void declare__height_field_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::HeightFieldToTriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("HeightFieldToTriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_scale", &Class::varScale, py::return_value_policy::reference)
		.def("var_translation", &Class::varTranslation, py::return_value_policy::reference)
		.def("in_height_field", &Class::inHeightField, py::return_value_policy::reference)
		.def("out_triangle_set", &Class::outTriangleSet, py::return_value_policy::reference);
}

void pybind_topology(py::module& m);