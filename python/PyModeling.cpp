#include "PyModeling.h"
#include "PyFramework.h"
#include "PyCore.h"

#include "CubeModel.h"
template <typename TDataType>
void declare_cube_model(py::module& m, std::string typestr) {
	using Class = dyno::CubeModel<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("CubeModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//public
		.def("caption", &Class::caption)
		.def("boundingBox", &Class::boundingBox)
		//DEF_VAR
		.def("var_length", &Class::varLength, py::return_value_policy::reference)
		.def("var_segments", &Class::varSegments, py::return_value_policy::reference)
		//DEF_INSTANCE_STATE
		.def("state_polygon_set", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("state_quad_set", &Class::stateQuadSet, py::return_value_policy::reference)
		//DEF_VAR_OUT
		.def("out_cube", &Class::outCube, py::return_value_policy::reference);
}

void pybind_modeling(py::module& m) {
	declare_var<dyno::TOrientedBox3D<float>>(m, "TOrientedBox3D");
	declare_cube_model <dyno::DataType3f>(m, "3f");
}