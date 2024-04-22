#pragma once
#include "PyCommon.h"

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

#include "SphereModel.h"
template <typename TDataType>
void declare_sphere_model(py::module& m, std::string typestr) {
	using Class = dyno::SphereModel<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("SphereModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>SM(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	SM.def(py::init<>())
		.def("caption", &Class::caption)
		.def("boundingBox", &Class::boundingBox)
		//DEF_VAR
		.def("var_center", &Class::varCenter, py::return_value_policy::reference)
		.def("var_radius", &Class::varRadius, py::return_value_policy::reference)
		.def("var_latitude", &Class::varLatitude, py::return_value_policy::reference)
		.def("var_longitude", &Class::varLongitude, py::return_value_policy::reference)
		//DEF_INSTANCE_STATE
		.def("state_polygon_set", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("state_triangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		//DEF_VAR_OUT
		.def("out_Sphere", &Class::outSphere, py::return_value_policy::reference)
		//DEF_ENUM
		.def("var_type", &Class::varType, py::return_value_policy::reference);

	py::enum_<typename Class::SphereType>(SM, "SphereType")
		.value("Standard", Class::SphereType::Standard)
		.value("Icosahedron", Class::SphereType::Icosahedron)
		.export_values();

}

#include "StaticTriangularMesh.h"
template <typename TDataType>
void declare_static_triangular_mesh(py::module& m, std::string typestr) {
	using Class = dyno::StaticTriangularMesh<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("StaticTriangularMesh") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEF_VAR
		.def("var_file_name", &Class::varFileName, py::return_value_policy::reference)
		//DEF_INSTANCE_STATE
		.def("state_initial_triangle_set", &Class::stateInitialTriangleSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference);
}

//------------------------- NEW END ------------------------------

#include "PlaneModel.h"
template <typename TDataType>
void declare_plane_model(py::module& m, std::string typestr) {
	using Class = dyno::PlaneModel<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("PlaneModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_scale", &Class::varScale, py::return_value_policy::reference)
		.def("state_triangleSet", &Class::stateTriangleSet, py::return_value_policy::reference);
}

void pybind_modeling(py::module& m);
