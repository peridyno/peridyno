#include "PyModeling.h"

#include "initializeModeling.h"
void declare_modeling_initializer(py::module& m) {
	using Class = dyno::ModelingInitializer;
	using Parent = dyno::PluginEntry;
	std::string pyclass_name = std::string("ModelingInitializer");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("instance", &Class::instance);
}

void pybind_modeling(py::module& m) {
	//declare_var<dyno::TOrientedBox3D<Real>>(m, "TOrientedBox3D");
	declare_capsule_model<dyno::DataType3f>(m, "3f");
	declare_cone_model<dyno::DataType3f>(m, "3f");
	declare_copy_model<dyno::DataType3f>(m, "3f");
	declare_copy_to_point<dyno::DataType3f>(m, "3f");
	declare_cube_model<dyno::DataType3f>(m, "3f");
	declare_cylinder_model<dyno::DataType3f>(m, "3f");
	declare_ear_clipper<dyno::DataType3f>(m, "3f");
	declare_extrude_model<dyno::DataType3f>(m, "3f");
	declare_gltf_loader<dyno::DataType3f>(m, "3f");
	declare_group<dyno::DataType3f>(m, "3f");
	declare_merge<dyno::DataType3f>(m, "3f");
	declare_normal<dyno::DataType3f>(m, "3f");
	declare_plane_model<dyno::DataType3f>(m, "3f");
	declare_point_clip<dyno::DataType3f>(m, "3f");
	declare_point_from_curve<dyno::DataType3f>(m, "3f");
	declare_poly_extrude<dyno::DataType3f>(m, "3f");
	declare_sphere_model<dyno::DataType3f>(m, "3f");
	declare_spline_constraint<dyno::DataType3f>(m, "3f");
	declare_static_triangular_mesh<dyno::DataType3f>(m, "3f");
	declare_sweep_model<dyno::DataType3f>(m, "3f");
	declare_transform_model<dyno::DataType3f>(m, "3f");
	declare_turning_model<dyno::DataType3f>(m, "3f");
	declare_vector_visual_node<dyno::DataType3f>(m, "3f");

	declare_modeling_initializer(m);
}