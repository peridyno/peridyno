#include "PyModeling.h"

void pybind_modeling(py::module& m)
{
	// Basic
	declare_model_editing<dyno::DataType3f>(m, "3f");
	declare_polygon_set_to_triangle_set_module<dyno::DataType3f>(m, "3f");
	declare_polygon_set_to_triangle_set_node<dyno::DataType3f>(m, "3f");
	declare_group<dyno::DataType3f>(m, "3f");

	// BasicShapes
	declare_basic_shape<dyno::DataType3f>(m, "3f");
	declare_capsule_model<dyno::DataType3f>(m, "3f");
	declare_cone_model<dyno::DataType3f>(m, "3f");
	declare_cube_model<dyno::DataType3f>(m, "3f");
	declare_cylinder_model<dyno::DataType3f>(m, "3f");
	declare_plane_model<dyno::DataType3f>(m, "3f");
	declare_sphere_model<dyno::DataType3f>(m, "3f");
	declare_tet_model<dyno::DataType3f>(m, "3f");

	// Commands
	declare_convert_to_texture_mesh<dyno::DataType3f>(m, "3f");
	declare_copy_model<dyno::DataType3f>(m, "3f");
	declare_copy_to_point<dyno::DataType3f>(m, "3f");
	declare_ear_clipper<dyno::DataType3f>(m, "3f");
	declare_editable_mesh<dyno::DataType3f>(m, "3f");
	declare_extract_shape<dyno::DataType3f>(m, "3f");
	declare_extrude_model<dyno::DataType3f>(m, "3f");
	declare_merge<dyno::DataType3f>(m, "3f");
	declare_point_clip<dyno::DataType3f>(m, "3f");
	declare_poly_extrude<dyno::DataType3f>(m, "3f");

	declare_sweep_model<dyno::DataType3f>(m, "3f");
	declare_texture_mesh_merge<dyno::DataType3f>(m, "3f");
	declare_transform_model<dyno::DataType3f>(m, "3f");
	declare_turning_model<dyno::DataType3f>(m, "3f");
	declare_triangle_set_to_triangle_sets<dyno::DataType3f>(m, "3f");
	declare_extract_triangle_sets<dyno::DataType3f>(m, "3f");

	// Samples
	declare_point_from_curve<dyno::DataType3f>(m, "3f");
	declare_sampler<dyno::DataType3f>(m, "3f");
	declare_points_behind_mesh<dyno::DataType3f>(m, "3f");
	declare_poisson_plane<dyno::DataType3f>(m, "3f");
	declare_shape_sampler<dyno::DataType3f>(m, "3f");

	// Moodeling
	declare_collision_detector<dyno::DataType3f>(m, "3f");
	declare_bounding_box_of_texture_mesh(m);
	declare_gltf_loader<dyno::DataType3f>(m, "3f");
	declare_joint_deform<dyno::DataType3f>(m, "3f");
	declare_normal_visualization<dyno::DataType3f>(m, "3f");
	declare_spline_constraint<dyno::DataType3f>(m, "3f");
	declare_subdivide<dyno::DataType3f>(m, "3f");
	declare_vector_visual_node<dyno::DataType3f>(m, "3f");

	py::enum_<dyno::BasicShapeType>(m, "BasicShapeType", py::arithmetic())
		.value("PLANE", dyno::BasicShapeType::PLANE)
		.value("CUBE", dyno::BasicShapeType::CUBE)
		.value("SPHERE", dyno::BasicShapeType::SPHERE)
		.value("CONE", dyno::BasicShapeType::CONE)
		.value("CAPSULE", dyno::BasicShapeType::CAPSULE)
		.value("CYLINDER", dyno::BasicShapeType::CYLINDER)
		.value("UNKNOWN", dyno::BasicShapeType::UNKNOWN)
		.export_values();
}