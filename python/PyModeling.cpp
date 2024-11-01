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

	py::class_<dyno::Ramp, dyno::Canvas, std::shared_ptr<dyno::Ramp>>(m, "Ramp")
		.def(py::init<>())
		//.def("var_changed", &dyno::Ramp::varChanged)
		.def("get_curve_value_by_x", &dyno::Ramp::getCurveValueByX)
		.def("add_float_item_to_coord", &dyno::Ramp::addFloatItemToCoord)
		.def("add_item_original_coord", &dyno::Ramp::addItemOriginalCoord)
		.def("add_point", &dyno::Ramp::addPoint)
		.def("clear_my_coord", &dyno::Ramp::clearMyCoord)
		.def("add_item_handle_point", &dyno::Ramp::addItemHandlePoint)
		.def("update_field_final_coord", &dyno::Ramp::UpdateFieldFinalCoord)
		.def("update_bezier_point_to_bezier_set", &dyno::Ramp::updateBezierPointToBezierSet)
		.def("update_bezier_curve", &dyno::Ramp::updateBezierCurve)
		.def("calculate_length_for_point_set", &dyno::Ramp::calculateLengthForPointSet)
		.def("set_use_squard", &dyno::Ramp::setUseSquard)
		.def("use_bezier", &dyno::Ramp::useBezier)
		.def("use_linear", &dyno::Ramp::useLinear)
		.def("set_resample", &dyno::Ramp::setResample)
		.def("remap_x", &dyno::Ramp::remapX)
		.def("remap_y", &dyno::Ramp::remapY)
		.def("remap_xy", &dyno::Ramp::remapXY)
		.def("update_resample_linear_line", &dyno::Ramp::updateResampleLinearLine)
		.def("update_resample_bezier_curve", &dyno::Ramp::updateResampleBezierCurve)
		.def("resample_point_from_line", &dyno::Ramp::resamplePointFromLine)
		.def("set_range_min_x", &dyno::Ramp::setRange_MinX)
		.def("set_range_max_x", &dyno::Ramp::setRange_MaxX)
		.def("set_range_min_y", &dyno::Ramp::setRange_MinY)
		.def("set_range_max_y", &dyno::Ramp::setRange_MaxY)
		.def("set_range", &dyno::Ramp::setRange)
		.def("set_spacing", &dyno::Ramp::setSpacing)
		.def("border_close_resort", &dyno::Ramp::borderCloseResort)
		//.def("set_display_use_ramp", &dyno::Ramp::setDisplayUseRamp)
		//.def("set_use_ramp", &dyno::Ramp::setUseRamp)
		.def("convert_coord_to_str", &dyno::Ramp::convertCoordToStr);
	//.def("var_changed", &dyno::Ramp::convertVarToStr)
	//.def("var_changed", &dyno::Ramp::setVarByStr)

	py::class_<dyno::Curve, dyno::Canvas, std::shared_ptr<dyno::Curve>>(m, "Curve")
		.def(py::init<>())
		.def("add_point", &dyno::Curve::addPoint)
		.def("add_point_and_handle_point", &dyno::Curve::addPointAndHandlePoint)
		.def("set_curve_close", &dyno::Curve::setCurveClose)
		.def("get_points", &dyno::Curve::getPoints)
		.def("use_bezier", &dyno::Curve::useBezier)
		.def("use_linear", &dyno::Curve::useLinear)
		.def("set_resample", &dyno::Curve::setResample)
		.def("set_interp_mode", &dyno::Curve::setInterpMode)
		.def("set_use_squard", &dyno::Curve::setUseSquard)
		.def("remap_x", &dyno::Curve::remapX)
		.def("remap_y", &dyno::Curve::remapY)
		.def("remap_xy", &dyno::Curve::remapXY)
		.def("get_point_size", &dyno::Curve::getPointSize)
		.def("set_spacing", &dyno::Curve::setSpacing)
		.def("update_field_final_coord", &dyno::Curve::UpdateFieldFinalCoord)
		.def("add_float_item_to_coord", &dyno::Curve::addFloatItemToCoord)
		.def("add_item_original_coord", &dyno::Curve::addItemOriginalCoord)
		.def("clear_my_coord", &dyno::Curve::clearMyCoord)
		.def("add_item_handle_point", &dyno::Curve::addItemHandlePoint)
		.def("update_bezier_curve", &dyno::Curve::updateBezierCurve)
		.def("update_resample_linear_line", &dyno::Curve::updateResampleLinearLine)
		.def("update_resample_bezier_curve", &dyno::Curve::updateResampleBezierCurve)
		.def("resample_point_from_line", &dyno::Curve::resamplePointFromLine)
		.def("set_range_max_x", &dyno::Curve::setRange_MaxX)
		.def("set_range_min_y", &dyno::Curve::setRange_MinY)
		.def("set_range_min_x", &dyno::Curve::setRange_MinX)
		.def("set_range_max_y", &dyno::Curve::setRange_MaxY)
		.def("set_range", &dyno::Curve::setRange)
		.def("convert_coord_to_str", &dyno::Curve::convertCoordToStr);
	//.def("add_point", &dyno::Curve::setVarByStr)
	//.def("add_point", &dyno::Curve::convertVarToStr)
}