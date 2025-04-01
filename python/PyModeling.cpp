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
	declare_joint_info(m);
	declare_joint_animation_info(m);
	declare_normal_visualization<dyno::DataType3f>(m, "3f");
	declare_spline_constraint<dyno::DataType3f>(m, "3f");
	declare_subdivide<dyno::DataType3f>(m, "3f");
	declare_vector_visual_node<dyno::DataType3f>(m, "3f");

	py::class_<dyno::Canvas, std::shared_ptr<dyno::Canvas>>CANVAS(m, "Canvas");
	CANVAS.def(py::init<>())
		.def("add_point", &dyno::Canvas::addPoint)
		.def("add_point_and_handle_point", &dyno::Canvas::addPointAndHandlePoint)
		.def("add_float_item_to_coord", &dyno::Canvas::addFloatItemToCoord)
		.def("clear_my_coord", &dyno::Canvas::clearMyCoord)
		// Comands
		.def("set_curve_close", &dyno::Canvas::setCurveClose)
		.def("get_interp_mode", &dyno::Canvas::setInterpMode)
		.def("set_resample", &dyno::Canvas::setResample)
		.def("set_use_squard", &dyno::Canvas::setUseSquard)
		.def("use_bezier", &dyno::Canvas::useBezier)
		.def("use_linear", &dyno::Canvas::useLinear)
		.def("set_spacing", &dyno::Canvas::setSpacing)
		//
		.def("remap_x", &dyno::Canvas::remapX)
		.def("remap_y", &dyno::Canvas::remapY)
		.def("remap_xy", &dyno::Canvas::remapXY)

		.def("set_range_max_x", &dyno::Canvas::setRange_MaxX)
		.def("set_range_min_y", &dyno::Canvas::setRange_MinY)
		.def("set_range_min_x", &dyno::Canvas::setRange_MinX)
		.def("set_range_max_y", &dyno::Canvas::setRange_MaxY)
		.def("set_range", &dyno::Canvas::setRange)

		.def("add_item_original_coord", &dyno::Canvas::addItemOriginalCoord)
		.def("add_item_handle_point", &dyno::Canvas::addItemHandlePoint)
		.def("update_bezier_curve", &dyno::Canvas::updateBezierCurve)
		.def("update_bezier_point_to_bezier_set", &dyno::Canvas::updateBezierPointToBezierSet)

		.def("update_resample_linear_line", &dyno::Canvas::updateResampleLinearLine)
		.def("resample_point_from_line", &dyno::Canvas::resamplePointFromLine)

		// get
		.def("get_points", &dyno::Canvas::getPoints)
		.def("get_points_size", &dyno::Canvas::getPointSize)
		.def("update_field_final_coord", &dyno::Canvas::UpdateFieldFinalCoord)

		.def("convert_coord_to_str", &dyno::Canvas::convertCoordToStr)
		//.def("convert_var_to_str", &dyno::Canvas::convertVarToStr)
		.def("set_var_by_str", py::overload_cast<std::string, double&>(&dyno::Canvas::setVarByStr))
		.def("set_var_by_str", py::overload_cast<std::string, float&>(&dyno::Canvas::setVarByStr))
		.def("set_var_by_str", py::overload_cast<std::string, int&>(&dyno::Canvas::setVarByStr))
		.def("set_var_by_str", py::overload_cast<std::string, bool&>(&dyno::Canvas::setVarByStr))
		.def("set_var_by_str", py::overload_cast<std::string, dyno::Canvas::Interpolation&>(&dyno::Canvas::setVarByStr))
		.def("set_var_by_str", py::overload_cast<std::string, dyno::Canvas::Direction&>(&dyno::Canvas::setVarByStr))

		.def_readwrite("mInterpMode", &dyno::Canvas::mInterpMode)
		.def_readwrite("mCoord", &dyno::Canvas::mCoord)
		.def_readwrite("mBezierPoint", &dyno::Canvas::mBezierPoint)
		.def_readwrite("mFinalCoord", &dyno::Canvas::mFinalCoord)
		.def_readwrite("mResamplePoint", &dyno::Canvas::mResamplePoint)
		.def_readwrite("mLengthArray", &dyno::Canvas::mLengthArray)
		.def_readwrite("myHandlePoint", &dyno::Canvas::myHandlePoint)
		//.def_readwrite("InterpStrings", &dyno::Canvas::InterpStrings)
		.def_readwrite("Originalcoord", &dyno::Canvas::Originalcoord)
		.def_readwrite("OriginalHandlePoint", &dyno::Canvas::OriginalHandlePoint)

		//.def_readwrite("remapRange", &dyno::Canvas::remapRange)

		.def_readwrite("NminX", &dyno::Canvas::NminX)
		.def_readwrite("NmaxX", &dyno::Canvas::NmaxX)
		.def_readwrite("mNewMinY", &dyno::Canvas::mNewMinY)
		.def_readwrite("NmaxY", &dyno::Canvas::NmaxY)

		.def_readwrite("lockSize", &dyno::Canvas::lockSize)
		.def_readwrite("useBezierInterpolation", &dyno::Canvas::useBezierInterpolation)
		.def_readwrite("resample", &dyno::Canvas::resample)
		.def_readwrite("curveClose", &dyno::Canvas::curveClose)
		.def_readwrite("useColseButton", &dyno::Canvas::useColseButton)
		.def_readwrite("useSquard", &dyno::Canvas::useSquard)
		.def_readwrite("useSquardButton", &dyno::Canvas::useSquardButton)
		.def_readwrite("Spacing", &dyno::Canvas::Spacing)
		.def_readwrite("segment", &dyno::Canvas::segment)
		.def_readwrite("resampleResolution", &dyno::Canvas::resampleResolution);

	py::class_<dyno::Ramp, dyno::Canvas, std::shared_ptr<dyno::Ramp>>(m, "Ramp")
		.def(py::init<>())
		.def("get_curve_value_by_x", &dyno::Ramp::getCurveValueByX)
		.def("add_float_item_to_coord", &dyno::Ramp::addFloatItemToCoord)
		.def("add_point", &dyno::Ramp::addPoint)
		.def("clear_my_coord", &dyno::Ramp::clearMyCoord)
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
		.def("update_field_final_coord", &dyno::Ramp::UpdateFieldFinalCoord)
		.def_readwrite("Dirmode", &dyno::Ramp::Dirmode)
		//.def_readwrite("DirectionStrings", &dyno::Ramp::DirectionStrings)
		.def_readwrite("myBezierPoint_H", &dyno::Ramp::myBezierPoint_H)
		.def_readwrite("FE_MyCoord", &dyno::Ramp::FE_MyCoord)
		.def_readwrite("FE_HandleCoord", &dyno::Ramp::FE_HandleCoord);

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