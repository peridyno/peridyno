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
	declare_normal_visualization<dyno::DataType3f>(m, "3f");
	declare_spline_constraint<dyno::DataType3f>(m, "3f");
	declare_subdivide<dyno::DataType3f>(m, "3f");
	declare_vector_visual_node<dyno::DataType3f>(m, "3f");

	py::class_<dyno::Canvas::Coord2D>(m, "CanvasCoord2D")
		.def(py::init<>())
		.def(py::init<double, double>())
		.def(py::init<double, double, double>())
		.def(py::init<dyno::Vector<float, 2>>())
		.def("set", &dyno::Canvas::Coord2D::set);

	py::class_<dyno::Canvas::EndPoint>(m, "CanvasEndPoint")
		.def(py::init<>())
		.def(py::init<int,int>());

	py::class_<dyno::Canvas::OriginalCoord>(m, "CanvasOriginalCoord")
		.def(py::init<int, int>())
		.def("set", &dyno::Canvas::OriginalCoord::set);


	py::class_<dyno::Canvas, std::shared_ptr<dyno::Canvas>>CANVAS(m, "Canvas");
	CANVAS.def(py::init<>())
		.def("addPoint", &dyno::Canvas::addPoint)
		.def("addPointAndHandlePoint", &dyno::Canvas::addPointAndHandlePoint)
		.def("addFloatItemToCoord", &dyno::Canvas::addFloatItemToCoord)
		.def("clearMyCoord", &dyno::Canvas::clearMyCoord)
		// Comands
		.def("setCurveClose", &dyno::Canvas::setCurveClose)
		.def("setInterpMode", &dyno::Canvas::setInterpMode)
		.def("setResample", &dyno::Canvas::setResample)
		.def("useBezier", &dyno::Canvas::useBezier)
		.def("useLinear", &dyno::Canvas::useLinear)
		.def("setSpacing", &dyno::Canvas::setSpacing)

		.def("updateBezierCurve", &dyno::Canvas::updateBezierCurve)
		.def("updateBezierPointToBezierSet", &dyno::Canvas::updateBezierPointToBezierSet)

		.def("updateResampleLinearLine", &dyno::Canvas::updateResampleLinearLine)
		.def("resamplePointFromLine", &dyno::Canvas::resamplePointFromLine)

		// get
		.def("getPoints", &dyno::Canvas::getPoints)
		.def("getPointSize", &dyno::Canvas::getPointSize)
		.def("UpdateFieldFinalCoord", &dyno::Canvas::UpdateFieldFinalCoord)

		.def("convertCoordToStr", &dyno::Canvas::convertCoordToStr)
		//.def("convert_var_to_str", &dyno::Canvas::convertVarToStr)
		.def("setVarByStr", py::overload_cast<std::string, double&>(&dyno::Canvas::setVarByStr))
		.def("setVarByStr", py::overload_cast<std::string, float&>(&dyno::Canvas::setVarByStr))
		.def("setVarByStr", py::overload_cast<std::string, int&>(&dyno::Canvas::setVarByStr))
		.def("setVarByStr", py::overload_cast<std::string, bool&>(&dyno::Canvas::setVarByStr))
		.def("setVarByStr", py::overload_cast<std::string, dyno::Canvas::Interpolation&>(&dyno::Canvas::setVarByStr));

	py::class_<dyno::Ramp, dyno::Canvas, std::shared_ptr<dyno::Ramp>>(m, "Ramp")
		.def(py::init<>())
		.def("getCurveValueByX", &dyno::Ramp::getCurveValueByX)
		.def("addFloatItemToCoord", &dyno::Ramp::addFloatItemToCoord)
		.def("addPoint", &dyno::Ramp::addPoint)
		.def("clearMyCoord", &dyno::Ramp::clearMyCoord)
		.def("UpdateFieldFinalCoord", &dyno::Ramp::UpdateFieldFinalCoord)
		.def("updateBezierPointToBezierSet", &dyno::Ramp::updateBezierPointToBezierSet)
		.def("updateBezierCurve", &dyno::Ramp::updateBezierCurve)
		.def("calculateLengthForPointSet", &dyno::Ramp::calculateLengthForPointSet)
		.def("useBezier", &dyno::Ramp::useBezier)
		.def("useLinear", &dyno::Ramp::useLinear)
		.def("setResample", &dyno::Ramp::setResample)

		.def("updateResampleLinearLine", &dyno::Ramp::updateResampleLinearLine)
		.def("updateResampleBezierCurve", &dyno::Ramp::updateResampleBezierCurve)
		.def("resamplePointFromLine", &dyno::Ramp::resamplePointFromLine)
		.def("setSpacing", &dyno::Ramp::setSpacing)
		.def("borderCloseResort", &dyno::Ramp::borderCloseResort)
		.def("UpdateFieldFinalCoord", &dyno::Ramp::UpdateFieldFinalCoord)
		.def_readwrite("myBezierPoint_H", &dyno::Ramp::myBezierPoint_H)
		.def_readwrite("FE_MyCoord", &dyno::Ramp::FE_MyCoord)
		.def_readwrite("FE_HandleCoord", &dyno::Ramp::FE_HandleCoord);

	py::class_<dyno::Curve, dyno::Canvas, std::shared_ptr<dyno::Curve>>(m, "Curve")
		.def(py::init<>())
		.def("addPoint", &dyno::Curve::addPoint)
		.def("addPointAndHandlePoint", &dyno::Curve::addPointAndHandlePoint)
		.def("setCurveClose", &dyno::Curve::setCurveClose)
		.def("getPoints", &dyno::Curve::getPoints)
		.def("useBezier", &dyno::Curve::useBezier)
		.def("useLinear", &dyno::Curve::useLinear)
		.def("setResample", &dyno::Curve::setResample)
		.def("setInterpMode", &dyno::Curve::setInterpMode)

		.def("getPointSize", &dyno::Curve::getPointSize)
		.def("setSpacing", &dyno::Curve::setSpacing)
		.def("UpdateFieldFinalCoord", &dyno::Curve::UpdateFieldFinalCoord)
		.def("addFloatItemToCoord", &dyno::Curve::addFloatItemToCoord)

		.def("clearMyCoord", &dyno::Curve::clearMyCoord)

		.def("updateBezierCurve", &dyno::Curve::updateBezierCurve)
		.def("updateResampleLinearLine", &dyno::Curve::updateResampleLinearLine)
		.def("updateResampleBezierCurve", &dyno::Curve::updateResampleBezierCurve)
		.def("resamplePointFromLine", &dyno::Curve::resamplePointFromLine)

		.def("convertCoordToStr", &dyno::Curve::convertCoordToStr);

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