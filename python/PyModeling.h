#pragma once
#include "PyCommon.h"

#include "BasicShapes/BasicShape.h"
template <typename TDataType>
void declare_basic_shape(py::module& m, std::string typestr) {
	using Class = dyno::BasicShape<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("BasicShape") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>BS(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	BS.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)
		.def("get_shape_type", &Class::getShapeType);
}

#include "BasicShapes/CapsuleModel.h"
template <typename TDataType>
void declare_capsule_model(py::module& m, std::string typestr) {
	using Class = dyno::CapsuleModel<TDataType>;
	using Parent = dyno::BasicShape<TDataType>;
	std::string pyclass_name = std::string("CapsuleModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("get_shape_type", &Class::getShapeType)
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

#include "BasicShapes/ConeModel.h"
template <typename TDataType>
void declare_cone_model(py::module& m, std::string typestr) {
	using Class = dyno::ConeModel<TDataType>;
	using Parent = dyno::BasicShape<TDataType>;
	std::string pyclass_name = std::string("ConeModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("get_shape_type", &Class::getShapeType)
		.def("var_columns", &Class::varColumns, py::return_value_policy::reference)
		.def("var_row", &Class::varRow, py::return_value_policy::reference)
		.def("var_end_segment", &Class::varEndSegment, py::return_value_policy::reference)
		.def("var_radius", &Class::varRadius, py::return_value_policy::reference)
		.def("var_height", &Class::varHeight, py::return_value_policy::reference)
		.def("state_polygon_set", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("out_cone", &Class::outCone, py::return_value_policy::reference)

		.def("boundingBox", &Class::boundingBox);
}

#include "BasicShapes/CubeModel.h"
template <typename TDataType>
void declare_cube_model(py::module& m, std::string typestr) {
	using Class = dyno::CubeModel<TDataType>;
	using Parent = dyno::BasicShape<TDataType>;
	std::string pyclass_name = std::string("CubeModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("get_shape_type", &Class::getShapeType)
		.def("boundingBox", &Class::boundingBox)
		.def("var_length", &Class::varLength, py::return_value_policy::reference)
		.def("var_segments", &Class::varSegments, py::return_value_policy::reference)
		.def("state_polygon_set", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("state_quad_set", &Class::stateQuadSet, py::return_value_policy::reference)
		.def("out_cube", &Class::outCube, py::return_value_policy::reference);
}

#include "BasicShapes/CylinderModel.h"
template <typename TDataType>
void declare_cylinder_model(py::module& m, std::string typestr) {
	using Class = dyno::CylinderModel<TDataType>;
	using Parent = dyno::BasicShape<TDataType>;
	std::string pyclass_name = std::string("CylinderModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("get_shape_type", &Class::getShapeType)

		.def("var_columns", &Class::varColumns, py::return_value_policy::reference)
		.def("var_row", &Class::varRow, py::return_value_policy::reference)
		.def("var_end_segment", &Class::varEndSegment, py::return_value_policy::reference)
		.def("var_radius", &Class::varRadius, py::return_value_policy::reference)
		.def("var_height", &Class::varHeight, py::return_value_policy::reference)
		.def("state_polygon_set", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("out_cylinder", &Class::outCylinder, py::return_value_policy::reference)

		.def("boundingBox", &Class::boundingBox);
}

#include "BasicShapes/PlaneModel.h"
template <typename TDataType>
void declare_plane_model(py::module& m, std::string typestr) {
	using Class = dyno::PlaneModel<TDataType>;
	using Parent = dyno::BasicShape<TDataType>;
	std::string pyclass_name = std::string("PlaneModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("get_shape_type", &Class::getShapeType)
		.def("bounding_box", &Class::boundingBox)

		.def("var_length_x", &Class::varLengthX, py::return_value_policy::reference)
		.def("var_length_z", &Class::varLengthZ, py::return_value_policy::reference)
		.def("var_segment_x", &Class::varSegmentX, py::return_value_policy::reference)
		.def("var_segment_z", &Class::varSegmentZ, py::return_value_policy::reference)
		.def("state_polygon_set", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("state_quad_set", &Class::stateQuadSet, py::return_value_policy::reference);
}

#include "BasicShapes/SphereModel.h"
template <typename TDataType>
void declare_sphere_model(py::module& m, std::string typestr) {
	using Class = dyno::SphereModel<TDataType>;
	using Parent = dyno::BasicShape<TDataType>;
	std::string pyclass_name = std::string("SphereModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>SM(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	SM.def(py::init<>())
		.def("caption", &Class::caption)
		.def("get_shape_type", &Class::getShapeType)
		.def("boundingBox", &Class::boundingBox)

		.def("var_center", &Class::varCenter, py::return_value_policy::reference)
		.def("var_radius", &Class::varRadius, py::return_value_policy::reference)
		.def("var_latitude", &Class::varLatitude, py::return_value_policy::reference)
		.def("var_longitude", &Class::varLongitude, py::return_value_policy::reference)
		.def("var_icosahedron_step", &Class::varIcosahedronStep, py::return_value_policy::reference)
		.def("state_polygon_set", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)

		.def("out_Sphere", &Class::outSphere, py::return_value_policy::reference)
		.def("var_type", &Class::varType, py::return_value_policy::reference);

	py::enum_<typename Class::SphereType>(SM, "SphereType")
		.value("Standard", Class::SphereType::Standard)
		.value("Icosahedron", Class::SphereType::Icosahedron)
		.export_values();
}

#include "Commands/ConvertToTextureMesh.h"
template <typename TDataType>
void declare_convert_to_texture_mesh(py::module& m, std::string typestr) {
	using Class = dyno::ConvertToTextureMesh<TDataType>;
	using Parent = dyno::ModelEditing<TDataType>;
	std::string pyclass_name = std::string("ConvertToTextureMesh") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_diffuse_texture", &Class::varDiffuseTexture, py::return_value_policy::reference)
		.def("var_normal_texture", &Class::varNormalTexture, py::return_value_policy::reference)
		.def("var_uv_scale_U", &Class::varUvScaleU, py::return_value_policy::reference)
		.def("var_uv_scale_v", &Class::varUvScaleV, py::return_value_policy::reference)
		.def("var_use_bounding_transform", &Class::varUseBoundingTransform, py::return_value_policy::reference)
		.def("in_topolopy", &Class::inTopology, py::return_value_policy::reference)
		.def("state_texture_mesh", &Class::stateTextureMesh, py::return_value_policy::reference);
}

#include "Commands/CopyModel.h"
template <typename TDataType>
void declare_copy_model(py::module& m, std::string typestr) {
	using Class = dyno::CopyModel<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("CopyModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>CM(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	CM.def(py::init<>())
		.def("var_total_number", &Class::varTotalNumber, py::return_value_policy::reference)
		.def("var_copy_transform", &Class::varCopyTransform, py::return_value_policy::reference)
		.def("var_copy_rotation", &Class::varCopyRotation, py::return_value_policy::reference)
		.def("var_copy_scale", &Class::varCopyScale, py::return_value_policy::reference)
		.def("var_scale_mode", &Class::varScaleMode, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("in_triangle_set_in", &Class::inTriangleSetIn, py::return_value_policy::reference);

	py::enum_<typename Class::ScaleMode>(CM, "ScaleMode")
		.value("Power", Class::ScaleMode::Power)
		.value("Multiply", Class::ScaleMode::Multiply)
		.export_values();
}

#include "Commands/CopyToPoint.h"
template <typename TDataType>
void declare_copy_to_point(py::module& m, std::string typestr) {
	using Class = dyno::CopyToPoint<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("CopyToPoint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>CTP(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	CTP.def(py::init<>())
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("in_triangle_set_in", &Class::inTriangleSetIn, py::return_value_policy::reference)
		.def("in_target_point_set", &Class::inTargetPointSet, py::return_value_policy::reference)
		.def("disable_render", &Class::disableRender);

	py::enum_<typename Class::ScaleMode>(CTP, "ScaleMode")
		.value("Power", Class::ScaleMode::Power)
		.value("Multiply", Class::ScaleMode::Multiply)
		.export_values();
}

#include "Commands/EarClipper.h"
template <typename TDataType>
void declare_ear_clipper(py::module& m, std::string typestr) {
	using Class = dyno::EarClipper<TDataType>;
	using Parent = dyno::ModelEditing<TDataType>;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	std::string pyclass_name = std::string("EarClipper") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<std::vector<Coord>, std::vector<dyno::TopologyModule::Triangle>&>())
		.def("var_changed", &Class::varChanged)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("in_point_set", &Class::inPointSet, py::return_value_policy::reference)
		.def("var_reverse_normal", &Class::varReverseNormal, py::return_value_policy::reference)

		.def("poly_clip", py::overload_cast<std::vector<Coord>, std::vector<dyno::TopologyModule::Triangle>&>(&Class::polyClip))
		.def("poly_clip", py::overload_cast<dyno::Array<Coord, DeviceType::GPU>, std::vector<dyno::TopologyModule::Triangle>&>(&Class::polyClip))
		.def("poly_clip", py::overload_cast<dyno::Array<Coord, DeviceType::CPU>, std::vector<dyno::TopologyModule::Triangle>&>(&Class::polyClip));
}

#include "Commands/EditableMesh.h"
template <typename TDataType>
void declare_editable_mesh(py::module& m, std::string typestr) {
	using Class = dyno::EditableMesh<TDataType>;
	using Parent = dyno::PolygonSetToTriangleSetNode<TDataType>;
	std::string pyclass_name = std::string("EditableMesh") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("in_polygon_set", &Class::inPolygonSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("state_vertex_normal", &Class::stateVertexNormal, py::return_value_policy::reference)
		.def("state_triangle_normal", &Class::stateTriangleNormal, py::return_value_policy::reference)
		.def("state_polygon_normal", &Class::statePolygonNormal, py::return_value_policy::reference);
}

#include "Commands/ExtractShape.h"
template <typename TDataType>
void declare_extract_shape(py::module& m, std::string typestr) {
	using Class = dyno::ExtractShape<TDataType>;
	using Parent = dyno::ModelEditing<TDataType>;
	std::string pyclass_name = std::string("ExtractShape") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_shape_id", &Class::varShapeId, py::return_value_policy::reference)
		.def("var_shape_transform", &Class::varShapeTransform, py::return_value_policy::reference)
		.def("var_offset", &Class::varOffset, py::return_value_policy::reference)
		.def("in_in_texture_mesh", &Class::inInTextureMesh, py::return_value_policy::reference)
		.def("state_result", &Class::stateResult, py::return_value_policy::reference);
}

#include "Commands/Extrude.h"
template <typename TDataType>
void declare_extrude_model(py::module& m, std::string typestr) {
	using Class = dyno::ExtrudeModel<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("ExtrudeModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_row", &Class::varRow, py::return_value_policy::reference)
		.def("var_height", &Class::varHeight, py::return_value_policy::reference)
		.def("var_reverse_normal", &Class::varReverseNormal, py::return_value_policy::reference)
		.def("var_curve", &Class::varCurve, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("in_point_set", &Class::inPointSet, py::return_value_policy::reference);
}

#include "Commands/Merge.h"
template <typename TDataType>
void declare_merge(py::module& m, std::string typestr) {
	using Class = dyno::Merge<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("Merge") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>M(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	M.def(py::init<>())
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("in_triangle_set_01", &Class::inTriangleSet01, py::return_value_policy::reference)
		.def("in_triangle_set_02", &Class::inTriangleSet02, py::return_value_policy::reference)
		.def("in_triangle_set_03", &Class::inTriangleSet03, py::return_value_policy::reference)
		.def("in_triangle_set_04", &Class::inTriangleSet04, py::return_value_policy::reference)
		.def("var_update_mode", &Class::varUpdateMode, py::return_value_policy::reference)
		.def("pre_update_states", &Class::preUpdateStates)
		.def("merge_gpu", &Class::MergeGPU);

	py::enum_<typename Class::UpdateMode>(M, "UpdateMode")
		.value("Reset", Class::UpdateMode::Reset)
		.value("Tick", Class::UpdateMode::Tick)
		.export_values();
}

#include "Commands/PointClip.h"
template <typename TDataType>
void declare_point_clip(py::module& m, std::string typestr) {
	using Class = dyno::PointClip<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("PointClip") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_point_set", &Class::inPointSet, py::return_value_policy::reference)
		.def("var_plane_size", &Class::varPlaneSize, py::return_value_policy::reference)
		.def("var_reverse", &Class::varReverse, py::return_value_policy::reference)
		.def("var_point_size", &Class::varPointSize, py::return_value_policy::reference)
		.def("var_point_color", &Class::varPointColor, py::return_value_policy::reference)
		.def("var_show_plane", &Class::varShowPlane, py::return_value_policy::reference)
		.def("state_clip_plane", &Class::stateClipPlane, py::return_value_policy::reference)
		.def("state_point_set", &Class::statePointSet, py::return_value_policy::reference);
}

#include "Commands/PolyExtrude.h"
template <typename TDataType>
void declare_poly_extrude(py::module& m, std::string typestr) {
	using Class = dyno::PolyExtrude<TDataType>;
	using Parent = dyno::Group<TDataType>;
	std::string pyclass_name = std::string("PolyExtrude") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_divisions", &Class::varDivisions, py::return_value_policy::reference)
		.def("var_distance", &Class::varDistance, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("state_normal_set", &Class::stateNormalSet, py::return_value_policy::reference)
		.def("var_reverse_normal", &Class::varReverseNormal, py::return_value_policy::reference);
}

#include "Commands/PolygonSetToTriangleSet.h"
template <typename TDataType>
void declare_polygon_set_to_triangle_set_module(py::module& m, std::string typestr) {
	using Class = dyno::PolygonSetToTriangleSetModule<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("PolygonSetToTriangleSetModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("in_polygon_set", &Class::inPolygonSet, py::return_value_policy::reference)
		.def("out_triangle_set", &Class::outTriangleSet, py::return_value_policy::reference)
		.def("out_polygon2_triangles", &Class::outPolygon2Triangles, py::return_value_policy::reference)

		.def("convert", &Class::convert);
}

template <typename TDataType>
void declare_polygon_set_to_triangle_set_node(py::module& m, std::string typestr) {
	using Class = dyno::PolygonSetToTriangleSetNode<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("PolygonSetToTriangleSetNode") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("in_polygon_set", &Class::inPolygonSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("state_polygon2_triangles", &Class::statePolygon2Triangles, py::return_value_policy::reference);
}

#include "Commands/Sweep.h"
template <typename TDataType>
void declare_sweep_model(py::module& m, std::string typestr) {
	using Class = dyno::SweepModel<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("SweepModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_radius", &Class::varRadius, py::return_value_policy::reference)
		.def("var_curve_ramp", &Class::varCurveRamp, py::return_value_policy::reference)
		.def("var_reverse_normal", &Class::varReverseNormal, py::return_value_policy::reference)
		.def("var_display_points", &Class::varDisplayPoints, py::return_value_policy::reference)
		.def("var_display_wireframe", &Class::varDisplayWireframe, py::return_value_policy::reference)
		.def("var_display_surface", &Class::varDisplaySurface, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("in_spline", &Class::inSpline, py::return_value_policy::reference)
		.def("in_curve", &Class::inCurve, py::return_value_policy::reference);
}

#include "Commands/TextureMeshMerge.h"
template <typename TDataType>
void declare_texture_mesh_merge(py::module& m, std::string typestr) {
	using Class = dyno::TextureMeshMerge<TDataType>;
	using Parent = dyno::ModelEditing<TDataType>;
	std::string pyclass_name = std::string("TextureMeshMerge") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_first", &Class::inFirst, py::return_value_policy::reference)
		.def("in_second", &Class::inSecond, py::return_value_policy::reference)
		.def("state_texture_mesh", &Class::stateTextureMesh, py::return_value_policy::reference);
}

#include "Commands/Transform.h"
template <typename TDataType>
void declare_transform_model(py::module& m, std::string typestr) {
	using Class = dyno::TransformModel<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("TransformModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>TM(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	TM.def(py::init<>())
		.def("in_topology", &Class::inTopology, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("state_point_set", &Class::statePointSet, py::return_value_policy::reference)
		.def("state_edge_set", &Class::stateEdgeSet, py::return_value_policy::reference)
		.def("disable_render", &Class::disableRender)
		.def("transform", &Class::Transform)
		.def_readwrite("inType", &Class::inType);

	py::enum_<typename Class::inputType>(TM, "inputType")
		.value("Point_", Class::inputType::Point_)
		.value("Edge_", Class::inputType::Edge_)
		.value("Triangle_", Class::inputType::Triangle_)
		.value("Null_", Class::inputType::Null_)
		.export_values();
}

#include "Commands/Turning.h"
template <typename TDataType>
void declare_turning_model(py::module& m, std::string typestr) {
	using Class = dyno::TurningModel<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("TurningModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_columns", &Class::varColumns, py::return_value_policy::reference)
		.def("var_end_segment", &Class::varEndSegment, py::return_value_policy::reference)
		.def("var_radius", &Class::varRadius, py::return_value_policy::reference)
		.def("state_polygon_set", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("in_point_set", &Class::inPointSet, py::return_value_policy::reference)
		.def("var_reverse_normal", &Class::varReverseNormal, py::return_value_policy::reference)
		.def("var_use_ramp", &Class::varUseRamp, py::return_value_policy::reference)
		.def("var_curve", &Class::varCurve, py::return_value_policy::reference);
}

#include "Samplers/PointFromCurve.h"
template <typename TDataType>
void declare_point_from_curve(py::module& m, std::string typestr) {
	using Class = dyno::PointFromCurve<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("PointFromCurve") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_uniform_scale", &Class::varUniformScale, py::return_value_policy::reference)
		.def("var_curve", &Class::varCurve, py::return_value_policy::reference)
		.def("state_point_set", &Class::statePointSet, py::return_value_policy::reference)
		.def("state_edge_set", &Class::stateEdgeSet, py::return_value_policy::reference);
}

#include "Samplers/Sampler.h"
template <typename TDataType>
void declare_sampler(py::module& m, std::string typestr) {
	using Class = dyno::Sampler<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("Sampler") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)
		.def("state_point_set", &Class::statePointSet, py::return_value_policy::reference);
}

#include "Samplers/PointsBehindMesh.h"
template <typename TDataType>
void declare_points_behind_mesh(py::module& m, std::string typestr) {
	using Class = dyno::PointsBehindMesh<TDataType>;
	using Parent = dyno::Sampler<TDataType>;
	std::string pyclass_name = std::string("PointsBehindMesh") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_thickness", &Class::varThickness, py::return_value_policy::reference)
		.def("var_sampling_distance", &Class::varSamplingDistance, py::return_value_policy::reference)
		.def("var_generating_direction", &Class::varGeneratingDirection, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("state_position", &Class::statePosition, py::return_value_policy::reference)
		.def("state_plane", &Class::statePlane, py::return_value_policy::reference)
		.def("state_point_normal", &Class::statePointNormal, py::return_value_policy::reference)
		.def("out_point_growth_direction", &Class::outPointGrowthDirection, py::return_value_policy::reference)
		.def("state_point_belong_triangle_index", &Class::statePointBelongTriangleIndex, py::return_value_policy::reference)
		.def("out_sampling_distance", &Class::outSamplingDistance, py::return_value_policy::reference);
}

#include "Samplers/PoissonPlane.h"
template <typename TDataType>
void declare_poisson_plane(py::module& m, std::string typestr) {
	using Class = dyno::PoissonPlane<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("PoissonPlane") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("construct_grid", &Class::ConstructGrid)
		.def("collision_judge", &Class::collisionJudge)
		.def("var_sampling_distance", &Class::varSamplingDistance, py::return_value_policy::reference)
		.def("var_upper", &Class::varUpper, py::return_value_policy::reference)
		.def("var_lower", &Class::varLower, py::return_value_policy::reference)
		.def("compute", &Class::compute)
		.def("get_points", &Class::getPoints);
}

#include "Samplers/ShapeSampler.h"
template <typename TDataType>
void declare_shape_sampler(py::module& m, std::string typestr) {
	using Class = dyno::ShapeSampler<TDataType>;
	using Parent = dyno::Sampler<TDataType>;
	std::string pyclass_name = std::string("CubeSampler") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEF_VAR
		.def("var_sampling_distance", &Class::varSamplingDistance, py::return_value_policy::reference)
		//DEF_VAR_IN
		.def("import_shape", &Class::importShape, py::return_value_policy::reference)
		.def("get_shape", &Class::getShape);
}

#include "CollisionDetector.h"
template <typename TDataType>
void declare_collision_detector(py::module& m, std::string typestr) {
	using Class = dyno::CollisionDetector<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("CollisionDetector") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)

		.def("import_shape_a", &Class::importShapeA, py::return_value_policy::reference)
		.def("get_shape_a", &Class::getShapeA)
		.def("import_shape_b", &Class::importShapeB, py::return_value_policy::reference)
		.def("get_shape_b", &Class::getShapeB)
		.def("state_contacts", &Class::stateContacts, py::return_value_policy::reference)
		.def("state_normals", &Class::stateNormals, py::return_value_policy::reference);
}

#include "GltfLoader.h"
void declare_bounding_box_of_texture_mesh(py::module& m) {
	using Class = dyno::BoundingBoxOfTextureMesh;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("BoundingBoxOfTextureMesh");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_shape_id", &Class::varShapeId, py::return_value_policy::reference)
		.def("var_center", &Class::varCenter, py::return_value_policy::reference)
		.def("var_lower_bound", &Class::varLowerBound, py::return_value_policy::reference)
		.def("var_upper_bound", &Class::varUpperBound, py::return_value_policy::reference)
		.def("in_texture_mesh", &Class::inTextureMesh, py::return_value_policy::reference)
		.def("out_bounding_box", &Class::outBoundingBox, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_gltf_loader(py::module& m, std::string typestr) {
	using Class = dyno::GltfLoader<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("GltfLoader") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)
		.def("var_file_name", &Class::varFileName, py::return_value_policy::reference)
		.def("var_import_animation", &Class::varImportAnimation, py::return_value_policy::reference)
		.def("var_joint_radius", &Class::varJointRadius, py::return_value_policy::reference)
		.def("state_tex_coord_0", &Class::stateTexCoord_0, py::return_value_policy::reference)
		.def("state_tex_coord_1", &Class::stateTexCoord_1, py::return_value_policy::reference)
		.def("state_initial_matrix", &Class::stateInitialMatrix, py::return_value_policy::reference)

		.def("state_transform", &Class::stateTransform, py::return_value_policy::reference)
		.def("state_skin", &Class::stateSkin, py::return_value_policy::reference)

		.def("state_joint_inverse_bind_matrix", &Class::stateJointInverseBindMatrix, py::return_value_policy::reference)
		.def("state_joint_local_matrix", &Class::stateJointLocalMatrix, py::return_value_policy::reference)
		.def("state_jont_world_matrix", &Class::stateJointWorldMatrix, py::return_value_policy::reference)

		.def("state_joints_data", &Class::stateJointsData, py::return_value_policy::reference)
		.def("state_texture_mesh", &Class::stateTextureMesh, py::return_value_policy::reference)

		.def("state_shape_center", &Class::stateShapeCenter, py::return_value_policy::reference)
		.def("state_joint_set", &Class::stateJointSet, py::return_value_policy::reference)
		.def("state_animation", &Class::stateAnimation, py::return_value_policy::reference);
}

#include "Group.h"
template <typename TDataType>
void declare_group(py::module& m, std::string typestr) {
	using Class = dyno::Group<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("Group") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_point_id", &Class::varPointId, py::return_value_policy::reference)
		.def("var_edge_id", &Class::varEdgeId, py::return_value_policy::reference)
		.def("var_primitive_id", &Class::varPrimitiveId, py::return_value_policy::reference)
		.def("in_point_id", &Class::inPointId, py::return_value_policy::reference)
		.def("in_edge_id", &Class::inEdgeId, py::return_value_policy::reference)
		.def("in_primitive_id", &Class::inPrimitiveId, py::return_value_policy::reference)

		.def("get_select_primitives", &Class::getSelectPrimitives)
		.def("get_select_edges", &Class::getSelectEdges)
		.def("get_select_points", &Class::getSelectPoints)
		.def_readwrite("selectedPointID", &Class::selectedPointID)
		.def_readwrite("selectedEdgeID", &Class::selectedEdgeID)
		.def_readwrite("selectedPrimitiveID", &Class::selectedPrimitiveID);
}

#include "JointDeform.h"
template <typename TDataType>
void declare_joint_deform(py::module& m, std::string typestr) {
	using Class = dyno::JointDeform<TDataType>;
	using Parent = dyno::ModelEditing<TDataType>;
	std::string pyclass_name = std::string("JointDeform") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_joint", &Class::inJoint, py::return_value_policy::reference)
		.def("in_skin", &Class::inSkin, py::return_value_policy::reference)
		.def("in_instance_transform", &Class::inInstanceTransform, py::return_value_policy::reference)
		.def("state_texture_mesh", &Class::stateTextureMesh, py::return_value_policy::reference);
}

#include "JointInfo.h"
void declare_joint_info(py::module& m) {
	using Class = dyno::JointInfo;
	using Parent = dyno::OBase;
	typedef int joint;
	std::string pyclass_name = std::string("JointInfo");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<dyno::Array<dyno::Mat4f, DeviceType::GPU>&, dyno::Array<dyno::Mat4f, DeviceType::GPU>&, dyno::Array<dyno::Mat4f, DeviceType::GPU>&, std::vector<int>&, std::map<joint, std::vector<joint>>&, std::map<joint, dyno::Vec3f>&, std::map<joint, dyno::Vec3f>&, std::map<joint, dyno::Quat1f>&>())
		.def("update_joint_info", &Class::UpdateJointInfo)
		.def("set_joint", &Class::setJoint)
		.def("is_empty", &Class::isEmpty)
		.def("update_world_matrix_by_transform", &Class::updateWorldMatrixByTransform)
		.def("update_current_pose", &Class::updateCurrentPose)
		.def("set_joint_name", &Class::setJointName)
		.def_readwrite("mJointName", &Class::mJointName)

		.def_readwrite("mJointInverseBindMatrix", &Class::mJointInverseBindMatrix)
		.def_readwrite("mJointLocalMatrix", &Class::mJointLocalMatrix)
		.def_readwrite("mJointWorldMatrix", &Class::mJointWorldMatrix)

		.def_readwrite("mBindPoseTranslation", &Class::mBindPoseTranslation)
		.def_readwrite("mBindPoseScale", &Class::mBindPoseScale)
		.def_readwrite("mBindPoseRotation", &Class::mBindPoseRotation)

		.def_readwrite("mCurrentTranslation", &Class::mCurrentTranslation)
		.def_readwrite("mCurrentRotation", &Class::mCurrentRotation)
		.def_readwrite("mCurrentScale", &Class::mCurrentScale)

		.def_readwrite("mAllJoints", &Class::mAllJoints)
		.def_readwrite("mJointDir", &Class::mJointDir)

		.def_readwrite("mMaxJointID", &Class::mMaxJointID);
}

void declare_joint_animation_info(py::module& m) {
	using Class = dyno::JointAnimationInfo;
	using Parent = dyno::OBase;
	typedef int joint;
	std::string pyclass_name = std::string("JointAnimationInfo");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("set_animation_data", &Class::setAnimationData)
		.def("update_joints_transform", &Class::updateJointsTransform)
		.def("update_transform", &Class::updateTransform)
		.def("get_joints_translation", &Class::getJointsTranslation)
		.def("get_joints_rotation", &Class::getJointsRotation)
		.def("get_joints_scale", &Class::getJointsScale)
		.def("get_total_time", &Class::getTotalTime)
		.def("find_max_smaller_index", &Class::findMaxSmallerIndex)
		.def("lerp", &Class::lerp)
		.def("normalize", &Class::normalize)
		.def("slerp", &Class::slerp)
		.def("nlerp", &Class::nlerp)
		.def("get_joint_dir", &Class::getJointDir)
		.def("set_loop", &Class::setLoop)
		.def("get_pose", &Class::getPose)
		.def("get_current_animation_time", &Class::getCurrentAnimationTime)
		.def("get_blend_in_time", &Class::getBlendInTime)
		.def("get_blend_out_time", &Class::getBlendOutTime)
		.def("get_play_rate", &Class::getPlayRate)

		.def_readwrite("mSkeleton", &Class::mSkeleton);
}

#include "ModelEditing.h"
template <typename TDataType>
void declare_model_editing(py::module& m, std::string typestr) {
	using Class = dyno::ModelEditing<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ModelEditing") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_node_type", &Class::getNodeType);
}

#include "NormalVisualization.h"
template <typename TDataType>
void declare_normal_visualization(py::module& m, std::string typestr) {
	using Class = dyno::NormalVisualization<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("Normal") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)

		.def("var_length", &Class::varLength, py::return_value_policy::reference)
		.def("var_normalize", &Class::varNormalize, py::return_value_policy::reference)

		.def("var_line_width", &Class::varLineWidth, py::return_value_policy::reference)
		.def("var_show_wireframe", &Class::varShowWireframe, py::return_value_policy::reference)
		.def("var_arrow_resolution", &Class::varArrowResolution, py::return_value_policy::reference)

		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("in_in_normal", &Class::inInNormal, py::return_value_policy::reference)
		.def("in_scalar", &Class::inScalar, py::return_value_policy::reference)

		.def("state_normal_set", &Class::stateNormalSet, py::return_value_policy::reference)
		.def("state_normal", &Class::stateNormal, py::return_value_policy::reference)
		.def("state_tiangle_center", &Class::stateTriangleCenter, py::return_value_policy::reference)

		.def("state_arrow_cylinder", &Class::stateArrowCylinder, py::return_value_policy::reference)
		.def("state_arrow_cone", &Class::stateArrowCone, py::return_value_policy::reference)
		.def("state_transforms_cylinder", &Class::stateTransformsCylinder, py::return_value_policy::reference)
		.def("state_transforms_cone", &Class::stateTransformsCone, py::return_value_policy::reference);
}

#include "SplineConstraint.h"
template <typename TDataType>
void declare_spline_constraint(py::module& m, std::string typestr) {
	using Class = dyno::SplineConstraint<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("SplineConstraint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_spline", &Class::inSpline, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("var_velocity", &Class::varVelocity, py::return_value_policy::reference)
		.def("var_offest", &Class::varOffest, py::return_value_policy::reference)
		.def("var_accelerate", &Class::varAccelerate, py::return_value_policy::reference)
		.def("var_accelerated_speed", &Class::varAcceleratedSpeed, py::return_value_policy::reference)
		.def("state_topology", &Class::stateTopology, py::return_value_policy::reference);
}

#include "StaticTriangularMesh.h"
template <typename TDataType>
void declare_static_triangular_mesh(py::module& m, std::string typestr) {
	using Class = dyno::StaticTriangularMesh<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("StaticTriangularMesh") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)

		.def("var_file_name", &Class::varFileName, py::return_value_policy::reference)

		.def("state_initial_triangle_set", &Class::stateInitialTriangleSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference);
}

#include "Subdivide.h"
template <typename TDataType>
void declare_subdivide(py::module& m, std::string typestr) {
	using Class = dyno::Subdivide<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("Subdivide") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("var_step", &Class::varStep, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inInTriangleSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference);
}

#include "VectorVisualNode.h"
template <typename TDataType>
void declare_vector_visual_node(py::module& m, std::string typestr) {
	using Class = dyno::VectorVisualNode<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("VectorVisualNode") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>VVN(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	VVN.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)

		.def("var_length", &Class::varLength, py::return_value_policy::reference)
		.def("var_normalize", &Class::varNormalize, py::return_value_policy::reference)

		.def("var_line_mode", &Class::varLineMode, py::return_value_policy::reference)
		.def("var_line_width", &Class::varLineWidth, py::return_value_policy::reference)
		.def("var_arrow_resolution", &Class::varArrowResolution, py::return_value_policy::reference)
		.def("var_debug", &Class::varDebug, py::return_value_policy::reference)

		.def("in_point_set", &Class::inPointSet, py::return_value_policy::reference)
		.def("in_vector", &Class::inInVector, py::return_value_policy::reference)
		.def("in_scalar", &Class::inScalar, py::return_value_policy::reference)

		.def("state_normal_set", &Class::stateNormalSet, py::return_value_policy::reference)
		.def("state_normal", &Class::stateNormal, py::return_value_policy::reference)

		.def("state_arrow_cylinder", &Class::stateArrowCylinder, py::return_value_policy::reference)
		.def("state_arrow_cone", &Class::stateArrowCone, py::return_value_policy::reference)
		.def("state_transforms_cylinder", &Class::stateTransformsCylinder, py::return_value_policy::reference)
		.def("state_transforms_cone", &Class::stateTransformsCone, py::return_value_policy::reference);

	py::enum_<typename Class::LineMode>(VVN, "LineMode")
		.value("Line", Class::LineMode::Line)
		.value("Cylnder", Class::LineMode::Cylnder)
		.value("Arrow", Class::LineMode::Arrow)
		.export_values();
}

void pybind_modeling(py::module& m);
