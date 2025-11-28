#pragma once
#include "PyCommon.h"

//#include "PyFramework.h"
#include "BasicShapes/BasicShape.h"
template <typename TDataType>
void declare_basic_shape(py::module& m, std::string typestr) {
	using Class = dyno::BasicShape<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("BasicShape") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>BS(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	BS.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("getShapeType", &Class::getShapeType);
}

#include "BasicShapes/CapsuleModel.h"
template <typename TDataType>
void declare_capsule_model(py::module& m, std::string typestr) {
	using Class = dyno::CapsuleModel<TDataType>;
	using Parent = dyno::BasicShape<TDataType>;

	class CapsuleModelTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::CapsuleModel<TDataType>,
				resetStates
			);
		}
	};

	class CapsuleModelPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("CapsuleModel") + typestr;
	py::class_<Class, Parent, CapsuleModelTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("getShapeType", &Class::getShapeType)
		.def("boundingBox", &Class::boundingBox)
		.def("varCenter", &Class::varCenter, py::return_value_policy::reference)
		.def("varRadius", &Class::varRadius, py::return_value_policy::reference)
		.def("varLatitude", &Class::varLatitude, py::return_value_policy::reference)
		.def("varLongitude", &Class::varLongitude, py::return_value_policy::reference)
		.def("varHeight", &Class::varHeight, py::return_value_policy::reference)
		.def("varHeightSegment", &Class::varHeightSegment, py::return_value_policy::reference)
		.def("statePolygonSet", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("outCapsule", &Class::outCapsule, py::return_value_policy::reference)
		// protected
		.def("resetStates", &CapsuleModelPublicist::resetStates);
}

#include "BasicShapes/ConeModel.h"
template <typename TDataType>
void declare_cone_model(py::module& m, std::string typestr) {
	using Class = dyno::ConeModel<TDataType>;
	using Parent = dyno::BasicShape<TDataType>;

	class ConeModelTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ConeModel<TDataType>,
				resetStates
			);
		}
	};

	class ConeModelPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("ConeModel") + typestr;
	py::class_<Class, Parent, ConeModelTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("getShapeType", &Class::getShapeType)
		.def("varColumns", &Class::varColumns, py::return_value_policy::reference)
		.def("varRow", &Class::varRow, py::return_value_policy::reference)
		.def("varEndSegment", &Class::varEndSegment, py::return_value_policy::reference)
		.def("varRadius", &Class::varRadius, py::return_value_policy::reference)
		.def("varHeight", &Class::varHeight, py::return_value_policy::reference)
		.def("statePolygonSet", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("outCone", &Class::outCone, py::return_value_policy::reference)
		.def("boundingBox", &Class::boundingBox)
		// protected
		.def("resetStates", &ConeModelPublicist::resetStates);
}

#include "BasicShapes/CubeModel.h"
template <typename TDataType>
void declare_cube_model(py::module& m, std::string typestr) {
	using Class = dyno::CubeModel<TDataType>;
	using Parent = dyno::BasicShape<TDataType>;

	class CubeModelTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::CubeModel<TDataType>,
				resetStates
			);
		}
	};

	class CubeModelPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("CubeModel") + typestr;
	py::class_<Class, Parent, CubeModelTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("getShapeType", &Class::getShapeType)
		.def("boundingBox", &Class::boundingBox)
		.def("varLength", &Class::varLength, py::return_value_policy::reference)
		.def("varSegments", &Class::varSegments, py::return_value_policy::reference)
		.def("statePolygonSet", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("stateQuadSet", &Class::stateQuadSet, py::return_value_policy::reference)
		.def("outCube", &Class::outCube, py::return_value_policy::reference)
		// protected
		.def("resetStates", &CubeModelPublicist::resetStates);
}

#include "BasicShapes/CylinderModel.h"
template <typename TDataType>
void declare_cylinder_model(py::module& m, std::string typestr) {
	using Class = dyno::CylinderModel<TDataType>;
	using Parent = dyno::BasicShape<TDataType>;

	class CylinderModelTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::CylinderModel<TDataType>,
				resetStates
			);
		}
	};

	class CylinderModelPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::varChanged;
	};

	std::string pyclass_name = std::string("CylinderModel") + typestr;
	py::class_<Class, Parent, CylinderModelTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("getShapeType", &Class::getShapeType)

		.def("varColumns", &Class::varColumns, py::return_value_policy::reference)
		.def("varRow", &Class::varRow, py::return_value_policy::reference)
		.def("varEndSegment", &Class::varEndSegment, py::return_value_policy::reference)
		.def("varRadius", &Class::varRadius, py::return_value_policy::reference)
		.def("varHeight", &Class::varHeight, py::return_value_policy::reference)
		.def("statePolygonSet", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("outCylinder", &Class::outCylinder, py::return_value_policy::reference)

		.def("boundingBox", &Class::boundingBox)
		// protected
		.def("resetStates", &CylinderModelPublicist::resetStates)
		.def("varChanged", &CylinderModelPublicist::varChanged);
}

#include "BasicShapes/PlaneModel.h"
template <typename TDataType>
void declare_plane_model(py::module& m, std::string typestr) {
	using Class = dyno::PlaneModel<TDataType>;
	using Parent = dyno::BasicShape<TDataType>;

	class PlaneModelTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PlaneModel<TDataType>,
				resetStates
			);
		}
	};

	class PlaneModelPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("PlaneModel") + typestr;
	py::class_<Class, Parent, PlaneModelTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("getShapeType", &Class::getShapeType)
		.def("boundingBox", &Class::boundingBox)

		.def("varLengthX", &Class::varLengthX, py::return_value_policy::reference)
		.def("varLengthZ", &Class::varLengthZ, py::return_value_policy::reference)

		.def("varSegmentX", &Class::varSegmentX, py::return_value_policy::reference)
		.def("varSegmentZ", &Class::varSegmentZ, py::return_value_policy::reference)
		.def("statePolygonSet", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("stateQuadSet", &Class::stateQuadSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &PlaneModelPublicist::resetStates);
}

#include "BasicShapes/SphereModel.h"
template <typename TDataType>
void declare_sphere_model(py::module& m, std::string typestr) {
	using Class = dyno::SphereModel<TDataType>;
	using Parent = dyno::BasicShape<TDataType>;

	class SphereModelTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SphereModel<TDataType>,
				resetStates
			);
		}
	};

	class SphereModelPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("SphereModel") + typestr;
	py::class_<Class, Parent, SphereModelTrampoline, std::shared_ptr<Class>>SM(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	SM.def(py::init<>())
		.def("caption", &Class::caption)
		.def("getShapeType", &Class::getShapeType)
		.def("boundingBox", &Class::boundingBox)

		.def("varCenter", &Class::varCenter, py::return_value_policy::reference)
		.def("varRadius", &Class::varRadius, py::return_value_policy::reference)
		.def("varLatitude", &Class::varLatitude, py::return_value_policy::reference)
		.def("varLongitude", &Class::varLongitude, py::return_value_policy::reference)
		.def("varIcosahedronStep", &Class::varIcosahedronStep, py::return_value_policy::reference)
		.def("statePolygonSet", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)

		.def("outSphere", &Class::outSphere, py::return_value_policy::reference)
		.def("varType", &Class::varType, py::return_value_policy::reference)
		// protected
		.def("resetStates", &SphereModelPublicist::resetStates);

	py::enum_<typename Class::SphereType>(SM, "SphereType")
		.value("Standard", Class::SphereType::Standard)
		.value("Icosahedron", Class::SphereType::Icosahedron)
		.export_values();
}


#include "BasicShapes/TetModel.h"
template <typename TDataType>
void declare_tet_model(py::module& m, std::string typestr) {
	using Class = dyno::TetModel<TDataType>;
	using Parent = dyno::BasicShape<TDataType>;

	class TetModelTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TetModel<TDataType>,
				resetStates
			);
		}
	};

	class TetModelPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("TetModel") + typestr;
	py::class_<Class, Parent, TetModelTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("getShapeType", &Class::getShapeType)
		.def("boundingBox", &Class::boundingBox)

		.def("varV0", &Class::varV0, py::return_value_policy::reference)
		.def("varV1", &Class::varV1, py::return_value_policy::reference)
		.def("varV2", &Class::varV2, py::return_value_policy::reference)
		.def("varV3", &Class::varV3, py::return_value_policy::reference)
		.def("stateTetSet", &Class::stateTetSet, py::return_value_policy::reference)
		.def("outTet", &Class::outTet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &TetModelPublicist::resetStates);
}

#include "Commands/ConvertToTextureMesh.h"
template <typename TDataType>
void declare_convert_to_texture_mesh(py::module& m, std::string typestr) {
	using Class = dyno::ConvertToTextureMesh<TDataType>;
	using Parent = dyno::ModelEditing<TDataType>;

	class ConvertToTextureMeshTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ConvertToTextureMesh<TDataType>,
				resetStates
			);
		}
	};

	class ConvertToTextureMeshPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::varChanged;
		using Class::createTextureMesh;
		using Class::createUV;
		using Class::createMaterial;
		using Class::createNormal;
		using Class::moveToCenter;
	};

	std::string pyclass_name = std::string("ConvertToTextureMesh") + typestr;
	py::class_<Class, Parent, ConvertToTextureMeshTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varDiffuseTexture", &Class::varDiffuseTexture, py::return_value_policy::reference)
		.def("varNormalTexture", &Class::varNormalTexture, py::return_value_policy::reference)
		.def("varUvScaleU", &Class::varUvScaleU, py::return_value_policy::reference)
		.def("varUvScaleV", &Class::varUvScaleV, py::return_value_policy::reference)
		.def("varUseBoundingTransform", &Class::varUseBoundingTransform, py::return_value_policy::reference)
		.def("in_topolopy", &Class::inTopology, py::return_value_policy::reference)
		.def("stateTextureMesh", &Class::stateTextureMesh, py::return_value_policy::reference)
		// protected
		.def("resetStates", &ConvertToTextureMeshPublicist::resetStates)
		.def("varChanged", &ConvertToTextureMeshPublicist::varChanged)
		.def("createTextureMesh", &ConvertToTextureMeshPublicist::createTextureMesh)
		.def("createUV", &ConvertToTextureMeshPublicist::createUV)
		.def("createMaterial", &ConvertToTextureMeshPublicist::createMaterial)
		.def("createNormal", &ConvertToTextureMeshPublicist::createNormal)
		.def("moveToCenter", &ConvertToTextureMeshPublicist::moveToCenter);
}

#include "Commands/CopyModel.h"
template <typename TDataType>
void declare_copy_model(py::module& m, std::string typestr) {
	using Class = dyno::CopyModel<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;

	class CopyModelTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::CopyModel<TDataType>,
				resetStates
			);
		}
	};

	class CopyModelPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("CopyModel") + typestr;
	py::class_<Class, Parent, CopyModelTrampoline, std::shared_ptr<Class>>CM(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	CM.def(py::init<>())
		.def("varTotalNumber", &Class::varTotalNumber, py::return_value_policy::reference)
		.def("varCopyTransform", &Class::varCopyTransform, py::return_value_policy::reference)
		.def("varCopyRotation", &Class::varCopyRotation, py::return_value_policy::reference)
		.def("varCopyScale", &Class::varCopyScale, py::return_value_policy::reference)
		.def("varScaleMode", &Class::varScaleMode, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("inTriangleSetIn", &Class::inTriangleSetIn, py::return_value_policy::reference)
		// protected
		.def("resetStates", &CopyModelPublicist::resetStates);

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

	class CopyToPointTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::CopyToPoint<TDataType>,
				resetStates
			);
		}
	};

	class CopyToPointPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("CopyToPoint") + typestr;
	py::class_<Class, Parent, CopyToPointTrampoline,  std::shared_ptr<Class>>CTP(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	CTP.def(py::init<>())
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("inTriangleSetIn", &Class::inTriangleSetIn, py::return_value_policy::reference)
		.def("inTargetPointSet", &Class::inTargetPointSet, py::return_value_policy::reference)
		.def("disableRender", &Class::disableRender)
		// protected
		.def("resetStates", &CopyToPointPublicist::resetStates);

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

	class EarClipperTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::EarClipper<TDataType>,
				resetStates
			);
		}
	};

	class EarClipperPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("EarClipper") + typestr;
	py::class_<Class, Parent, EarClipperTrampoline,  std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<std::vector<Coord>, std::vector<dyno::TopologyModule::Triangle>&>())
		.def("varChanged", &Class::varChanged)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("inPointSet", &Class::inPointSet, py::return_value_policy::reference)
		.def("varReverseNormal", &Class::varReverseNormal, py::return_value_policy::reference)

		.def("polyClip", py::overload_cast<std::vector<Coord>, std::vector<dyno::TopologyModule::Triangle>&>(&Class::polyClip))
		.def("polyClip", py::overload_cast<dyno::Array<Coord, DeviceType::GPU>, std::vector<dyno::TopologyModule::Triangle>&>(&Class::polyClip))
		.def("polyClip", py::overload_cast<dyno::Array<Coord, DeviceType::CPU>, std::vector<dyno::TopologyModule::Triangle>&>(&Class::polyClip))
		// protected
		.def("resetStates", &EarClipperPublicist::resetStates);
}

#include "Commands/EditableMesh.h"
template <typename TDataType>
void declare_editable_mesh(py::module& m, std::string typestr) {
	using Class = dyno::EditableMesh<TDataType>;
	using Parent = dyno::PolygonSetToTriangleSetNode<TDataType>;

	class EditableMeshTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::EditableMesh<TDataType>,
				resetStates
			);
		}
	};

	class EditableMeshPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("EditableMesh") + typestr;
	py::class_<Class, Parent, EditableMeshTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("inPolygonSet", &Class::inPolygonSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("stateVertexNormal", &Class::stateVertexNormal, py::return_value_policy::reference)
		.def("stateTriangleNormal", &Class::stateTriangleNormal, py::return_value_policy::reference)
		.def("statePolygonNormal", &Class::statePolygonNormal, py::return_value_policy::reference)
		// protected
		.def("resetStates", &EditableMeshPublicist::resetStates);
}

#include "Commands/ExtractShape.h"
template <typename TDataType>
void declare_extract_shape(py::module& m, std::string typestr) {
	using Class = dyno::ExtractShape<TDataType>;
	using Parent = dyno::ModelEditing<TDataType>;

	class ExtractShapeTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ExtractShape<TDataType>,
				resetStates
			);
		}
	};

	class ExtractShapePublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("ExtractShape") + typestr;
	py::class_<Class, Parent, ExtractShapeTrampoline,  std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varShapeId", &Class::varShapeId, py::return_value_policy::reference)
		.def("varShapeTransform", &Class::varShapeTransform, py::return_value_policy::reference)
		.def("varOffset", &Class::varOffset, py::return_value_policy::reference)
		.def("inInTextureMesh", &Class::inInTextureMesh, py::return_value_policy::reference)
		.def("stateResult", &Class::stateResult, py::return_value_policy::reference)
		// protected
		.def("resetStates", &ExtractShapePublicist::resetStates);
}

#include "Commands/Extrude.h"
template <typename TDataType>
void declare_extrude_model(py::module& m, std::string typestr) {
	using Class = dyno::ExtrudeModel<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;

	class ExtrudeModelTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ExtrudeModel<TDataType>,
				resetStates
			);
		}
	};

	class ExtrudeModelPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::varChanged;
	};

	std::string pyclass_name = std::string("ExtrudeModel") + typestr;
	py::class_<Class, Parent, ExtrudeModelTrampoline,  std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varRow", &Class::varRow, py::return_value_policy::reference)
		.def("varHeight", &Class::varHeight, py::return_value_policy::reference)
		.def("varReverseNormal", &Class::varReverseNormal, py::return_value_policy::reference)
		.def("varCurve", &Class::varCurve, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("inPointSet", &Class::inPointSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &ExtrudeModelPublicist::resetStates)
		.def("varChanged", &ExtrudeModelPublicist::varChanged);
}

#include "Commands/Merge.h"
template <typename TDataType>
void declare_merge(py::module& m, std::string typestr) {
	using Class = dyno::Merge<TDataType>;
	using Parent = dyno::Node;

	class MergeTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Merge<TDataType>,
				resetStates
			);
		}

		void preUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Merge<TDataType>,
				preUpdateStates
			);
		}
	};

	class MergePublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::preUpdateStates;
	};

	std::string pyclass_name = std::string("Merge") + typestr;
	py::class_<Class, Parent, MergeTrampoline,  std::shared_ptr<Class>>M(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	M.def(py::init<>())
		.def("stateTriangleSets", &Class::stateTriangleSets, py::return_value_policy::reference)
		.def("inTriangleSets", &Class::inTriangleSets, py::return_value_policy::reference)
		.def("varUpdateMode", &Class::varUpdateMode, py::return_value_policy::reference)
		.def("caption", &Class::caption)
		// protected
		.def("resetStates", &MergePublicist::resetStates)
		.def("preUpdateStates", &MergePublicist::preUpdateStates);

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

	class PointClipTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PointClip<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PointClip<TDataType>,
				updateStates
			);
		}
	};

	class PointClipPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
		using Class::clip;
		using Class::transformPlane;
		using Class::showPlane;
	};

	std::string pyclass_name = std::string("PointClip") + typestr;
	py::class_<Class, Parent, PointClipTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inPointSet", &Class::inPointSet, py::return_value_policy::reference)
		.def("varPlaneSize", &Class::varPlaneSize, py::return_value_policy::reference)
		.def("varReverse", &Class::varReverse, py::return_value_policy::reference)
		.def("varPointSize", &Class::varPointSize, py::return_value_policy::reference)
		.def("varPointColor", &Class::varPointColor, py::return_value_policy::reference)
		.def("varShowPlane", &Class::varShowPlane, py::return_value_policy::reference)
		.def("stateClipPlane", &Class::stateClipPlane, py::return_value_policy::reference)
		.def("statePointSet", &Class::statePointSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &PointClipPublicist::resetStates)
		.def("updateStates", &PointClipPublicist::updateStates)
		.def("clip", &PointClipPublicist::clip)
		.def("transformPlane", &PointClipPublicist::transformPlane)
		.def("showPlane", &PointClipPublicist::showPlane);
}

#include "Commands/PolyExtrude.h"
template <typename TDataType>
void declare_poly_extrude(py::module& m, std::string typestr) {
	using Class = dyno::PolyExtrude<TDataType>;
	using Parent = dyno::Group<TDataType>;

	class PolyExtrudeTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PolyExtrude<TDataType>,
				resetStates
			);
		}
	};

	class PolyExtrudePublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::pointcount;
		using Class::substrFromTwoString;
	};

	std::string pyclass_name = std::string("PolyExtrude") + typestr;
	py::class_<Class, Parent, PolyExtrudeTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varDivisions", &Class::varDivisions, py::return_value_policy::reference)
		.def("varDistance", &Class::varDistance, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("stateNormalSet", &Class::stateNormalSet, py::return_value_policy::reference)
		.def("varReverseNormal", &Class::varReverseNormal, py::return_value_policy::reference)
		// protected
		.def("resetStates", &PolyExtrudePublicist::resetStates)
		.def("pointcount", &PolyExtrudePublicist::pointcount)
		.def("substrFromTwoString", &PolyExtrudePublicist::substrFromTwoString);
}

#include "Commands/PolygonSetToTriangleSet.h"
template <typename TDataType>
void declare_polygon_set_to_triangle_set_module(py::module& m, std::string typestr) {
	using Class = dyno::PolygonSetToTriangleSetModule<TDataType>;
	using Parent = dyno::TopologyMapping;

	class PolygonSetToTriangleSetModuleTrampoline : public Class
	{
	public:
		using Class::Class;

		bool apply() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::PolygonSetToTriangleSetModule<TDataType>,
				apply
			);
		}
	};

	class PolygonSetToTriangleSetModulePublicist : public Class
	{
	public:
		using Class::apply;
	};

	std::string pyclass_name = std::string("PolygonSetToTriangleSetModule") + typestr;
	py::class_<Class, Parent, PolygonSetToTriangleSetModuleTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("inPolygonSet", &Class::inPolygonSet, py::return_value_policy::reference)
		.def("outTriangleSet", &Class::outTriangleSet, py::return_value_policy::reference)
		.def("outPolygon2Triangles", &Class::outPolygon2Triangles, py::return_value_policy::reference)

		.def("convert", &Class::convert)
		// protected
		.def("apply", &PolygonSetToTriangleSetModulePublicist::apply);
}

template <typename TDataType>
void declare_polygon_set_to_triangle_set_node(py::module& m, std::string typestr) {
	using Class = dyno::PolygonSetToTriangleSetNode<TDataType>;
	using Parent = dyno::Node;

	class PolygonSetToTriangleSetNodeTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PolygonSetToTriangleSetNode<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PolygonSetToTriangleSetNode<TDataType>,
				updateStates
			);
		}
	};

	class PolygonSetToTriangleSetNodePublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("PolygonSetToTriangleSetNode") + typestr;
	py::class_<Class, Parent, PolygonSetToTriangleSetNodeTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("inPolygonSet", &Class::inPolygonSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("statePolygon2Triangles", &Class::statePolygon2Triangles, py::return_value_policy::reference)
		// protected
		.def("resetStates", &PolygonSetToTriangleSetNodePublicist::resetStates)
		.def("updateStates", &PolygonSetToTriangleSetNodePublicist::updateStates);
}

#include "Commands/Sweep.h"
template <typename TDataType>
void declare_sweep_model(py::module& m, std::string typestr) {
	using Class = dyno::SweepModel<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;

	class SweepModelTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SweepModel<TDataType>,
				resetStates
			);
		}
	};

	class SweepModelPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::varChanged;
		using Class::displayChanged;
		using Class::RealScale;
	};

	std::string pyclass_name = std::string("SweepModel") + typestr;
	py::class_<Class, Parent, SweepModelTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varRadius", &Class::varRadius, py::return_value_policy::reference)
		.def("varCurveRamp", &Class::varCurveRamp, py::return_value_policy::reference)
		.def("varReverseNormal", &Class::varReverseNormal, py::return_value_policy::reference)
		.def("varDisplayPoints", &Class::varDisplayPoints, py::return_value_policy::reference)
		.def("varDisplayWireframe", &Class::varDisplayWireframe, py::return_value_policy::reference)
		.def("varDisplaySurface", &Class::varDisplaySurface, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("inSpline", &Class::inSpline, py::return_value_policy::reference)
		.def("inCurve", &Class::inCurve, py::return_value_policy::reference)
		// protected
		.def("resetStates", &SweepModelPublicist::resetStates)
		.def("varChanged", &SweepModelPublicist::varChanged)
		.def("displayChanged", &SweepModelPublicist::displayChanged)
		.def("RealScale", &SweepModelPublicist::RealScale);
}

#include "Commands/TextureMeshMerge.h"
template <typename TDataType>
void declare_texture_mesh_merge(py::module& m, std::string typestr) {
	using Class = dyno::TextureMeshMerge<TDataType>;
	using Parent = dyno::ModelEditing<TDataType>;

	class TextureMeshMergeTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TextureMeshMerge<TDataType>,
				resetStates
			);
		}
	};

	class TextureMeshMergePublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::merge;
	};

	std::string pyclass_name = std::string("TextureMeshMerge") + typestr;
	py::class_<Class, Parent, TextureMeshMergeTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inFirst", &Class::inFirst, py::return_value_policy::reference)
		.def("inSecond", &Class::inSecond, py::return_value_policy::reference)
		.def("stateTextureMesh", &Class::stateTextureMesh, py::return_value_policy::reference)
		// protected
		.def("resetStates", &TextureMeshMergePublicist::resetStates)
		.def("merge", &TextureMeshMergePublicist::merge);
}

#include "Commands/Transform.h"
template <typename TDataType>
void declare_transform_model(py::module& m, std::string typestr) {
	using Class = dyno::TransformModel<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;

	class TransformModelTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TransformModel<TDataType>,
				resetStates
			);
		}
	};

	class TransformModelPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("TransformModel") + typestr;
	py::class_<Class, Parent, TransformModelTrampoline, std::shared_ptr<Class>>TM(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	TM.def(py::init<>())
		.def("inTopology", &Class::inTopology, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("statePointSet", &Class::statePointSet, py::return_value_policy::reference)
		.def("stateEdgeSet", &Class::stateEdgeSet, py::return_value_policy::reference)
		.def("disableRender", &Class::disableRender)
		.def("Transform", &Class::Transform)
		.def_readwrite("inType", &Class::inType)
		// protected
		.def("resetStates", &TransformModelPublicist::resetStates);

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

	class TurningModelTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TurningModel<TDataType>,
				resetStates
			);
		}
	};

	class TurningModelPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("TurningModel") + typestr;
	py::class_<Class, Parent, TurningModelTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varColumns", &Class::varColumns, py::return_value_policy::reference)
		.def("varEndSegment", &Class::varEndSegment, py::return_value_policy::reference)
		.def("varRadius", &Class::varRadius, py::return_value_policy::reference)
		.def("statePolygonSet", &Class::statePolygonSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("inPointSet", &Class::inPointSet, py::return_value_policy::reference)
		.def("varReverseNormal", &Class::varReverseNormal, py::return_value_policy::reference)
		.def("varUseRamp", &Class::varUseRamp, py::return_value_policy::reference)
		.def("varCurve", &Class::varCurve, py::return_value_policy::reference)
		// protected
		.def("resetStates", &TurningModelPublicist::resetStates);
}

#include "Commands/TriangleSetToTriangleSets.h"
template <typename TDataType>
void declare_triangle_set_to_triangle_sets(py::module& m, std::string typestr) {
	using Class = dyno::TriangleSetToTriangleSets<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;

	class TriangleSetToTriangleSetsTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TriangleSetToTriangleSets<TDataType>,
				resetStates
			);
		}
	};

	class TriangleSetToTriangleSetsPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("TriangleSetToTriangleSets") + typestr;
	py::class_<Class, Parent, TriangleSetToTriangleSetsTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("stateTriangleSets", &Class::stateTriangleSets, py::return_value_policy::reference)
		// protected
		.def("resetStates", &TriangleSetToTriangleSetsPublicist::resetStates);
}

#include "Commands/ExtractTriangleSets.h"
template <typename TDataType>
void declare_extract_triangle_sets(py::module& m, std::string typestr) {
	using Class = dyno::ExtractTriangleSets<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;

	class ExtractTriangleSetsTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ExtractTriangleSets<TDataType>,
				resetStates
			);
		}
	};

	class ExtractTriangleSetsPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("ExtractTriangleSets") + typestr;
	py::class_<Class, Parent, ExtractTriangleSetsTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varID", &Class::varID, py::return_value_policy::reference)
		.def("varShapeTransform", &Class::varShapeTransform, py::return_value_policy::reference)
		.def("inTriangleSets", &Class::inTriangleSets, py::return_value_policy::reference)
		.def("stateTriangleSets", &Class::stateTriangleSets, py::return_value_policy::reference)
		// protected
		.def("resetStates", &ExtractTriangleSetsPublicist::resetStates);
}

#include "Samplers/PointFromCurve.h"
template <typename TDataType>
void declare_point_from_curve(py::module& m, std::string typestr) {
	using Class = dyno::PointFromCurve<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;

	class PointFromCurveTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PointFromCurve<TDataType>,
				resetStates
			);
		}
	};

	class PointFromCurvePublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("PointFromCurve") + typestr;
	py::class_<Class, Parent, PointFromCurveTrampoline,  std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varUniformScale", &Class::varUniformScale, py::return_value_policy::reference)
		.def("varCurve", &Class::varCurve, py::return_value_policy::reference)
		.def("statePointSet", &Class::statePointSet, py::return_value_policy::reference)
		.def("stateEdgeSet", &Class::stateEdgeSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &PointFromCurvePublicist::resetStates);
}

#include "Samplers/Sampler.h"
template <typename TDataType>
void declare_sampler(py::module& m, std::string typestr) {
	using Class = dyno::Sampler<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("Sampler") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("statePointSet", &Class::statePointSet, py::return_value_policy::reference);
}

#include "Samplers/PointsBehindMesh.h"
template <typename TDataType>
void declare_points_behind_mesh(py::module& m, std::string typestr) {
	using Class = dyno::PointsBehindMesh<TDataType>;
	using Parent = dyno::Sampler<TDataType>;

	class PointsBehindMeshTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PointsBehindMesh<TDataType>,
				resetStates
			);
		}
	};

	class PointsBehindMeshPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("PointsBehindMesh") + typestr;
	py::class_<Class, Parent, PointsBehindMeshTrampoline,  std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varThickness", &Class::varThickness, py::return_value_policy::reference)
		.def("varSamplingDistance", &Class::varSamplingDistance, py::return_value_policy::reference)
		.def("varGeneratingDirection", &Class::varGeneratingDirection, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("statePosition", &Class::statePosition, py::return_value_policy::reference)
		.def("statePlane", &Class::statePlane, py::return_value_policy::reference)
		.def("statePointNormal", &Class::statePointNormal, py::return_value_policy::reference)
		.def("outPointGrowthDirection", &Class::outPointGrowthDirection, py::return_value_policy::reference)
		.def("statePointBelongTriangleIndex", &Class::statePointBelongTriangleIndex, py::return_value_policy::reference)
		.def("outSamplingDistance", &Class::outSamplingDistance, py::return_value_policy::reference)
		// protected
		.def("resetStates", &PointsBehindMeshPublicist::resetStates);
}

#include "Samplers/PoissonPlane.h"
template <typename TDataType>
void declare_poisson_plane(py::module& m, std::string typestr) {
	using Class = dyno::PoissonPlane<TDataType>;
	using Parent = dyno::ComputeModule;

	class PoissonPlanePublicist : public Class
	{
	public:
		using Class::searchGrid; 
		using Class::indexTransform;
		using Class::pointNumberRecommend;
	};

	std::string pyclass_name = std::string("PoissonPlane") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("ConstructGrid", &Class::ConstructGrid)
		.def("collisionJudge", &Class::collisionJudge)
		.def("varSamplingDistance", &Class::varSamplingDistance, py::return_value_policy::reference)
		.def("varUpper", &Class::varUpper, py::return_value_policy::reference)
		.def("varLower", &Class::varLower, py::return_value_policy::reference)
		.def("compute", &Class::compute)
		.def("getPoints", &Class::getPoints)
		// protected
		.def("searchGrid", &PoissonPlanePublicist::searchGrid)
		.def("indexTransform", &PoissonPlanePublicist::indexTransform)
		.def("pointNumberRecommend", &PoissonPlanePublicist::pointNumberRecommend);
}

#include "Samplers/ShapeSampler.h"
template <typename TDataType>
void declare_shape_sampler(py::module& m, std::string typestr) {
	using Class = dyno::ShapeSampler<TDataType>;
	using Parent = dyno::Sampler<TDataType>;

	class ShapeSamplerTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ShapeSampler<TDataType>,
				resetStates
			);
		}
	};

	class ShapeSamplerPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("ShapeSampler") + typestr;
	py::class_<Class, Parent, ShapeSamplerTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEF_VAR
		.def("varSamplingDistance", &Class::varSamplingDistance, py::return_value_policy::reference)
		//DEF_VAR_IN
		.def("importShape", &Class::importShape, py::return_value_policy::reference)
		.def("getShape", &Class::getShape)
		// protected
		.def("resetStates", &ShapeSamplerPublicist::resetStates);
}

#include "CollisionDetector.h"
template <typename TDataType>
void declare_collision_detector(py::module& m, std::string typestr) {
	using Class = dyno::CollisionDetector<TDataType>;
	using Parent = dyno::Node;

	class CollisionDetectorTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::CollisionDetector<TDataType>,
				resetStates
			);
		}

		bool validateInputs() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::CollisionDetector<TDataType>,
				validateInputs
			);
		}
	};

	class CollisionDetectorPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::validateInputs;
	};

	std::string pyclass_name = std::string("CollisionDetector") + typestr;
	py::class_<Class, Parent, CollisionDetectorTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)

		.def("importShapeA", &Class::importShapeA, py::return_value_policy::reference)
		.def("getShapeA", &Class::getShapeA)
		.def("importShapeB", &Class::importShapeB, py::return_value_policy::reference)
		.def("getShapeB", &Class::getShapeB)
		.def("stateContacts", &Class::stateContacts, py::return_value_policy::reference)
		.def("stateNormals", &Class::stateNormals, py::return_value_policy::reference)
		// protected
		.def("resetStates", &CollisionDetectorPublicist::resetStates)
		.def("validateInputs", &CollisionDetectorPublicist::validateInputs);
}

#include "GltfLoader.h"
inline void declare_bounding_box_of_texture_mesh(py::module& m) {
	using Class = dyno::BoundingBoxOfTextureMesh;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("BoundingBoxOfTextureMesh");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varShapeId", &Class::varShapeId, py::return_value_policy::reference)
		.def("varCenter", &Class::varCenter, py::return_value_policy::reference)
		.def("varLowerBound", &Class::varLowerBound, py::return_value_policy::reference)
		.def("varUpperBound", &Class::varUpperBound, py::return_value_policy::reference)
		.def("inTextureMesh", &Class::inTextureMesh, py::return_value_policy::reference)
		.def("outBoundingBox", &Class::outBoundingBox, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_gltf_loader(py::module& m, std::string typestr) {
	using Class = dyno::GltfLoader<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;

	class GltfLoaderTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::GltfLoader<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::GltfLoader<TDataType>,
				updateStates
			);
		}
	};

	class GltfLoaderPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("GltfLoader") + typestr;
	py::class_<Class, Parent, GltfLoaderTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("varFileName", &Class::varFileName, py::return_value_policy::reference)
		.def("varImportAnimation", &Class::varImportAnimation, py::return_value_policy::reference)
		.def("varJointRadius", &Class::varJointRadius, py::return_value_policy::reference)
		.def("stateTexCoord_0", &Class::stateTexCoord_0, py::return_value_policy::reference)
		.def("stateTexCoord_1", &Class::stateTexCoord_1, py::return_value_policy::reference)
		.def("stateInitialMatrix", &Class::stateInitialMatrix, py::return_value_policy::reference)

		.def("stateTransform", &Class::stateTransform, py::return_value_policy::reference)
		.def("stateSkin", &Class::stateSkin, py::return_value_policy::reference)

		.def("stateJointInverseBindMatrix", &Class::stateJointInverseBindMatrix, py::return_value_policy::reference)
		.def("stateJointLocalMatrix", &Class::stateJointLocalMatrix, py::return_value_policy::reference)
		.def("state_jont_world_matrix", &Class::stateJointWorldMatrix, py::return_value_policy::reference)

		.def("stateJointsData", &Class::stateJointsData, py::return_value_policy::reference)
		.def("stateTextureMesh", &Class::stateTextureMesh, py::return_value_policy::reference)

		.def("stateTextureMesh", &Class::stateTextureMesh, py::return_value_policy::reference)
		.def("stateJointSet", &Class::stateJointSet, py::return_value_policy::reference)
		.def("stateAnimation", &Class::stateAnimation, py::return_value_policy::reference)
		// protected
		.def("resetStates", &GltfLoaderPublicist::resetStates)
		.def("updateStates", &GltfLoaderPublicist::updateStates);
}

#include "Group.h"
template <typename TDataType>
void declare_group(py::module& m, std::string typestr) {
	using Class = dyno::Group<TDataType>;
	using Parent = dyno::Node;

	class GroupTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Group<TDataType>,
				resetStates
			);
		}
	};

	class GroupPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::varChanged;
	};

	std::string pyclass_name = std::string("Group") + typestr;
	py::class_<Class, Parent, GroupTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varPointId", &Class::varPointId, py::return_value_policy::reference)
		.def("varEdgeId", &Class::varEdgeId, py::return_value_policy::reference)
		.def("varPrimitiveId", &Class::varPrimitiveId, py::return_value_policy::reference)
		.def("inPointId", &Class::inPointId, py::return_value_policy::reference)
		.def("inEdgeId", &Class::inEdgeId, py::return_value_policy::reference)
		.def("inPrimitiveId", &Class::inPrimitiveId, py::return_value_policy::reference)

		.def("getSelectPrimitives", &Class::getSelectPrimitives)
		.def("getSelectEdges", &Class::getSelectEdges)
		.def("getSelectPoints", &Class::getSelectPoints)
		.def_readwrite("selectedPointID", &Class::selectedPointID)
		.def_readwrite("selectedEdgeID", &Class::selectedEdgeID)
		.def_readwrite("selectedPrimitiveID", &Class::selectedPrimitiveID)
		// protected
		.def("resetStates", &GroupPublicist::resetStates)
		.def("varChanged", &GroupPublicist::varChanged);
}

#include "JointDeform.h"
template <typename TDataType>
void declare_joint_deform(py::module& m, std::string typestr) {
	using Class = dyno::JointDeform<TDataType>;
	using Parent = dyno::ModelEditing<TDataType>;

	class JointDeformTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::JointDeform<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::JointDeform<TDataType>,
				updateStates
			);
		}
	};

	class JointDeformPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("JointDeform") + typestr;
	py::class_<Class, Parent, JointDeformTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inJoint", &Class::inJoint, py::return_value_policy::reference)
		.def("inSkin", &Class::inSkin, py::return_value_policy::reference)
		.def("inAnimation", &Class::inAnimation, py::return_value_policy::reference)
		.def("inInstanceTransform", &Class::inInstanceTransform, py::return_value_policy::reference)
		.def("inTextureMesh", &Class::inTextureMesh, py::return_value_policy::reference)
		// protected
		.def("resetStates", &JointDeformPublicist::resetStates)
		.def("updateStates", &JointDeformPublicist::updateStates);

}

#include "ModelEditing.h"
template <typename TDataType>
void declare_model_editing(py::module& m, std::string typestr) {
	using Class = dyno::ModelEditing<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ModelEditing") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType);
}

#include "NormalVisualization.h"
template <typename TDataType>
void declare_normal_visualization(py::module& m, std::string typestr) {
	using Class = dyno::NormalVisualization<TDataType>;
	using Parent = dyno::Node;

	class NormalVisualizationTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::NormalVisualization<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::NormalVisualization<TDataType>,
				updateStates
			);
		}
	};

	class NormalVisualizationPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("Normal") + typestr;
	py::class_<Class, Parent, NormalVisualizationTrampoline,  std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)

		.def("varLength", &Class::varLength, py::return_value_policy::reference)
		.def("varNormalize", &Class::varNormalize, py::return_value_policy::reference)

		.def("varLineWidth", &Class::varLineWidth, py::return_value_policy::reference)
		.def("varShowWireframe", &Class::varShowWireframe, py::return_value_policy::reference)
		.def("varArrowResolution", &Class::varArrowResolution, py::return_value_policy::reference)

		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("inInNormal", &Class::inInNormal, py::return_value_policy::reference)
		.def("inScalar", &Class::inScalar, py::return_value_policy::reference)

		.def("stateNormalSet", &Class::stateNormalSet, py::return_value_policy::reference)
		.def("stateNormal", &Class::stateNormal, py::return_value_policy::reference)
		.def("stateTriangleCenter", &Class::stateTriangleCenter, py::return_value_policy::reference)

		.def("stateArrowCylinder", &Class::stateArrowCylinder, py::return_value_policy::reference)
		.def("stateArrowCone", &Class::stateArrowCone, py::return_value_policy::reference)
		.def("stateTransformsCylinder", &Class::stateTransformsCylinder, py::return_value_policy::reference)
		.def("stateTransformsCone", &Class::stateTransformsCone, py::return_value_policy::reference)
		// protected
		.def("resetStates", &NormalVisualizationPublicist::resetStates)
		.def("updateStates", &NormalVisualizationPublicist::updateStates);
}

#include "SplineConstraint.h"
template <typename TDataType>
void declare_spline_constraint(py::module& m, std::string typestr) {
	using Class = dyno::SplineConstraint<TDataType>;
	using Parent = dyno::Node;

	class SplineConstraintTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SplineConstraint<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SplineConstraint<TDataType>,
				updateStates
			);
		}
	};

	class SplineConstraintPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
		using Class::updateTransform;
		using Class::SLerp;
		using Class::getQuatFromVector;
		using Class::UpdateCurrentVelocity;
	};

	std::string pyclass_name = std::string("SplineConstraint") + typestr;
	py::class_<Class, Parent, SplineConstraintTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inSpline", &Class::inSpline, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("varVelocity", &Class::varVelocity, py::return_value_policy::reference)
		.def("varOffest", &Class::varOffest, py::return_value_policy::reference)
		.def("varAccelerate", &Class::varAccelerate, py::return_value_policy::reference)
		.def("varAcceleratedSpeed", &Class::varAcceleratedSpeed, py::return_value_policy::reference)
		.def("stateTopology", &Class::stateTopology, py::return_value_policy::reference)
		// protected
		.def("resetStates", &SplineConstraintPublicist::resetStates)
		.def("updateStates", &SplineConstraintPublicist::updateStates)
		.def("updateTransform", &SplineConstraintPublicist::updateTransform)
		.def("SLerp", &SplineConstraintPublicist::SLerp)
		.def("getQuatFromVector", &SplineConstraintPublicist::getQuatFromVector)
		.def("UpdateCurrentVelocity", &SplineConstraintPublicist::UpdateCurrentVelocity);
}

#include "Subdivide.h"
template <typename TDataType>
void declare_subdivide(py::module& m, std::string typestr) {
	using Class = dyno::Subdivide<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;

	class SubdivideTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Subdivide<TDataType>,
				resetStates
			);
		}
	};

	class SubdividePublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("Subdivide") + typestr;
	py::class_<Class, Parent, SubdivideTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("varStep", &Class::varStep, py::return_value_policy::reference)
		.def("inInTriangleSet", &Class::inInTriangleSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &SubdividePublicist::resetStates);
}

#include "VectorVisualNode.h"
template <typename TDataType>
void declare_vector_visual_node(py::module& m, std::string typestr) {
	using Class = dyno::VectorVisualNode<TDataType>;
	using Parent = dyno::Node;

	class VectorVisualNodeTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::VectorVisualNode<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::VectorVisualNode<TDataType>,
				updateStates
			);
		}
	};

	class VectorVisualNodePublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("VectorVisualNode") + typestr;
	py::class_<Class, Parent, VectorVisualNodeTrampoline, std::shared_ptr<Class>>VVN(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	VVN.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)

		.def("varLength", &Class::varLength, py::return_value_policy::reference)
		.def("varNormalize", &Class::varNormalize, py::return_value_policy::reference)

		.def("varLineMode", &Class::varLineMode, py::return_value_policy::reference)
		.def("varLineWidth", &Class::varLineWidth, py::return_value_policy::reference)
		.def("varArrowResolution", &Class::varArrowResolution, py::return_value_policy::reference)
		.def("varDebug", &Class::varDebug, py::return_value_policy::reference)

		.def("inPointSet", &Class::inPointSet, py::return_value_policy::reference)
		.def("in_vector", &Class::inInVector, py::return_value_policy::reference)
		.def("inScalar", &Class::inScalar, py::return_value_policy::reference)

		.def("stateNormalSet", &Class::stateNormalSet, py::return_value_policy::reference)
		.def("stateNormal", &Class::stateNormal, py::return_value_policy::reference)

		.def("stateArrowCylinder", &Class::stateArrowCylinder, py::return_value_policy::reference)
		.def("stateArrowCone", &Class::stateArrowCone, py::return_value_policy::reference)
		.def("stateTransformsCylinder", &Class::stateTransformsCylinder, py::return_value_policy::reference)
		.def("stateTransformsCone", &Class::stateTransformsCone, py::return_value_policy::reference)
		// protected
		.def("resetStates", &VectorVisualNodePublicist::resetStates)
		.def("updateStates", &VectorVisualNodePublicist::updateStates);

	py::enum_<typename Class::LineMode>(VVN, "LineMode")
		.value("Line", Class::LineMode::Line)
		.value("Cylnder", Class::LineMode::Cylnder)
		.value("Arrow", Class::LineMode::Arrow)
		.export_values();
}

void pybind_modeling(py::module& m);
