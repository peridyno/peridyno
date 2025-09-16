#pragma once
#include "PyCommon.h"

#include "Node.h"
#include "FInstance.h"
#include "Field.h"
#include "Module/VisualModule.h"
#include "Module/VisualModule.h"
#include "Module/AnimationPipeline.h"
#include "Module/GraphicsPipeline.h"
#include "Module/MouseInputModule.h"
#include "Module/OutputModule.h"
#include "Module/ConstraintModule.h"

#include "Module/CalculateNorm.h"
#include "Module/ComputeModule.h"
#include "Module/KeyboardInputModule.h"

#include "Topology/PointSet.h"
#include "Topology/TriangleSet.h"
#include "Topology/EdgeSet.h"
#include "Topology/TextureMesh.h"

#include "Module/TopologyMapping.h"
#include "BasicShapes/PlaneModel.h"
#include "BasicShapes/SphereModel.h"
#include "Module/GroupModule.h"

#include "Mapping/DiscreteElementsToTriangleSet.h"

#include "ParticleSystem/ParticleSystem.h"
#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"
#include "ParticleSystem/Emitters/ParticleEmitter.h"

#include "Peridynamics/TriangularSystem.h"

#include "HeightField/CapillaryWave.h"
#include "HeightField/Ocean.h"
#include "HeightField/OceanPatch.h"
#include "HeightField/GranularMedia.h"

#include "RigidBody/RigidBody.h"

#include "SceneGraph.h"
#include "Log.h"

#include "Field/Color.h"
#include "Field/FilePath.h"

#include "Auxiliary/DataSource.h"
#include "Collision/CollisionData.h"

#include "Field/Curve.h"
#include "Field/Ramp.h"

//ScemeGraph->add_node
#include "PointsLoader.h"
#include "GltfLoader.h"
#include "GeometryLoader.h"
#include "ParticleSystem/MakeParticleSystem.h"
#include "ParticleSystem/ParticleFluid.h"
#include "StaticMeshLoader.h"
#include "Multiphysics/VolumeBoundary.h"

#include "Action/ActNodeInfo.h"
#include "Auxiliary/Add.h"
//#include "Auxiliary/DebugInfo.h"
#include "Auxiliary/Divide.h"
#include "Auxiliary/Multiply.h"
#include "Auxiliary/Subtract.h"

#include "Plugin/PluginManager.h"
#include "DirectedAcyclicGraph.h"
#include "NodeFactory.h"
#include "SceneGraphFactory.h"
#include "SceneLoaderFactory.h"
#include "SceneLoaderXML.h"

#include <GLSurfaceVisualModule.h>
#include <ColorMapping.h>
#include <Peridynamics/Bond.h>
#include "ParticleSystem/GhostFluid.h"

using FBase = dyno::FBase;
using OBase = dyno::OBase;
using InstanceBase = dyno::InstanceBase;
using Node = dyno::Node;
using NodePort = dyno::NodePort;
using Module = dyno::Module;
using ComputeModule = dyno::ComputeModule;
using TopologyModule = dyno::TopologyModule;
using Pipeline = dyno::Pipeline;
using GraphicsPipeline = dyno::GraphicsPipeline;
using AnimationPipeline = dyno::AnimationPipeline;
using SceneGraph = dyno::SceneGraph;
using VisualModule = dyno::VisualModule;
using Log = dyno::Log;
//new
using Color = dyno::Color;
using ConstraintModule = dyno::ConstraintModule;
using InputModule = dyno::InputModule;
using MouseInputModule = dyno::MouseInputModule;
using GroupModule = dyno::GroupModule;
using TopologyMappingdyno = dyno::TopologyMapping;
using OutputModule = dyno::OutputModule;
using Object = dyno::Object;
using DataSource = dyno::DataSource;
using CollisionMask = dyno::CollisionMask;
using Vec3f = dyno::Vec3f;
using KeyboardInputModule = dyno::KeyboardInputModule;
using Add = dyno::Add;
//using DebugInfo = dyno::DebugInfo;
//using PrintInt = dyno::PrintInt;
//using PrintVector = dyno::PrintVector;
//using PrintFloat = dyno::PrintFloat;
//using PrintUnsigned = dyno::PrintUnsigned;
using Divide = dyno::Divide;
using Multiply = dyno::Multiply;
using Subtract = dyno::Subtract;

using uint = unsigned int;
using uchar = unsigned char;
using uint64 = unsigned long long;
using int64 = signed long long;


//class NodeTrampoline : public dyno::Node
//{
//public:
//	void resetStates() override { PYBIND11_OVERRIDE(void, dyno::Node, resetStates); }
//};
//
//class NodePublicist : public dyno::Node
//{
//public:
//	using dyno::Node::resetStates;
//};


template<class TNode, class ...Args>
std::shared_ptr<TNode> create_root(SceneGraph& scene, Args&& ... args) {
	return scene.createNewScene<TNode>(std::forward<Args>(args)...);
}

template<typename T>
void declare_var(py::module& m, std::string typestr) {
	using Class = dyno::FVar<T>;
	std::string pyclass_name = std::string("FVar") + typestr;
	py::class_<Class, FBase, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<std::string, std::string, dyno::FieldTypeEnum, OBase*>())
		.def(py::init<T, std::string, std::string, dyno::FieldTypeEnum, OBase*>())
		.def("getTemplateName", &Class::getTemplateName)
		.def("getClassName", &Class::getClassName)
		.def("size", &Class::size)
		.def("setValue", &Class::setValue, py::arg("val"), py::arg("notify") = true)
		.def("getValue", &Class::getValue)
		.def("serialize", &Class::serialize)
		//.def("deserialize", &Class::deserialize)
		.def("isEmpty", &Class::isEmpty)
		//.def("connect", &Class::connect)
		.def("getData", &Class::getData);
	//.def("constDataPtr", &Class::constDataPtr, py::return_value_policy::reference)
	//.def("getDataPtr", &Class::getDataPtr, py::return_value_policy::reference);
}

template<typename T, DeviceType deviceType>
void declare_farray(py::module& m, std::string typestr) {
	using Class = dyno::FArray<T, deviceType>;
	using Parent = FBase;
	typedef T							VarType;
	std::string pyclass_name = std::string("FArray") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<std::string, std::string, dyno::FieldTypeEnum, OBase*>())
		.def("size", &Class::size)
		.def("resize", &Class::resize)
		.def("reset", &Class::reset)
		.def("clear", &Class::clear)
		//.def("assign", py::overload_cast<const T&>(&Class::assign))
		//.def("assign", py::overload_cast<const std::vector<T>&>(&Class::assign))
		//.def("assign", py::overload_cast<const dyno::Array<T, DeviceType::GPU>&>(&Class::assign))
		//.def("assign", py::overload_cast<const dyno::Array<T, DeviceType::CPU>&>(&Class::assign))
		.def("isEmpty", &Class::isEmpty);
}

template<typename T, DeviceType deviceType>
void declare_array_list(py::module& m, std::string typestr) {
	using Class = dyno::FArrayList<T, deviceType>;
	using Parent = FBase;
	std::string pyclass_name = std::string("FArrayList") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getTemplateName", &Class::getTemplateName)
		.def("getClassName", &Class::getClassName)
		.def("getDataPtr", &Class::getDataPtr, py::return_value_policy::reference)
		.def("constDataPtr", &Class::constDataPtr, py::return_value_policy::reference)
		.def("allocate", &Class::allocate)
		//.def("connect", &Class::connect)
		.def("getData", &Class::getData, py::return_value_policy::reference)
		.def("constData", &Class::constData, py::return_value_policy::reference)
		.def("size", &Class::size)
		.def("isEmpty", &Class::isEmpty);
	//²»È«
}

template<typename T>
void declare_instance(py::module& m, std::string typestr) {
	using Class = dyno::FInstance<T>;
	using Parent = InstanceBase;
	std::string pyclass_name = std::string("FInstance") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<std::string, std::string, dyno::FieldTypeEnum, dyno::OBase*>())
		.def("getTemplateName", &Class::getTemplateName)
		.def("getClassName", &Class::getClassName)
		.def("getDataPtr", &Class::getDataPtr)
		.def("constDataPtr", &Class::constDataPtr)
		.def("setDataPtr", &Class::setDataPtr)
		.def("allocate", &Class::allocate)
		.def("isEmpty", &Class::isEmpty)
		.def("connect", &Class::connect)
		.def("disconnect", &Class::disconnect)
		.def("getData", &Class::getData, py::return_value_policy::reference)
		.def("size", &Class::size)
		.def("objectPointer", &Class::objectPointer)
		.def("standardObjectPointer", &Class::standardObjectPointer)
		.def("setObjectPointer", &Class::setObjectPointer)
		.def("canBeConnectedBy", &Class::canBeConnectedBy);
}

template<typename T>
void declare_instances(py::module& m, std::string typestr) {
	using Class = dyno::FInstances<T>;
	using Parent = InstanceBase;
	std::string pyclass_name = std::string("FInstances") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<std::string, std::string, dyno::FieldTypeEnum, dyno::OBase*>())
		.def("getTemplateName", &Class::getTemplateName)
		.def("getClassName", &Class::getClassName)
		.def("constDataPtr", &Class::constDataPtr)
		.def("connect", &Class::connect)
		.def("disconnect", &Class::disconnect)
		.def("isEmpty", &Class::isEmpty)
		.def("size", &Class::size)
		.def("inputPolicy", &Class::inputPolicy)
		.def("setObjectPointer", &Class::setObjectPointer)
		.def("objectPointer", &Class::objectPointer)
		.def("standardObjectPointer", &Class::standardObjectPointer)
		.def("canBeConnectedBy", &Class::canBeConnectedBy);
}

//------------------------- New ------------------------------

template <typename TDataType>
void declare_multi_node_port(py::module& m, std::string typestr) {
	using Class = dyno::MultipleNodePort<TDataType>;
	using Parent = dyno::NodePort;
	std::string pyclass_name = std::string("MultipleNodePort") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("clear", &Class::clear)
		.def("addDerivedNode", &Class::addDerivedNode)
		.def("removeDerivedNode", &Class::removeDerivedNode)
		.def("isKindOf", &Class::isKindOf)
		.def("hasNode", &Class::hasNode)
		.def("getNodes", &Class::getNodes, py::return_value_policy::reference)
		.def("getDerivedNodes", &Class::getDerivedNodes, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_single_node_port(py::module& m, std::string typestr) {
	using Class = dyno::SingleNodePort<TDataType>;
	using Parent = dyno::NodePort;
	std::string pyclass_name = std::string("SingleNodePorParametricModelt_") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("isKindOf", &Class::isKindOf)
		.def("hasNode", &Class::hasNode)
		.def("getNodes", &Class::getNodes, py::return_value_policy::reference)
		.def("getDerivedNode", &Class::getDerivedNode, py::return_value_policy::reference)
		.def("setDerivedNode", &Class::setDerivedNode);
}

template <typename TDataType>
void declare_parametric_model(py::module& m, std::string typestr) {
	using Class = dyno::ParametricModel<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ParametricModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr(), py::multiple_inheritance())
		.def(py::init<>())
		.def("computeQuaternion", &Class::computeQuaternion)
		.def("varLocation", &Class::varLocation, py::return_value_policy::reference)
		.def("varRotation", &Class::varRotation, py::return_value_policy::reference)
		.def("varScale", &Class::varScale, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_floating_number(py::module& m, std::string typestr) {
	using Class = dyno::FloatingNumber<TDataType>;
	using Parent = dyno::DataSource;
	std::string pyclass_name = std::string("FloatingNumber") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varValue", &Class::varValue, py::return_value_policy::reference)
		.def("outFloating", &Class::outFloating, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_vector_3_source(py::module& m, std::string typestr) {
	using Class = dyno::Vector3Source<TDataType>;
	using Parent = dyno::DataSource;
	std::string pyclass_name = std::string("Vector3Source") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varValue", &Class::varValue, py::return_value_policy::reference)
		.def("outCoord", &Class::outCoord, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_add_real_and_real(py::module& m, std::string typestr) {
	using Class = dyno::AddRealAndReal<TDataType>;
	using Parent = dyno::Add;
	std::string pyclass_name = std::string("AddRealAndReal") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inA", &Class::inA, py::return_value_policy::reference)
		.def("inB", &Class::inB, py::return_value_policy::reference)
		.def("outO", &Class::outO, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_divide_real_and_real(py::module& m, std::string typestr) {
	using Class = dyno::DivideRealAndReal<TDataType>;
	using Parent = dyno::Divide;
	std::string pyclass_name = std::string("DivideRealAndReal") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inA", &Class::inA, py::return_value_policy::reference)
		.def("inB", &Class::inB, py::return_value_policy::reference)
		.def("outO", &Class::outO, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_multiply_real_and_real(py::module& m, std::string typestr) {
	using Class = dyno::MultiplyRealAndReal<TDataType>;
	using Parent = dyno::Multiply;
	std::string pyclass_name = std::string("MultiplyRealAndReal") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inA", &Class::inA, py::return_value_policy::reference)
		.def("inB", &Class::inB, py::return_value_policy::reference)
		.def("outO", &Class::outO, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_subtract_real_and_real(py::module& m, std::string typestr) {
	using Class = dyno::SubtractRealAndReal<TDataType>;
	using Parent = dyno::Subtract;
	std::string pyclass_name = std::string("SubtractRealAndReal") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inA", &Class::inA, py::return_value_policy::reference)
		.def("inB", &Class::inB, py::return_value_policy::reference)
		.def("outO", &Class::outO, py::return_value_policy::reference);
}

//Init_static_plugin  - for example_3 WaterPouring
#include "initializeModeling.h"
#include "ParticleSystem/initializeParticleSystem.h"
#include "SemiAnalyticalScheme/initializeSemiAnalyticalScheme.h"

//------------------------- NEW END ------------------------------

void declare_camera(py::module& m);

void pybind_log(py::module& m);

void pybind_framework(py::module& m);
