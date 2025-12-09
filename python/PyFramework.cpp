 #include "PyFramework.h"

#include "RigidBody/RigidBodySystem.h"
#include "HeightField/Vessel.h"

#include "Camera.h"
void declare_camera(py::module& m)
{
	using Class = dyno::Camera;
	std::string pyclass_name = std::string("Camera");
	py::class_<Class, std::shared_ptr<Class>>camera(m, pyclass_name.c_str());
	camera.def("getViewMat", &Class::getViewMat) // °ó¶¨ getViewMat ·½·¨
		.def("getProjMat", &Class::getProjMat) // °ó¶¨ getProjMat ·½·¨

		.def("rotateToPoint", &Class::rotateToPoint) // °ó¶¨ rotateToPoint ·½·¨
		.def("translateToPoint", &Class::translateToPoint) // °ó¶¨ translateToPoint ·½·¨
		.def("zoom", &Class::zoom) // °ó¶¨ zoom ·½·¨

		.def("registerPoint", &Class::registerPoint) // °ó¶¨ registerPoint ·½·¨

		.def("setWidth", &Class::setWidth) // °ó¶¨ setWidth ·½·¨
		.def("setHeight", &Class::setHeight) // °ó¶¨ setHeight ·½·¨

		.def("setClipNear", &Class::setClipNear) // °ó¶¨ setClipNear ·½·¨
		.def("setClipFar", &Class::setClipFar) // °ó¶¨ setClipFar ·½·¨

		.def("viewportWidth", &Class::viewportWidth) // °ó¶¨ viewportWidth ·½·¨
		.def("viewportHeight", &Class::viewportHeight) // °ó¶¨ viewportHeight ·½·¨

		.def("clipNear", &Class::clipNear) // °ó¶¨ clipNear ·½·¨
		.def("clipFar", &Class::clipFar) // °ó¶¨ clipFar ·½·¨

		.def("setEyePos", &Class::setEyePos) // °ó¶¨ setEyePos ·½·¨
		.def("setTargetPos", &Class::setTargetPos) // °ó¶¨ setTargetPos ·½·¨

		.def("getEyePos", &Class::getEyePos) // °ó¶¨ getEyePos ·½·¨
		.def("getTargetPos", &Class::getTargetPos) // °ó¶¨ getTargetPos ·½·¨

		.def("castRayInWorldSpace", &Class::castRayInWorldSpace) // °ó¶¨ castRayInWorldSpace ·½·¨

		.def("setUnitScale", &Class::setUnitScale) // °ó¶¨ setUnitScale ·½·¨
		.def("unitScale", &Class::unitScale) // °ó¶¨ unitScale ·½·¨
		.def("setProjectionType", &Class::setProjectionType) // °ó¶¨ setProjectionType ·½·¨
		.def("projectionType", &Class::projectionType); // °ó¶¨ projectionType ·½·¨;

	py::enum_<Class::ProjectionType>(camera, "ProjectionType")
		.value("Perspective", Class::ProjectionType::Perspective)
		.value("Orthogonal", Class::ProjectionType::Orthogonal);
}

#include "DeclareEnum.h"
void declare_p_enum(py::module& m)
{
	using Class = dyno::PEnum;
	std::string pyclass_name = std::string("PEnum");
	py::class_<Class, std::shared_ptr<Class>>camera(m, pyclass_name.c_str());
	camera.def(py::init<>())
		.def(py::init<std::string, int, const std::string>())
		.def("operator==", &Class::operator==)
		.def("operator!=", &Class::operator!=)
		.def("currentKey", &Class::currentKey)
		.def("currentString", &Class::currentString)
		.def("setCurrentKey", &Class::setCurrentKey)
		.def("enumMap", &Class::enumMap);
}

#include "Action/ActNodeInfo.h"
void declare_act_node_info(py::module& m)
{
	using Class = dyno::NodeInfoAct;
	using Parent = dyno::Action;
	std::string pyclass_name = std::string("NodeInfoAct");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str())
		.def(py::init<>());
}

#include "Action/ActPostProcessing.h"
void declare_post_processing(py::module& m)
{
	using Class = dyno::PostProcessing;
	using Parent = dyno::Action;
	std::string pyclass_name = std::string("PostProcessing");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str())
		.def(py::init<>());
}

#include "Action/ActReset.h"
using namespace dyno;
void declare_reset_act(py::module& m)
{
	using Class = dyno::ResetAct;
	using Parent = dyno::Action;
	std::string pyclass_name = std::string("ResetAct");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str())
		.def(py::init<>());
}

#include "Field/VehicleInfo.h"
void declare_vehicle_rigid_body_info(py::module& m) {
	using Class = dyno::VehicleRigidBodyInfo;
	std::string pyclass_name = std::string("VehicleRigidBodyInfo");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<dyno::Name_Shape, int, dyno::ConfigShapeType, Real>())
		.def(py::init<dyno::Name_Shape, int, dyno::ConfigShapeType, dyno::Transform3f, Real>());
}

void declare_vehicle_joint_info(py::module& m) {
	using Class = dyno::VehicleJointInfo;
	std::string pyclass_name = std::string("VehicleJointInfo");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
	//.def(py::init<dyno::Name_Shape, dyno::Name_Shape, dyno::ConfigShapeType, dyno::Vector<Real, 3>, dyno::Vector<Real, 3>, bool, Real, bool, Real, Real>());
}


void declare_vehicle_bind(py::module& m) {
	using Class = dyno::VehicleBind;
	std::string pyclass_name = std::string("VehicleBind");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<int>())
		.def("isValid", &Class::isValid, py::return_value_policy::reference)
		.def_readwrite("mVehicleRigidBodyInfo", &Class::mVehicleRigidBodyInfo)
		.def_readwrite("mVehicleJointInfo", &Class::mVehicleJointInfo);
}

void declare_animation_2_joint_config(py::module& m) {
	using Class = dyno::Animation2JointConfig;
	std::string pyclass_name = std::string("Animation2JointConfig");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<std::string, int, uint>())
		.def(py::init<std::string, int, uint, float>())
		.def_readwrite("JointName", &Class::JointName)
		.def_readwrite("JointId", &Class::JointId)
		.def_readwrite("Axis", &Class::Axis)
		.def_readwrite("Intensity", &Class::Intensity);
}

void declare_hinge_action(py::module& m) {
	using Class = dyno::HingeAction;
	std::string pyclass_name = std::string("HingeAction");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<int, float>())
		.def_readwrite("joint", &Class::joint)
		.def_readwrite("value", &Class::value);
}

void declare_key_2_hinge_config(py::module& m) {
	using Class = dyno::Key2HingeConfig;
	std::string pyclass_name = std::string("Key2HingeConfig");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("addMap", &Class::addMap)
		.def_readwrite("key2Hinge", &Class::key2Hinge);
}

void pybind_log(py::module& m) {}

void pybind_framework(py::module& m)
{
	pybind_log(m);

	//py::class_<dyno::ClassInfo, std::shared_ptr<dyno::ClassInfo>>(m, "ClassInfo")
	//	.def("create_object", &dyno::ClassInfo::createObject, py::return_value_policy::reference)
	//	.def("isDynamic", &dyno::ClassInfo::isDynamic)
	//	.def("getClassName", &dyno::ClassInfo::getClassName)
	//	.def("getConstructor", &dyno::ClassInfo::getConstructor);

	//basic
	py::class_<Object, std::shared_ptr<Object>>(m, "Object")
		.def(py::init<>())
		.def("registerClass", &Object::registerClass)
		//.def("createObject", &Object::createObject, py::return_value_policy::reference)
		.def("getClassMap", &Object::getClassMap, py::return_value_policy::reference)
		.def("baseId", &Object::baseId)
		.def("objectId", &Object::objectId);

	//OBase
	typedef std::map<dyno::FieldID, FBase*> FieldMap;
	py::class_<OBase, Object, std::shared_ptr<OBase>>(m, "OBase")
		.def(py::init<>())
		.def("caption", &OBase::caption)
		.def("captionVisible", &OBase::captionVisible)
		.def("description", &OBase::description)
		.def("getName", &OBase::getName)
		.def("addField", py::overload_cast<dyno::FBase*>(&OBase::addField))
		.def("addField", py::overload_cast<dyno::FieldID, dyno::FBase*>(&OBase::addField))
		.def("addFieldAlias", py::overload_cast<dyno::FieldID, dyno::FBase*>(&OBase::addFieldAlias))
		.def("addFieldAlias", py::overload_cast<dyno::FieldID, dyno::FBase*, FieldMap&>(&OBase::addFieldAlias))
		.def("findField", &OBase::findField)
		.def("findFieldAlias", py::overload_cast<dyno::FieldID, FieldMap&>(&OBase::findFieldAlias))
		.def("findFieldAlias", py::overload_cast<dyno::FieldID>(&OBase::findFieldAlias))
		.def("removeField", &OBase::removeField)
		.def("removeFieldAlias", py::overload_cast<dyno::FieldID>(&OBase::removeFieldAlias))
		.def("removeFieldAlias", py::overload_cast<dyno::FieldID, FieldMap&>(&OBase::removeFieldAlias))
		//.def("get_field", py::overload_cast<dyno::FieldID>(&OBase::getField), py::return_value_policy::reference)
		.def("getAllFields", &OBase::getAllFields, py::return_value_policy::reference)
		.def("attachField", &OBase::attachField)
		.def("isAllFieldsReady", &OBase::isAllFieldsReady)
		.def("getFieldAlias", &OBase::getFieldAlias)
		.def("getFieldAliasCount", &OBase::getFieldAliasCount)
		.def("setBlockCoord", &OBase::setBlockCoord)
		.def("bx", &OBase::bx)
		.def("by", &OBase::by)
		.def("findInputField", &OBase::findInputField)
		.def("addInputField", &OBase::addInputField)
		.def("removeInputField", &OBase::removeInputField)
		.def("getInputFields", &OBase::getInputFields, py::return_value_policy::reference)
		.def("findOutputField", &OBase::findOutputField)
		.def("addOutputField", &OBase::addOutputField)
		.def("addToOutput", &OBase::addToOutput)
		.def("removeOutputField", &OBase::removeOutputField)
		.def("removeFromOutput", &OBase::removeFromOutput)
		.def("getOutputFields", &OBase::getOutputFields, py::return_value_policy::reference)
		.def("findParameter", &OBase::findParameter)
		.def("addParameter", &OBase::addParameter)
		.def("removeParameter", &OBase::removeParameter)
		.def("getParameters", &OBase::getParameters, py::return_value_policy::reference);


	class NodePublicist : public Node
	{
	public:
		using Node::appendExportNode;
		using Node::removeExportNode;
		using Node::preUpdateStates;
		using Node::updateStates;
		using Node::postUpdateStates;
		using Node::updateTopology;
		using Node::resetStates;
		using Node::validateInputs;
		using Node::requireUpdate;
		using Node::tick;
	};

	py::class_<Node, OBase, std::shared_ptr<Node>>(m, "Node", py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("setName", &Node::setName)
		.def("getName", &Node::getName)
		.def("getNodeType", &Node::getNodeType)
		.def("isAutoSync", &Node::isAutoSync)
		.def("isAutoHidden", &Node::isAutoHidden)
		.def("setAutoSync", &Node::setAutoSync)
		.def("setAutoHidden", &Node::setAutoHidden)
		.def("isActive", &Node::isActive)
		.def("setActive", &Node::setActive)
		.def("isVisible", &Node::isVisible)
		.def("setVisible", &Node::setVisible)
		.def("getDt", &Node::getDt)
		.def("setDt", &Node::setDt)
		.def("setSceneGraph", &Node::setSceneGraph)
		.def("getSceneGraph", &Node::getSceneGraph, py::return_value_policy::reference)
		.def("getImportNodes", &Node::getImportNodes)
		.def("getExportNodes", &Node::getExportNodes)
		.def("addModule", static_cast<bool(Node::*)(std::shared_ptr<Module>)>(&Node::addModule))
		.def("deleteModule", static_cast<bool(Node::*)(std::shared_ptr<Module>)>(&Node::deleteModule))
		.def("getModuleList", &Node::getModuleList)
		.def("hasModule", &Node::hasModule)
		.def("getModule", static_cast<std::shared_ptr<Module>(Node::*)(std::string)> (&Node::getModule))
		.def("resetPipeline", &Node::resetPipeline)
		.def("graphicsPipeline", &Node::graphicsPipeline)
		.def("animationPipeline", &Node::animationPipeline)
		.def("update", &Node::update)
		.def("updateGraphicsContext", &Node::updateGraphicsContext)
		.def("reset", &Node::reset)
		.def("boundingBox", &Node::boundingBox)
		.def("connect", &Node::connect)
		.def("disconnect", &Node::disconnect)
		.def("attachField", &Node::attachField)
		.def("getAllNodePorts", &Node::getAllNodePorts)
		.def("sizeOfNodePorts", &Node::sizeOfNodePorts)
		.def("sizeOfImportNodes", &Node::sizeOfImportNodes)
		.def("sizeOfExportNodes", &Node::sizeOfExportNodes)
		.def("stateElapsedTime", &Node::stateElapsedTime, py::return_value_policy::reference)
		.def("stateTimeStep", &Node::stateTimeStep, py::return_value_policy::reference)
		.def("stateFrameNumber", &Node::stateFrameNumber, py::return_value_policy::reference)
		.def("setForceUpdate", &Node::setForceUpdate, py::return_value_policy::reference)
		// protected
		.def("appendExportNode", &NodePublicist::appendExportNode, py::return_value_policy::reference)
		.def("removeExportNode", &NodePublicist::removeExportNode, py::return_value_policy::reference)
		.def("preUpdateStates", &NodePublicist::preUpdateStates, py::return_value_policy::reference)
		.def("updateStates", &NodePublicist::updateStates, py::return_value_policy::reference)
		.def("postUpdateStates", &NodePublicist::postUpdateStates, py::return_value_policy::reference)
		.def("updateTopology", &NodePublicist::updateTopology, py::return_value_policy::reference)
		.def("resetStates", &NodePublicist::resetStates, py::return_value_policy::reference)
		.def("validateInputs", &NodePublicist::validateInputs, py::return_value_policy::reference)
		.def("requireUpdate", &NodePublicist::requireUpdate, py::return_value_policy::reference)
		.def("tick", &NodePublicist::tick, py::return_value_policy::reference);
	//.def("resetStates", &NodePublicist::resetStates);

	py::class_<dyno::NBoundingBox>(m, "NBoundingBox")
		.def(py::init<>())
		.def(py::init<dyno::Vec3f, dyno::Vec3f>())
		.def_readwrite("lower", &dyno::NBoundingBox::lower)
		.def_readwrite("upper", &dyno::NBoundingBox::upper)
		.def("join", &dyno::NBoundingBox::join, py::return_value_policy::reference)
		.def("intersect", &dyno::NBoundingBox::intersect, py::return_value_policy::reference)
		.def("maxLength", &dyno::NBoundingBox::maxLength);

	py::class_<dyno::NodeAction, std::shared_ptr<dyno::NodeAction>>(m, "NodeAction")
		.def("icon", &dyno::NodeAction::icon)
		.def("caption", &dyno::NodeAction::caption)
		.def("action", &dyno::NodeAction::action);

	py::class_<dyno::NodeGroup, std::shared_ptr<dyno::NodeGroup>>(m, "NodeGroup")
		//.def("add_action", &dyno::NodeAction::addAction)
		.def("actions", &dyno::NodeGroup::actions, py::return_value_policy::reference)
		.def("caption", &dyno::NodeGroup::caption);

	py::class_<dyno::NodePage, std::shared_ptr<dyno::NodePage>>(m, "NodePage")
		.def("addGroup", &dyno::NodePage::addGroup)
		.def("hasGroup", &dyno::NodePage::hasGroup)
		.def("groups", &dyno::NodePage::groups, py::return_value_policy::reference)
		.def("icon", &dyno::NodePage::icon)
		.def("caption", &dyno::NodePage::caption);

	py::class_<dyno::NodeIterator, std::shared_ptr<dyno::NodeIterator>>(m, "NodeIterator")
		.def(py::init<>())
		.def("get", &dyno::NodeIterator::get);


	//py::class_<dyno::NodeFactory, std::shared_ptr<dyno::NodeFactory>>(m, "NodeFactory")
	//	.def(py::init<>())
	//	.def("instance", &dyno::NodeFactory::instance)
	//	.def("addPage", &dyno::NodeFactory::addPage)
	//	.def("hasPage", &dyno::NodeFactory::hasPage)
	//	.def("nodePages", &dyno::NodeFactory::nodePages)
	//	.def("nodeContentActions", &dyno::NodeFactory::nodeContentActions)
	//	.def("addContentAction", &dyno::NodeFactory::addContentAction);

	py::class_<dyno::Action, std::shared_ptr<dyno::Action>>(m, "Action")
		.def(py::init<>())
		.def("start", &dyno::Action::start)
		.def("process", &dyno::Action::process)
		.def("end", &dyno::Action::end);

	py::class_<dyno::TimeStamp, std::shared_ptr<dyno::TimeStamp>>(m, "TimeStamp")
		.def(py::init<>())
		.def("mark", &dyno::TimeStamp::mark)
		.def("__gt__", [](dyno::TimeStamp& self, dyno::TimeStamp& ts) -> bool {
		return self > ts;
			})
		.def("__st__", [](dyno::TimeStamp& self, dyno::TimeStamp& ts) -> bool {
				return self < ts;
			});


	py::class_<dyno::DirectedAcyclicGraph, std::shared_ptr<dyno::DirectedAcyclicGraph>>(m, "DirectedAcyclicGraph")
		.def(py::init<>())
		.def("addEdge", &dyno::DirectedAcyclicGraph::addEdge)
		//.def("add_edge", &dyno::DirectedAcyclicGraph::topologicalSort)
		//.def("add_edge", &dyno::DirectedAcyclicGraph::topologicalSortUtil)
		.def("sizeOfVertex", &dyno::DirectedAcyclicGraph::sizeOfVertex)
		.def("OtherVerticesSize", &dyno::DirectedAcyclicGraph::OtherVerticesSize)
		.def("getOtherVertices", &dyno::DirectedAcyclicGraph::getOtherVertices)
		.def("vertices", &dyno::DirectedAcyclicGraph::vertices)
		.def("edges", &dyno::DirectedAcyclicGraph::edges, py::return_value_policy::reference)
		.def("reverseEdges", &dyno::DirectedAcyclicGraph::reverseEdges, py::return_value_policy::reference)
		.def("addOtherVertices", &dyno::DirectedAcyclicGraph::addOtherVertices)
		.def("addtoRemoveList", &dyno::DirectedAcyclicGraph::addtoRemoveList)
		.def("removeID", &dyno::DirectedAcyclicGraph::removeID);

	py::class_<dyno::Plugin>(m, "Plugin")
		.def(py::init<>())
		.def("getInfo", &dyno::Plugin::getInfo, py::return_value_policy::reference)
		.def("isLoaded", &dyno::Plugin::isLoaded)
		.def("unload", &dyno::Plugin::unload)
		.def("load", &dyno::Plugin::load);

	py::class_<dyno::PluginManager>(m, "PluginManager")
		.def("instance", &dyno::PluginManager::instance, py::return_value_policy::reference)
		.def("getExtension", &dyno::PluginManager::getExtension)
		.def("loadPlugin", &dyno::PluginManager::loadPlugin)
		.def("loadPluginByPath", &dyno::PluginManager::loadPluginByPath)
		.def("get_plugin", &dyno::PluginManager::getPlugin);

	class PluginEntryPublicist : public dyno::PluginEntry
	{
	public:
		using PluginEntry::initializeNodeCreators;
		using PluginEntry::initializeActions;
	};

	py::class_<dyno::PluginEntry, std::shared_ptr<dyno::PluginEntry >>(m, "PluginEntry")
		.def(py::init<>())
		.def("name", &dyno::PluginEntry::name)
		.def("version", &dyno::PluginEntry::version)
		.def("description", &dyno::PluginEntry::description)
		.def("setName", &dyno::PluginEntry::setName, py::arg("pluginName"))
		.def("setVersion", &dyno::PluginEntry::setVersion, py::arg("pluginVersion"))
		.def("setDescription", &dyno::PluginEntry::setDescription, py::arg("desc"))
		.def("initialize", &dyno::PluginEntry::initialize)
		// protected
		.def("initializeNodeCreators", &PluginEntryPublicist::initializeNodeCreators)
		.def("initializeActions", &PluginEntryPublicist::initializeActions);

	py::enum_<typename dyno::NodePortType>(m, "NodePortType")
		.value("Single", dyno::NodePortType::Single)
		.value("Multiple", dyno::NodePortType::Multiple)
		.value("Unknown", dyno::NodePortType::Unknown);

	class NodePortPublicist : public NodePort
	{
	public:
		using NodePort::addNode;
		using NodePort::removeNode;
		using NodePort::notify;
	};

	py::class_<NodePort, std::shared_ptr<dyno::NodePort>>(m, "NodePort")
		.def("getPortName", &NodePort::getPortName)
		.def("getPortType", &NodePort::getPortType)
		.def("setPortType", &NodePort::setPortType)
		.def("getNodes", &NodePort::getNodes)
		.def("isKindOf", &NodePort::isKindOf)
		.def("hasNode", &NodePort::hasNode)
		.def("getParent", &NodePort::getParent, py::return_value_policy::reference)
		.def("attach", &NodePort::attach)
		// protected
		.def("addNode", &NodePortPublicist::addNode)
		.def("removeNode", &NodePortPublicist::removeNode)
		.def("notify", &NodePortPublicist::notify);

	class ModulePortPublicist : public ModulePort
	{
	public:
		using ModulePort::addModule;
		using ModulePort::removeModule;
		using ModulePort::notify;
	};

	py::class_<ModulePort, std::shared_ptr<dyno::ModulePort>>(m, "ModulePort")
		.def("getPortName", &ModulePort::getPortName)
		.def("getPortType", &ModulePort::getPortType)
		.def("setPortType", &ModulePort::setPortType)
		.def("getModules", &ModulePort::getModules)
		.def("isKindOf", &ModulePort::isKindOf)
		.def("hasModule", &ModulePort::hasModule)
		.def("getParent", &ModulePort::getParent, py::return_value_policy::reference)
		.def("clear", &ModulePort::clear)
		.def("attach", &ModulePort::attach)
		// protected
		.def("addModule", &ModulePortPublicist::addModule)
		.def("removeModule", &ModulePortPublicist::removeModule)
		.def("notify", &ModulePortPublicist::notify);

	py::enum_<typename dyno::FieldTypeEnum>(m, "FieldTypeEnum")
		.value("In", dyno::FieldTypeEnum::In)
		.value("Out", dyno::FieldTypeEnum::Out)
		.value("IO", dyno::FieldTypeEnum::IO)
		.value("Param", dyno::FieldTypeEnum::Param)
		.value("State", dyno::FieldTypeEnum::State)
		.value("Next", dyno::FieldTypeEnum::Next);


	class FBasePublicist : public FBase
	{
	public:
		using FBase::setSource;
		using FBase::addSink;
		using FBase::removeSink;
		using FBase::addSource;
		using FBase::removeSource;
		using FBase::connectField;
		using FBase::disconnectField;
	};

	py::class_<FBase, std::shared_ptr<FBase>>(m, "FBase")
		.def("getTemplateName", &FBase::getTemplateName)
		.def("getClassName", &FBase::getClassName)
		.def("getObjectName", &FBase::getObjectName)
		.def("getDescription", &FBase::getDescription)
		.def("getDeviceType", &FBase::getDeviceType)
		.def("setObjectName", &FBase::setObjectName)
		.def("setParent", &FBase::setParent)
		.def("isDerived", &FBase::isDerived)
		.def("isAutoDestroyable", &FBase::isAutoDestroyable)
		.def("setAutoDestroy", &FBase::setAutoDestroy)
		.def("setDerived", &FBase::setDerived)
		.def("isModified", &FBase::isModified)
		.def("tick", &FBase::tick)
		.def("tack", &FBase::tack)
		.def("isOptional", &FBase::isOptional)
		.def("tagOptional", &FBase::tagOptional)
		.def("getMin", &FBase::getMin)
		.def("setMin", &FBase::setMin)
		.def("getMax", &FBase::getMax)
		.def("setMax", &FBase::setMax)
		.def("setRange", &FBase::setRange)
		.def("connect", &FBase::connect)
		.def("disconnect", &FBase::disconnect)
		.def("serialize", &FBase::serialize)
		//.def("deserialize", &FBase::deserialize)
		.def("getTopField", &FBase::getTopField)
		.def("getSource", &FBase::getSource)
		.def("promoteOuput", &FBase::promoteOuput)
		.def("promoteInput", &FBase::promoteInput)
		.def("demoteOuput", &FBase::demoteOuput)
		.def("demoteInput", &FBase::demoteInput)
		.def("isEmpty", &FBase::isEmpty)
		.def("update", &FBase::update)
		.def("attach", &FBase::attach)
		.def("detach", &FBase::detach)
		// protected
		.def("setSource", &FBasePublicist::setSource)
		.def("addSink", &FBasePublicist::addSink)
		.def("removeSink", &FBasePublicist::removeSink)
		.def("addSource", &FBasePublicist::addSource)
		.def("removeSource", &FBasePublicist::removeSource)
		.def("connectField", &FBasePublicist::connectField)
		.def("disconnectField", &FBasePublicist::disconnectField);

	py::class_<dyno::FCallBackFunc, std::shared_ptr<dyno::FCallBackFunc>>(m, "FCallBackFunc")
		.def("update", &dyno::FCallBackFunc::update)
		.def("addInput", &dyno::FCallBackFunc::addInput);

	py::class_<dyno::Canvas::Coord2D>(m, "CanvasCoord2D")
		.def(py::init<>())
		.def(py::init<double, double>())
		.def(py::init<double, double, double>())
		.def(py::init<dyno::Vector<float, 2>>())
		.def("set", &dyno::Canvas::Coord2D::set);

	py::class_<dyno::Canvas::EndPoint>(m, "CanvasEndPoint")
		.def(py::init<>())
		.def(py::init<int, int>());

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

	py::class_<Color>(m, "Color")
		.def(py::init<>())
		.def(py::init<float>())
		.def(py::init<float, float, float>())
		.def("HSVtoRGB", &dyno::Color::HSVtoRGB)
		.def("Snow", &dyno::Color::Snow)
		.def("GhostWhite", &dyno::Color::GhostWhite)
		.def("WhiteSmoke", &dyno::Color::WhiteSmoke)
		.def("Gainsboro", &dyno::Color::Gainsboro)
		.def("FloralWhite", &dyno::Color::FloralWhite)
		.def("OldLace", &dyno::Color::OldLace)
		.def("Linen", &dyno::Color::Linen)
		.def("AntiqueWhite", &dyno::Color::AntiqueWhite)
		.def("PapayaWhip", &dyno::Color::PapayaWhip)
		.def("BlanchedAlmond", &dyno::Color::BlanchedAlmond)
		.def("Bisque", &dyno::Color::Bisque)
		.def("PeachPuff", &dyno::Color::PeachPuff)
		.def("NavajoWhite", &dyno::Color::NavajoWhite)
		.def("Moccasin", &dyno::Color::Moccasin)
		.def("Cornsilk", &dyno::Color::Cornsilk)
		.def("Ivory", &dyno::Color::Ivory)
		.def("LemonChiffon", &dyno::Color::LemonChiffon)
		.def("Seashell", &dyno::Color::Seashell)
		.def("Honeydew", &dyno::Color::Honeydew)
		.def("MintCream", &dyno::Color::MintCream)
		.def("Azure", &dyno::Color::Azure)
		.def("AliceBlue", &dyno::Color::AliceBlue)
		.def("lavender", &dyno::Color::lavender)
		.def("LavenderBlush", &dyno::Color::LavenderBlush)
		.def("MistyRose", &dyno::Color::MistyRose)
		.def("White", &dyno::Color::White)
		.def("Black", &dyno::Color::Black)
		.def("DarkSlateGray", &dyno::Color::DarkSlateGray)
		.def("DimGrey", &dyno::Color::DimGrey)
		.def("SlateGrey", &dyno::Color::SlateGrey)
		.def("LightSlateGray", &dyno::Color::LightSlateGray)
		.def("Grey", &dyno::Color::Grey)
		.def("LightGray", &dyno::Color::LightGray)
		.def("MidnightBlue", &dyno::Color::MidnightBlue)
		.def("NavyBlue", &dyno::Color::NavyBlue)
		.def("CornflowerBlue", &dyno::Color::CornflowerBlue)
		.def("DarkSlateBlue", &dyno::Color::DarkSlateBlue)
		.def("SlateBlue", &dyno::Color::SlateBlue)
		.def("MediumSlateBlue", &dyno::Color::MediumSlateBlue)
		.def("LightSlateBlue", &dyno::Color::LightSlateBlue)
		.def("MediumBlue", &dyno::Color::MediumBlue)
		.def("RoyalBlue", &dyno::Color::RoyalBlue)
		.def("Blue", &dyno::Color::Blue)
		.def("DodgerBlue", &dyno::Color::DodgerBlue)
		.def("DeepSkyBlue", &dyno::Color::DeepSkyBlue)
		.def("SkyBlue", &dyno::Color::SkyBlue)
		.def("LightSkyBlue", &dyno::Color::LightSkyBlue)
		.def("SteelBlue", &dyno::Color::SteelBlue)
		.def("LightSteelBlue", &dyno::Color::LightSteelBlue)
		.def("LightBlue", &dyno::Color::LightBlue)
		.def("PowderBlue", &dyno::Color::PowderBlue)
		.def("PaleTurquoise", &dyno::Color::PaleTurquoise)
		.def("DarkTurquoise", &dyno::Color::DarkTurquoise)
		.def("MediumTurquoise", &dyno::Color::MediumTurquoise)
		.def("Turquoise", &dyno::Color::Turquoise)
		.def("Cyan", &dyno::Color::Cyan)
		.def("LightCyan", &dyno::Color::LightCyan)
		.def("CadetBlue", &dyno::Color::CadetBlue)
		.def("MediumAquamarine", &dyno::Color::MediumAquamarine)
		.def("Aquamarine", &dyno::Color::Aquamarine)
		.def("DarkGreen", &dyno::Color::DarkGreen)
		.def("DarkOliveGreen", &dyno::Color::DarkOliveGreen)
		.def("DarkSeaGreen", &dyno::Color::DarkSeaGreen)
		.def("SeaGreen", &dyno::Color::SeaGreen)
		.def("MediumSeaGreen", &dyno::Color::MediumSeaGreen)
		.def("LightSeaGreen", &dyno::Color::LightSeaGreen)
		.def("PaleGreen", &dyno::Color::PaleGreen)
		.def("SpringGreen", &dyno::Color::SpringGreen)
		.def("LawnGreen", &dyno::Color::LawnGreen)
		.def("Green", &dyno::Color::Green)
		.def("Chartreuse", &dyno::Color::Chartreuse)
		.def("MedSpringGreen", &dyno::Color::MedSpringGreen)
		.def("GreenYellow", &dyno::Color::GreenYellow)
		.def("LimeGreen", &dyno::Color::LimeGreen)
		.def("YellowGreen", &dyno::Color::YellowGreen)
		.def("ForestGreen", &dyno::Color::ForestGreen)
		.def("OliveDrab", &dyno::Color::OliveDrab)
		.def("DarkKhaki", &dyno::Color::DarkKhaki)
		.def("PaleGoldenrod", &dyno::Color::PaleGoldenrod)
		.def("LtGoldenrodYello", &dyno::Color::LtGoldenrodYello)
		.def("LightYellow", &dyno::Color::LightYellow)
		.def("Yellow", &dyno::Color::Yellow)
		.def("Gold", &dyno::Color::Gold)
		.def("LightGoldenrod", &dyno::Color::LightGoldenrod)
		.def("goldenrod", &dyno::Color::goldenrod)
		.def("DarkGoldenrod", &dyno::Color::DarkGoldenrod)
		.def("RosyBrown", &dyno::Color::RosyBrown)
		.def("IndianRed", &dyno::Color::IndianRed)
		.def("SaddleBrown", &dyno::Color::SaddleBrown)
		.def("Sienna", &dyno::Color::Sienna)
		.def("Peru", &dyno::Color::Peru)
		.def("Burlywood", &dyno::Color::Burlywood)
		.def("Beige", &dyno::Color::Beige)
		.def("Wheat", &dyno::Color::Wheat)
		.def("SandyBrown", &dyno::Color::SandyBrown)
		.def("Tan", &dyno::Color::Tan)
		.def("Chocolate", &dyno::Color::Chocolate)
		.def("Firebrick", &dyno::Color::Firebrick)
		.def("Brown", &dyno::Color::Brown)
		.def("DarkSalmon", &dyno::Color::DarkSalmon)
		.def("Salmon", &dyno::Color::Salmon)
		.def("LightSalmon", &dyno::Color::LightSalmon)
		.def("Orange", &dyno::Color::Orange)
		.def("DarkOrange", &dyno::Color::DarkOrange)
		.def("Coral", &dyno::Color::Coral)
		.def("LightCoral", &dyno::Color::LightCoral)
		.def("Tomato", &dyno::Color::Tomato)
		.def("OrangeRed", &dyno::Color::OrangeRed)
		.def("Red", &dyno::Color::Red)
		.def("HotPink", &dyno::Color::HotPink)
		.def("DeepPink", &dyno::Color::DeepPink)
		.def("Pink", &dyno::Color::Pink)
		.def("LightPink", &dyno::Color::LightPink)
		.def("PaleVioletRed", &dyno::Color::PaleVioletRed)
		.def("Maroon", &dyno::Color::Maroon)
		.def("MediumVioletRed", &dyno::Color::MediumVioletRed)
		.def("VioletRed", &dyno::Color::VioletRed)
		.def("Magenta", &dyno::Color::Magenta)
		.def("Violet", &dyno::Color::Violet)
		.def("Plum", &dyno::Color::Plum)
		.def("Orchid", &dyno::Color::Orchid)
		.def("MediumOrchid", &dyno::Color::MediumOrchid)
		.def("DarkOrchid", &dyno::Color::DarkOrchid)
		.def("DarkViolet", &dyno::Color::DarkViolet)
		.def("BlueViolet", &dyno::Color::BlueViolet)
		.def("Purple", &dyno::Color::Purple)
		.def("MediumPurple", &dyno::Color::MediumPurple)
		.def("Thistle", &dyno::Color::Thistle)
		.def("Snow1", &dyno::Color::Snow1)
		.def("Snow2", &dyno::Color::Snow2)
		.def("Snow3", &dyno::Color::Snow3)
		.def("Snow4", &dyno::Color::Snow4)
		.def("Seashell1", &dyno::Color::Seashell1)
		.def("Seashell2", &dyno::Color::Seashell2)
		.def("Seashell3", &dyno::Color::Seashell3)
		.def("Seashell4", &dyno::Color::Seashell4)
		.def("AntiqueWhite1", &dyno::Color::AntiqueWhite1)
		.def("AntiqueWhite2", &dyno::Color::AntiqueWhite2)
		.def("AntiqueWhite3", &dyno::Color::AntiqueWhite3)
		.def("AntiqueWhite4", &dyno::Color::AntiqueWhite4)
		.def("Bisque1", &dyno::Color::Bisque1)
		.def("Bisque2", &dyno::Color::Bisque2)
		.def("Bisque3", &dyno::Color::Bisque3)
		.def("Bisque4", &dyno::Color::Bisque4)
		.def("PeachPuff1", &dyno::Color::PeachPuff1)
		.def("PeachPuff2", &dyno::Color::PeachPuff2)
		.def("PeachPuff3", &dyno::Color::PeachPuff3)
		.def("PeachPuff4", &dyno::Color::PeachPuff4)
		.def("NavajoWhite1", &dyno::Color::NavajoWhite1)
		.def("NavajoWhite2", &dyno::Color::NavajoWhite2)
		.def("NavajoWhite3", &dyno::Color::NavajoWhite3)
		.def("NavajoWhite4", &dyno::Color::NavajoWhite4)
		.def("LemonChiffon1", &dyno::Color::LemonChiffon1)
		.def("LemonChiffon2", &dyno::Color::LemonChiffon2)
		.def("LemonChiffon3", &dyno::Color::LemonChiffon3)
		.def("LemonChiffon4", &dyno::Color::LemonChiffon4)
		.def("Cornsilk1", &dyno::Color::Cornsilk1)
		.def("Cornsilk2", &dyno::Color::Cornsilk2)
		.def("Cornsilk3", &dyno::Color::Cornsilk3)
		.def("Cornsilk4", &dyno::Color::Cornsilk4)
		.def("Ivory1", &dyno::Color::Ivory1)
		.def("Ivory2", &dyno::Color::Ivory2)
		.def("Ivory3", &dyno::Color::Ivory3)
		.def("Ivory4", &dyno::Color::Ivory4)
		.def("Honeydew1", &dyno::Color::Honeydew1)
		.def("Honeydew2", &dyno::Color::Honeydew2)
		.def("Honeydew3", &dyno::Color::Honeydew3)
		.def("Honeydew4", &dyno::Color::Honeydew4)
		.def("LavenderBlush1", &dyno::Color::LavenderBlush1)
		.def("LavenderBlush2", &dyno::Color::LavenderBlush2)
		.def("LavenderBlush3", &dyno::Color::LavenderBlush3)
		.def("LavenderBlush4", &dyno::Color::LavenderBlush4)
		.def("MistyRose1", &dyno::Color::MistyRose1)
		.def("MistyRose2", &dyno::Color::MistyRose2)
		.def("MistyRose3", &dyno::Color::MistyRose3)
		.def("MistyRose4", &dyno::Color::MistyRose4)
		.def("Azure1", &dyno::Color::Azure1)
		.def("Azure2", &dyno::Color::Azure2)
		.def("Azure3", &dyno::Color::Azure3)
		.def("Azure4", &dyno::Color::Azure4)
		.def("SlateBlue1", &dyno::Color::SlateBlue1)
		.def("SlateBlue2", &dyno::Color::SlateBlue2)
		.def("SlateBlue3", &dyno::Color::SlateBlue3)
		.def("SlateBlue4", &dyno::Color::SlateBlue4)
		.def("RoyalBlue1", &dyno::Color::RoyalBlue1)
		.def("RoyalBlue2", &dyno::Color::RoyalBlue2)
		.def("RoyalBlue3", &dyno::Color::RoyalBlue3)
		.def("RoyalBlue4", &dyno::Color::RoyalBlue4)
		.def("Blue1", &dyno::Color::Blue1)
		.def("Blue2", &dyno::Color::Blue2)
		.def("Blue3", &dyno::Color::Blue3)
		.def("Blue4", &dyno::Color::Blue4)
		.def("DodgerBlue1", &dyno::Color::DodgerBlue1)
		.def("DodgerBlue2", &dyno::Color::DodgerBlue2)
		.def("DodgerBlue3", &dyno::Color::DodgerBlue3)
		.def("DodgerBlue4", &dyno::Color::DodgerBlue4)
		.def("SteelBlue1", &dyno::Color::SteelBlue1)
		.def("SteelBlue2", &dyno::Color::SteelBlue2)
		.def("SteelBlue3", &dyno::Color::SteelBlue3)
		.def("SteelBlue4", &dyno::Color::SteelBlue4)
		.def("DeepSkyBlue1", &dyno::Color::DeepSkyBlue1)
		.def("DeepSkyBlue2", &dyno::Color::DeepSkyBlue2)
		.def("DeepSkyBlue3", &dyno::Color::DeepSkyBlue3)
		.def("DeepSkyBlue4", &dyno::Color::DeepSkyBlue4)
		.def("SkyBlue1", &dyno::Color::SkyBlue1)
		.def("SkyBlue2", &dyno::Color::SkyBlue2)
		.def("SkyBlue3", &dyno::Color::SkyBlue3)
		.def("SkyBlue4", &dyno::Color::SkyBlue4)
		.def("LightSkyBlue1", &dyno::Color::LightSkyBlue1)
		.def("LightSkyBlue2", &dyno::Color::LightSkyBlue2)
		.def("LightSkyBlue3", &dyno::Color::LightSkyBlue3)
		.def("LightSkyBlue4", &dyno::Color::LightSkyBlue4)
		.def("SlateGray1", &dyno::Color::SlateGray1)
		.def("SlateGray2", &dyno::Color::SlateGray2)
		.def("SlateGray3", &dyno::Color::SlateGray3)
		.def("SlateGray4", &dyno::Color::SlateGray4)
		.def("LightSteelBlue1", &dyno::Color::LightSteelBlue1)
		.def("LightSteelBlue2", &dyno::Color::LightSteelBlue2)
		.def("LightSteelBlue3", &dyno::Color::LightSteelBlue3)
		.def("LightSteelBlue4", &dyno::Color::LightSteelBlue4)
		.def("LightBlue1", &dyno::Color::LightBlue1)
		.def("LightBlue2", &dyno::Color::LightBlue2)
		.def("LightBlue3", &dyno::Color::LightBlue3)
		.def("LightBlue4", &dyno::Color::LightBlue4)
		.def("LightCyan1", &dyno::Color::LightCyan1)
		.def("LightCyan2", &dyno::Color::LightCyan2)
		.def("LightCyan3", &dyno::Color::LightCyan3)
		.def("LightCyan4", &dyno::Color::LightCyan4)
		.def("PaleTurquoise1", &dyno::Color::PaleTurquoise1)
		.def("PaleTurquoise2", &dyno::Color::PaleTurquoise2)
		.def("PaleTurquoise3", &dyno::Color::PaleTurquoise3)
		.def("PaleTurquoise4", &dyno::Color::PaleTurquoise4)
		.def("CadetBlue1", &dyno::Color::CadetBlue1)
		.def("CadetBlue2", &dyno::Color::CadetBlue2)
		.def("CadetBlue3", &dyno::Color::CadetBlue3)
		.def("CadetBlue4", &dyno::Color::CadetBlue4)
		.def("Turquoise1", &dyno::Color::Turquoise1)
		.def("Turquoise2", &dyno::Color::Turquoise2)
		.def("Turquoise3", &dyno::Color::Turquoise3)
		.def("Turquoise4", &dyno::Color::Turquoise4)
		.def("Cyan1", &dyno::Color::Cyan1)
		.def("Cyan2", &dyno::Color::Cyan2)
		.def("Cyan3", &dyno::Color::Cyan3)
		.def("Cyan4", &dyno::Color::Cyan4)
		.def("DarkSlateGray1", &dyno::Color::DarkSlateGray1)
		.def("DarkSlateGray2", &dyno::Color::DarkSlateGray2)
		.def("DarkSlateGray3", &dyno::Color::DarkSlateGray3)
		.def("DarkSlateGray4", &dyno::Color::DarkSlateGray4)
		.def("Aquamarine1", &dyno::Color::Aquamarine1)
		.def("Aquamarine2", &dyno::Color::Aquamarine2)
		.def("Aquamarine3", &dyno::Color::Aquamarine3)
		.def("Aquamarine4", &dyno::Color::Aquamarine4)
		.def("DarkSeaGreen1", &dyno::Color::DarkSeaGreen1)
		.def("DarkSeaGreen2", &dyno::Color::DarkSeaGreen2)
		.def("DarkSeaGreen3", &dyno::Color::DarkSeaGreen3)
		.def("DarkSeaGreen4", &dyno::Color::DarkSeaGreen4)
		.def("SeaGreen1", &dyno::Color::SeaGreen1)
		.def("SeaGreen2", &dyno::Color::SeaGreen2)
		.def("SeaGreen3", &dyno::Color::SeaGreen3)
		.def("SeaGreen4", &dyno::Color::SeaGreen4)
		.def("PaleGreen1", &dyno::Color::PaleGreen1)
		.def("PaleGreen2", &dyno::Color::PaleGreen2)
		.def("PaleGreen3", &dyno::Color::PaleGreen3)
		.def("PaleGreen4", &dyno::Color::PaleGreen4)
		.def("SpringGreen1", &dyno::Color::SpringGreen1)
		.def("SpringGreen2", &dyno::Color::SpringGreen2)
		.def("SpringGreen3", &dyno::Color::SpringGreen3)
		.def("SpringGreen4", &dyno::Color::SpringGreen4)
		.def("Green1", &dyno::Color::Green1)
		.def("Green2", &dyno::Color::Green2)
		.def("Green3", &dyno::Color::Green3)
		.def("Green4", &dyno::Color::Green4)
		.def("Chartreuse1", &dyno::Color::Chartreuse1)
		.def("Chartreuse2", &dyno::Color::Chartreuse2)
		.def("Chartreuse3", &dyno::Color::Chartreuse3)
		.def("Chartreuse4", &dyno::Color::Chartreuse4)
		.def("OliveDrab1", &dyno::Color::OliveDrab1)
		.def("OliveDrab2", &dyno::Color::OliveDrab2)
		.def("OliveDrab3", &dyno::Color::OliveDrab3)
		.def("OliveDrab4", &dyno::Color::OliveDrab4)
		.def("DarkOliveGreen1", &dyno::Color::DarkOliveGreen1)
		.def("DarkOliveGreen2", &dyno::Color::DarkOliveGreen2)
		.def("DarkOliveGreen3", &dyno::Color::DarkOliveGreen3)
		.def("DarkOliveGreen4", &dyno::Color::DarkOliveGreen4)
		.def("Khaki1", &dyno::Color::Khaki1)
		.def("Khaki2", &dyno::Color::Khaki2)
		.def("Khaki3", &dyno::Color::Khaki3)
		.def("Khaki4", &dyno::Color::Khaki4)
		.def("LightGoldenrod1", &dyno::Color::LightGoldenrod1)
		.def("LightGoldenrod2", &dyno::Color::LightGoldenrod2)
		.def("LightGoldenrod3", &dyno::Color::LightGoldenrod3)
		.def("LightGoldenrod4", &dyno::Color::LightGoldenrod4)
		.def("LightYellow1", &dyno::Color::LightYellow1)
		.def("LightYellow2", &dyno::Color::LightYellow2)
		.def("LightYellow3", &dyno::Color::LightYellow3)
		.def("LightYellow4", &dyno::Color::LightYellow4)
		.def("Yellow1", &dyno::Color::Yellow1)
		.def("Yellow2", &dyno::Color::Yellow2)
		.def("Yellow3", &dyno::Color::Yellow3)
		.def("Yellow4", &dyno::Color::Yellow4)
		.def("Gold1", &dyno::Color::Gold1)
		.def("Gold2", &dyno::Color::Gold2)
		.def("Gold3", &dyno::Color::Gold3)
		.def("Gold4", &dyno::Color::Gold4)
		.def("Goldenrod1", &dyno::Color::Goldenrod1)
		.def("Goldenrod2", &dyno::Color::Goldenrod2)
		.def("Goldenrod3", &dyno::Color::Goldenrod3)
		.def("Goldenrod4", &dyno::Color::Goldenrod4)
		.def("DarkGoldenrod1", &dyno::Color::DarkGoldenrod1)
		.def("DarkGoldenrod2", &dyno::Color::DarkGoldenrod2)
		.def("DarkGoldenrod3", &dyno::Color::DarkGoldenrod3)
		.def("DarkGoldenrod4", &dyno::Color::DarkGoldenrod4)
		.def("RosyBrown1", &dyno::Color::RosyBrown1)
		.def("RosyBrown2", &dyno::Color::RosyBrown2)
		.def("RosyBrown3", &dyno::Color::RosyBrown3)
		.def("RosyBrown4", &dyno::Color::RosyBrown4)
		.def("IndianRed1", &dyno::Color::IndianRed1)
		.def("IndianRed2", &dyno::Color::IndianRed2)
		.def("IndianRed3", &dyno::Color::IndianRed3)
		.def("IndianRed4", &dyno::Color::IndianRed4)
		.def("Sienna1", &dyno::Color::Sienna1)
		.def("Sienna2", &dyno::Color::Sienna2)
		.def("Sienna3", &dyno::Color::Sienna3)
		.def("Sienna4", &dyno::Color::Sienna4)
		.def("Burlywood1", &dyno::Color::Burlywood1)
		.def("Burlywood2", &dyno::Color::Burlywood2)
		.def("Burlywood3", &dyno::Color::Burlywood3)
		.def("Burlywood4", &dyno::Color::Burlywood4)
		.def("Wheat1", &dyno::Color::Wheat1)
		.def("Wheat2", &dyno::Color::Wheat2)
		.def("Wheat3", &dyno::Color::Wheat3)
		.def("Wheat4", &dyno::Color::Wheat4)
		.def("Tan1", &dyno::Color::Tan1)
		.def("Tan2", &dyno::Color::Tan2)
		.def("Tan3", &dyno::Color::Tan3)
		.def("Tan4", &dyno::Color::Tan4)
		.def("Chocolate1", &dyno::Color::Chocolate1)
		.def("Chocolate2", &dyno::Color::Chocolate2)
		.def("Chocolate3", &dyno::Color::Chocolate3)
		.def("Chocolate4", &dyno::Color::Chocolate4)
		.def("Firebrick1", &dyno::Color::Firebrick1)
		.def("Firebrick2", &dyno::Color::Firebrick2)
		.def("Firebrick3", &dyno::Color::Firebrick3)
		.def("Firebrick4", &dyno::Color::Firebrick4)
		.def("Brown1", &dyno::Color::Brown1)
		.def("Brown2", &dyno::Color::Brown2)
		.def("Brown3", &dyno::Color::Brown3)
		.def("Brown4", &dyno::Color::Brown4)
		.def("Salmon1", &dyno::Color::Salmon1)
		.def("Salmon2", &dyno::Color::Salmon2)
		.def("Salmon3", &dyno::Color::Salmon3)
		.def("Salmon4", &dyno::Color::Salmon4)
		.def("LightSalmon1", &dyno::Color::LightSalmon1)
		.def("LightSalmon2", &dyno::Color::LightSalmon2)
		.def("LightSalmon3", &dyno::Color::LightSalmon3)
		.def("LightSalmon4", &dyno::Color::LightSalmon4)
		.def("Orange1", &dyno::Color::Orange1)
		.def("Orange2", &dyno::Color::Orange2)
		.def("Orange3", &dyno::Color::Orange3)
		.def("Orange4", &dyno::Color::Orange4)
		.def("DarkOrange1", &dyno::Color::DarkOrange1)
		.def("DarkOrange2", &dyno::Color::DarkOrange2)
		.def("DarkOrange3", &dyno::Color::DarkOrange3)
		.def("DarkOrange4", &dyno::Color::DarkOrange4)
		.def("Coral1", &dyno::Color::Coral1)
		.def("Coral2", &dyno::Color::Coral2)
		.def("Coral3", &dyno::Color::Coral3)
		.def("Coral4", &dyno::Color::Coral4)
		.def("Tomato1", &dyno::Color::Tomato1)
		.def("Tomato2", &dyno::Color::Tomato2)
		.def("Tomato3", &dyno::Color::Tomato3)
		.def("Tomato4", &dyno::Color::Tomato4)
		.def("OrangeRed1", &dyno::Color::OrangeRed1)
		.def("OrangeRed2", &dyno::Color::OrangeRed2)
		.def("OrangeRed3", &dyno::Color::OrangeRed3)
		.def("OrangeRed4", &dyno::Color::OrangeRed4)
		.def("Red1", &dyno::Color::Red1)
		.def("Red2", &dyno::Color::Red2)
		.def("Red3", &dyno::Color::Red3)
		.def("Red4", &dyno::Color::Red4)
		.def("DeepPink1", &dyno::Color::DeepPink1)
		.def("DeepPink2", &dyno::Color::DeepPink2)
		.def("DeepPink3", &dyno::Color::DeepPink3)
		.def("DeepPink4", &dyno::Color::DeepPink4)
		.def("HotPink1", &dyno::Color::HotPink1)
		.def("HotPink2", &dyno::Color::HotPink2)
		.def("HotPink3", &dyno::Color::HotPink3)
		.def("HotPink4", &dyno::Color::HotPink4)
		.def("Pink1", &dyno::Color::Pink1)
		.def("Pink2", &dyno::Color::Pink2)
		.def("Pink3", &dyno::Color::Pink3)
		.def("Pink4", &dyno::Color::Pink4)
		.def("LightPink1", &dyno::Color::LightPink1)
		.def("LightPink2", &dyno::Color::LightPink2)
		.def("LightPink3", &dyno::Color::LightPink3)
		.def("LightPink4", &dyno::Color::LightPink4)
		.def("PaleVioletRed1", &dyno::Color::PaleVioletRed1)
		.def("PaleVioletRed2", &dyno::Color::PaleVioletRed2)
		.def("PaleVioletRed3", &dyno::Color::PaleVioletRed3)
		.def("PaleVioletRed4", &dyno::Color::PaleVioletRed4)
		.def("Maroon1", &dyno::Color::Maroon1)
		.def("Maroon2", &dyno::Color::Maroon2)
		.def("Maroon3", &dyno::Color::Maroon3)
		.def("Maroon4", &dyno::Color::Maroon4)
		.def("VioletRed1", &dyno::Color::VioletRed1)
		.def("VioletRed2", &dyno::Color::VioletRed2)
		.def("VioletRed3", &dyno::Color::VioletRed3)
		.def("VioletRed4", &dyno::Color::VioletRed4)
		.def("Magenta1", &dyno::Color::Magenta1)
		.def("Magenta2", &dyno::Color::Magenta2)
		.def("Magenta3", &dyno::Color::Magenta3)
		.def("Magenta4", &dyno::Color::Magenta4)
		.def("Orchid1", &dyno::Color::Orchid1)
		.def("Orchid2", &dyno::Color::Orchid2)
		.def("Orchid3", &dyno::Color::Orchid3)
		.def("Orchid4", &dyno::Color::Orchid4)
		.def("Plum1", &dyno::Color::Plum1)
		.def("Plum2", &dyno::Color::Plum2)
		.def("Plum3", &dyno::Color::Plum3)
		.def("Plum4", &dyno::Color::Plum4)
		.def("MediumOrchid1", &dyno::Color::MediumOrchid1)
		.def("MediumOrchid2", &dyno::Color::MediumOrchid2)
		.def("MediumOrchid3", &dyno::Color::MediumOrchid3)
		.def("MediumOrchid4", &dyno::Color::MediumOrchid4)
		.def("DarkOrchid1", &dyno::Color::DarkOrchid1)
		.def("DarkOrchid2", &dyno::Color::DarkOrchid2)
		.def("DarkOrchid3", &dyno::Color::DarkOrchid3)
		.def("DarkOrchid4", &dyno::Color::DarkOrchid4)
		.def("Purple1", &dyno::Color::Purple1)
		.def("Purple2", &dyno::Color::Purple2)
		.def("Purple3", &dyno::Color::Purple3)
		.def("Purple4", &dyno::Color::Purple4)
		.def("MediumPurple1", &dyno::Color::MediumPurple1)
		.def("MediumPurple2", &dyno::Color::MediumPurple2)
		.def("MediumPurple3", &dyno::Color::MediumPurple3)
		.def("MediumPurple4", &dyno::Color::MediumPurple4)
		.def("Thistle1", &dyno::Color::Thistle1)
		.def("Thistle2", &dyno::Color::Thistle2)
		.def("Thistle3", &dyno::Color::Thistle3)
		.def("Thistle4", &dyno::Color::Thistle4)
		.def("Grey11", &dyno::Color::Grey11)
		.def("Grey21", &dyno::Color::Grey21)
		.def("Grey31", &dyno::Color::Grey31)
		.def("Grey41", &dyno::Color::Grey41)
		.def("Grey51", &dyno::Color::Grey51)
		.def("Grey61", &dyno::Color::Grey61)
		.def("Grey71", &dyno::Color::Grey71)
		.def("Grey81", &dyno::Color::Grey81)
		.def("Grey91", &dyno::Color::Grey91)
		.def("DarkGrey", &dyno::Color::DarkGrey)
		.def("DarkBlue", &dyno::Color::DarkBlue)
		.def("DarkCyan", &dyno::Color::DarkCyan)
		.def("DarkMagenta", &dyno::Color::DarkMagenta)
		.def("DarkRed", &dyno::Color::DarkRed)
		.def("LightGreen", &dyno::Color::LightGreen);

	py::class_<dyno::FilePath>(m, "FilePath")
		.def(py::init<const std::string&>())
		.def("string", &dyno::FilePath::string)
		.def("path", &dyno::FilePath::path)
		.def("__eq__", &dyno::FilePath::operator==)
		.def("__ne__", &dyno::FilePath::operator!=)
		.def("is_path", &dyno::FilePath::is_path)
		.def("extensions", &dyno::FilePath::extensions)
		.def("add_extension", &dyno::FilePath::add_extension)
		.def("set_as_path", &dyno::FilePath::set_as_path)
		.def("set_path", &dyno::FilePath::set_path);

	//FBase
	py::class_<InstanceBase, FBase, std::shared_ptr<InstanceBase>>(m, "InstanceBase")
		.def("canBeConnectedBy", &InstanceBase::canBeConnectedBy)
		.def("setObjectPointer", &InstanceBase::setObjectPointer)
		.def("objectPointer", &InstanceBase::objectPointer)
		.def("standardObjectPointer", &InstanceBase::standardObjectPointer)
		.def("className", &InstanceBase::className);

	class ModulePublicist : public Module
	{
	public:
		using Module::appendExportModule;
		using Module::removeExportModule;
		using Module::preprocess;
		using Module::postprocess;
		using Module::validateInputs;
		using Module::validateOutputs;
		using Module::requireUpdate;
		using Module::updateStarted;
		using Module::updateEnded;
	};

	//module
	py::class_<Module, OBase, std::shared_ptr<Module>>(m, "Module")
		.def(py::init<>())
		.def("initialize", &Module::initialize)
		.def("update", &Module::update)
		.def("setName", &Module::setName)
		.def("getName", &Module::getName)
		.def("setParentNode", &Module::setParentNode)
		.def("getParentNode", &Module::getParentNode)
		.def("getSceneGraph", &Module::getSceneGraph)
		.def("isInitialized", &Module::isInitialized)
		.def("getModuleType", &Module::getModuleType)
		.def("attachField", &Module::attachField)
		.def("isInputComplete", &Module::isInputComplete)
		.def("isOutputCompete", &Module::isOutputCompete)
		.def("varForceUpdate", &Module::varForceUpdate, py::return_value_policy::reference)
		.def("setUpdateAlways", &Module::setUpdateAlways)
		.def("initializeImpl", &Module::initializeImpl)
		.def("updateImpl", &Module::updateImpl)
		// protected
		.def("appendExportModule", &ModulePublicist::appendExportModule)
		.def("removeExportModule", &ModulePublicist::removeExportModule)
		.def("preprocess", &ModulePublicist::preprocess)
		.def("postprocess", &ModulePublicist::postprocess)
		.def("validateInputs", &ModulePublicist::validateInputs)
		.def("validateOutputs", &ModulePublicist::validateOutputs)
		.def("requireUpdate", &ModulePublicist::requireUpdate)
		.def("updateStarted", &ModulePublicist::updateStarted)
		.def("updateEnded", &ModulePublicist::updateEnded);

	py::class_<DebugInfo, Module, std::shared_ptr<DebugInfo>>(m, "DebugInfo")
		.def("print", &DebugInfo::print)
		.def("varPrefix", &DebugInfo::varPrefix)
		.def("getModuleType", &DebugInfo::getModuleType);

	py::class_<PrintInt, Module, std::shared_ptr<PrintInt>>(m, "PrintInt")
		.def("print", &PrintInt::print)
		.def("inInt", &PrintInt::inInt);

	py::class_<PrintUnsigned, Module, std::shared_ptr<PrintUnsigned>>(m, "PrintUnsigned")
		.def("print", &PrintUnsigned::print)
		.def("inUnsigned", &PrintUnsigned::inUnsigned);

	py::class_<PrintFloat, Module, std::shared_ptr<PrintFloat>>(m, "PrintFloat")
		.def("print", &PrintFloat::print)
		.def("inFloat", &PrintFloat::inFloat);

	py::class_<PrintVector, Module, std::shared_ptr<PrintVector>>(m, "PrintVector")
		.def("print", &PrintVector::print)
		.def("inVector", &PrintVector::inVector);

	py::class_<VisualModule, Module, std::shared_ptr<VisualModule>>(m, "VisualModule")
		.def(py::init<>())
		.def("setVisible", &VisualModule::setVisible)
		.def("isVisible", &VisualModule::isVisible)
		.def("getModuleType", &VisualModule::getModuleType);


	class ComputeModuleTrampoline : public ComputeModule
	{
	public:

		void compute() override
		{
			PYBIND11_OVERRIDE_PURE(
				void,
				dyno::ComputeModule,
				compute
			);
		}
	};

	class ComputeModulePublicist : public ComputeModule
	{
	public:
		using ComputeModule::compute;
	};

	py::class_<ComputeModule, Module, ComputeModuleTrampoline, std::shared_ptr<ComputeModule>>(m, "ComputeModule")
		.def("getModuleType", &dyno::ComputeModule::getModuleType)
		// protected
		.def("compute", &ComputeModulePublicist::compute);

	py::class_<Add, ComputeModule, std::shared_ptr<Add>>(m, "Add")
		.def("caption", &Add::caption)
		.def("getModuleType", &Add::getModuleType);

	py::class_<Divide, ComputeModule, std::shared_ptr<Divide>>(m, "Divide")
		.def("caption", &Divide::caption)
		.def("getModuleType", &Divide::getModuleType);

	py::class_<Multiply, ComputeModule, std::shared_ptr<Multiply>>(m, "Multiply")
		.def("caption", &Multiply::caption)
		.def("getModuleType", &Multiply::getModuleType);

	py::class_<Subtract, ComputeModule, std::shared_ptr<Subtract>>(m, "Subtract")
		.def("caption", &Subtract::caption)
		.def("getModuleType", &Subtract::getModuleType);


	class PipelineTrampoline : public Pipeline
	{
	public:

		void updateImpl() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Pipeline,
				updateImpl
			);
		}

		bool printDebugInfo() override
		{
			PYBIND11_OVERRIDE_PURE(
				bool,
				dyno::Pipeline,
				printDebugInfo
			);
		}
	};

	class PipelinePublicist : public Pipeline
	{
	public:
		using Pipeline::preprocess;
		using Pipeline::updateImpl;
		using Pipeline::requireUpdate;
		using Pipeline::printDebugInfo;
	};

	py::class_<Pipeline, Module, PipelineTrampoline, std::shared_ptr<Pipeline>>(m, "Pipeline")
		.def("sizeOfDynamicModules", &Pipeline::sizeOfDynamicModules)
		.def("sizeOfPersistentModules", &Pipeline::sizeOfPersistentModules)
		.def("pushModule", &Pipeline::pushModule)
		.def("popModule", &Pipeline::popModule)
		.def("createModule", static_cast<std::shared_ptr<Module>(Pipeline::*)()>(&Pipeline::createModule))
		.def("findFirstModule", static_cast<std::shared_ptr<Module>(Pipeline::*)()>(&Pipeline::findFirstModule))
		.def("findFirstModuleSurface", static_cast<std::shared_ptr<dyno::GLSurfaceVisualModule>(Pipeline::*)()>(&Pipeline::findFirstModule))
		.def("findFirstModulePoint", static_cast<std::shared_ptr<dyno::GLPointVisualModule>(Pipeline::*)()>(&Pipeline::findFirstModule))
		.def("findFirstModuleMapping", static_cast<std::shared_ptr<dyno::ColorMapping<dyno::DataType3f>>(Pipeline::*)()>(&Pipeline::findFirstModule))
		.def("clear", &Pipeline::clear)
		.def("pushPersistentModule", &Pipeline::pushPersistentModule)
		.def("activeModules", &Pipeline::activeModules, py::return_value_policy::reference)
		.def("allModules", &Pipeline::allModules, py::return_value_policy::reference)
		.def("enable", &Pipeline::enable)
		.def("disable", &Pipeline::disable)
		.def("updateExecutionQueue", &Pipeline::updateExecutionQueue)
		.def("forceUpdate", &Pipeline::forceUpdate)
		.def("promoteOutputToNode", &Pipeline::promoteOutputToNode, py::return_value_policy::reference)
		.def("demoteOutputFromNode", &Pipeline::demoteOutputFromNode)
		// protected
		.def("preprocess", &PipelinePublicist::preprocess)
		.def("updateImpl", &PipelinePublicist::updateImpl)
		.def("requireUpdate", &PipelinePublicist::requireUpdate)
		.def("printDebugInfo", &PipelinePublicist::printDebugInfo);

	class ConstraintModuleTrampoline : public ConstraintModule
	{
	public:

		void updateImpl() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ConstraintModule,
				updateImpl
			);
		}
	};

	class ConstraintModulePublicist : public ConstraintModule
	{
	public:
		using ConstraintModule::updateImpl;
	};

	py::class_<ConstraintModule, Module, ConstraintModuleTrampoline, std::shared_ptr<ConstraintModule>>(m, "ConstraintModule")
		.def("constrain", &dyno::ConstraintModule::constrain)
		.def("getModuleType", &dyno::ConstraintModule::getModuleType)
		// protected
		.def("updateImpl", &ConstraintModulePublicist::updateImpl);

	class GroupModuleTrampoline : public GroupModule
	{
	public:

		void updateImpl() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::GroupModule,
				updateImpl
			);
		}
	};

	class GroupModulePublicist : public GroupModule
	{
	public:
		using GroupModule::preprocess;
		using GroupModule::updateImpl;
	};

	py::class_<GroupModule, Module, GroupModuleTrampoline, std::shared_ptr<GroupModule>>(m, "GroupModule")
		.def(py::init<>())
		.def("pushModule", &GroupModule::pushModule)
		.def("moduleList", &GroupModule::moduleList, py::return_value_policy::reference)
		.def("setParentNode", &GroupModule::setParentNode)
		// protected
		.def("preprocess", &GroupModulePublicist::preprocess)
		.def("updateImpl", &GroupModulePublicist::updateImpl);


	py::class_<TopologyMapping, Module, std::shared_ptr< TopologyMapping>>(m, "TopologyMapping");


	py::enum_<typename dyno::PButtonType>(m, "PButtonType")
		.value("BT_UNKOWN", dyno::PButtonType::BT_UNKOWN)
		.value("BT_LEFT", dyno::PButtonType::BT_LEFT)
		.value("BT_RIGHT", dyno::PButtonType::BT_RIGHT)
		.value("BT_MIDDLE", dyno::PButtonType::BT_MIDDLE);

	py::enum_<typename dyno::PActionType>(m, "PActionType")
		.value("AT_UNKOWN", dyno::PActionType::AT_UNKOWN)
		.value("AT_RELEASE", dyno::PActionType::AT_RELEASE)
		.value("AT_PRESS", dyno::PActionType::AT_PRESS)
		.value("AT_REPEAT", dyno::PActionType::AT_REPEAT);

	py::enum_<typename dyno::PKeyboardType>(m, "PKeyboardType")
		.value("PKEY_UNKNOWN", dyno::PKeyboardType::PKEY_UNKNOWN)
		.value("PKEY_SPACE", dyno::PKeyboardType::PKEY_SPACE)
		.value("PKEY_APOSTROPHE", dyno::PKeyboardType::PKEY_APOSTROPHE)
		.value("PKEY_COMMA", dyno::PKeyboardType::PKEY_COMMA)
		.value("PKEY_MINUS", dyno::PKeyboardType::PKEY_MINUS)
		.value("PKEY_PERIOD", dyno::PKeyboardType::PKEY_PERIOD)
		.value("PKEY_SLASH", dyno::PKeyboardType::PKEY_SLASH)
		.value("PKEY_0", dyno::PKeyboardType::PKEY_0)
		.value("PKEY_1", dyno::PKeyboardType::PKEY_1)
		.value("PKEY_2", dyno::PKeyboardType::PKEY_2)
		.value("PKEY_3", dyno::PKeyboardType::PKEY_3)
		.value("PKEY_4", dyno::PKeyboardType::PKEY_4)
		.value("PKEY_5", dyno::PKeyboardType::PKEY_5)
		.value("PKEY_6", dyno::PKeyboardType::PKEY_6)
		.value("PKEY_7", dyno::PKeyboardType::PKEY_7)
		.value("PKEY_8", dyno::PKeyboardType::PKEY_8)
		.value("PKEY_9", dyno::PKeyboardType::PKEY_9)
		.value("PKEY_SEMICOLON", dyno::PKeyboardType::PKEY_SEMICOLON)
		.value("PKEY_EQUAL", dyno::PKeyboardType::PKEY_EQUAL)
		.value("PKEY_A", dyno::PKeyboardType::PKEY_A)
		.value("PKEY_B", dyno::PKeyboardType::PKEY_B)
		.value("PKEY_C", dyno::PKeyboardType::PKEY_C)
		.value("PKEY_D", dyno::PKeyboardType::PKEY_D)
		.value("PKEY_E", dyno::PKeyboardType::PKEY_E)
		.value("PKEY_F", dyno::PKeyboardType::PKEY_F)
		.value("PKEY_G", dyno::PKeyboardType::PKEY_G)
		.value("PKEY_H", dyno::PKeyboardType::PKEY_H)
		.value("PKEY_I", dyno::PKeyboardType::PKEY_I)
		.value("PKEY_J", dyno::PKeyboardType::PKEY_J)
		.value("PKEY_K", dyno::PKeyboardType::PKEY_K)
		.value("PKEY_L", dyno::PKeyboardType::PKEY_L)
		.value("PKEY_M", dyno::PKeyboardType::PKEY_M)
		.value("PKEY_N", dyno::PKeyboardType::PKEY_N)
		.value("PKEY_O", dyno::PKeyboardType::PKEY_O)
		.value("PKEY_P", dyno::PKeyboardType::PKEY_P)
		.value("PKEY_Q", dyno::PKeyboardType::PKEY_Q)
		.value("PKEY_R", dyno::PKeyboardType::PKEY_R)
		.value("PKEY_S", dyno::PKeyboardType::PKEY_S)
		.value("PKEY_T", dyno::PKeyboardType::PKEY_T)
		.value("PKEY_U", dyno::PKeyboardType::PKEY_U)
		.value("PKEY_V", dyno::PKeyboardType::PKEY_V)
		.value("PKEY_W", dyno::PKeyboardType::PKEY_W)
		.value("PKEY_X", dyno::PKeyboardType::PKEY_X)
		.value("PKEY_Y", dyno::PKeyboardType::PKEY_Y)
		.value("PKEY_Z", dyno::PKeyboardType::PKEY_Z)
		.value("PKEY_LEFT_BRACKET", dyno::PKeyboardType::PKEY_LEFT_BRACKET)
		.value("PKEY_BACKSLASH", dyno::PKeyboardType::PKEY_BACKSLASH)
		.value("PKEY_RIGHT_BRACKET", dyno::PKeyboardType::PKEY_RIGHT_BRACKET)
		.value("PKEY_GRAVE_ACCENT", dyno::PKeyboardType::PKEY_GRAVE_ACCENT)
		.value("PKEY_WORLD_1", dyno::PKeyboardType::PKEY_WORLD_1)
		.value("PKEY_WORLD_2", dyno::PKeyboardType::PKEY_WORLD_2)
		.value("PKEY_ESCAPE", dyno::PKeyboardType::PKEY_ESCAPE)
		.value("PKEY_ENTER", dyno::PKeyboardType::PKEY_ENTER)
		.value("PKEY_TAB", dyno::PKeyboardType::PKEY_TAB)
		.value("PKEY_BACKSPACE", dyno::PKeyboardType::PKEY_BACKSPACE)
		.value("PKEY_INSERT", dyno::PKeyboardType::PKEY_INSERT)
		.value("PKEY_DELETE", dyno::PKeyboardType::PKEY_DELETE)
		.value("PKEY_RIGHT", dyno::PKeyboardType::PKEY_RIGHT)
		.value("PKEY_LEFT", dyno::PKeyboardType::PKEY_LEFT)
		.value("PKEY_DOWN", dyno::PKeyboardType::PKEY_DOWN)
		.value("PKEY_UP", dyno::PKeyboardType::PKEY_UP)
		.value("PKEY_PAGE_UP", dyno::PKeyboardType::PKEY_PAGE_UP)
		.value("PKEY_PAGE_DOWN", dyno::PKeyboardType::PKEY_PAGE_DOWN)
		.value("PKEY_HOME", dyno::PKeyboardType::PKEY_HOME)
		.value("PKEY_END", dyno::PKeyboardType::PKEY_END)
		.value("PKEY_CAPS_LOCK", dyno::PKeyboardType::PKEY_CAPS_LOCK)
		.value("PKEY_SCROLL_LOCK", dyno::PKeyboardType::PKEY_SCROLL_LOCK)
		.value("PKEY_NUM_LOCK", dyno::PKeyboardType::PKEY_NUM_LOCK)
		.value("PKEY_PRINT_SCREEN", dyno::PKeyboardType::PKEY_PRINT_SCREEN)
		.value("PKEY_PAUSE", dyno::PKeyboardType::PKEY_PAUSE)
		.value("PKEY_F1", dyno::PKeyboardType::PKEY_F1)
		.value("PKEY_F2", dyno::PKeyboardType::PKEY_F2)
		.value("PKEY_F3", dyno::PKeyboardType::PKEY_F3)
		.value("PKEY_F4", dyno::PKeyboardType::PKEY_F4)
		.value("PKEY_F5", dyno::PKeyboardType::PKEY_F5)
		.value("PKEY_F6", dyno::PKeyboardType::PKEY_F6)
		.value("PKEY_F7", dyno::PKeyboardType::PKEY_F7)
		.value("PKEY_F8", dyno::PKeyboardType::PKEY_F8)
		.value("PKEY_F9", dyno::PKeyboardType::PKEY_F9)
		.value("PKEY_F10", dyno::PKeyboardType::PKEY_F10)
		.value("PKEY_F11", dyno::PKeyboardType::PKEY_F11)
		.value("PKEY_F12", dyno::PKeyboardType::PKEY_F12)
		.value("PKEY_F13", dyno::PKeyboardType::PKEY_F13)
		.value("PKEY_F14", dyno::PKeyboardType::PKEY_F14)
		.value("PKEY_F15", dyno::PKeyboardType::PKEY_F15)
		.value("PKEY_F16", dyno::PKeyboardType::PKEY_F16)
		.value("PKEY_F17", dyno::PKeyboardType::PKEY_F17)
		.value("PKEY_F18", dyno::PKeyboardType::PKEY_F18)
		.value("PKEY_F19", dyno::PKeyboardType::PKEY_F19)
		.value("PKEY_F20", dyno::PKeyboardType::PKEY_F20)
		.value("PKEY_F21", dyno::PKeyboardType::PKEY_F21)
		.value("PKEY_F22", dyno::PKeyboardType::PKEY_F22)
		.value("PKEY_F23", dyno::PKeyboardType::PKEY_F23)
		.value("PKEY_F24", dyno::PKeyboardType::PKEY_F24)
		.value("PKEY_F25", dyno::PKeyboardType::PKEY_F25)
		.value("PKEY_KP_0", dyno::PKeyboardType::PKEY_KP_0)
		.value("PKEY_KP_1", dyno::PKeyboardType::PKEY_KP_1)
		.value("PKEY_KP_2", dyno::PKeyboardType::PKEY_KP_2)
		.value("PKEY_KP_3", dyno::PKeyboardType::PKEY_KP_3)
		.value("PKEY_KP_4", dyno::PKeyboardType::PKEY_KP_4)
		.value("PKEY_KP_5", dyno::PKeyboardType::PKEY_KP_5)
		.value("PKEY_KP_6", dyno::PKeyboardType::PKEY_KP_6)
		.value("PKEY_KP_7", dyno::PKeyboardType::PKEY_KP_7)
		.value("PKEY_KP_8", dyno::PKeyboardType::PKEY_KP_8)
		.value("PKEY_KP_9", dyno::PKeyboardType::PKEY_KP_9)
		.value("PKEY_KP_DECIMAL", dyno::PKeyboardType::PKEY_KP_DECIMAL)
		.value("PKEY_KP_DIVIDE", dyno::PKeyboardType::PKEY_KP_DIVIDE)
		.value("PKEY_KP_MULTIPLY", dyno::PKeyboardType::PKEY_KP_MULTIPLY)
		.value("PKEY_KP_SUBTRACT", dyno::PKeyboardType::PKEY_KP_SUBTRACT)
		.value("PKEY_KP_ADD", dyno::PKeyboardType::PKEY_KP_ADD)
		.value("PKEY_KP_ENTER", dyno::PKeyboardType::PKEY_KP_ENTER)
		.value("PKEY_KP_EQUAL", dyno::PKeyboardType::PKEY_KP_EQUAL)
		.value("PKEY_LEFT_SHIFT", dyno::PKeyboardType::PKEY_LEFT_SHIFT)
		.value("PKEY_LEFT_CONTROL", dyno::PKeyboardType::PKEY_LEFT_CONTROL)
		.value("PKEY_LEFT_ALT", dyno::PKeyboardType::PKEY_LEFT_ALT)
		.value("PKEY_LEFT_SUPER", dyno::PKeyboardType::PKEY_LEFT_SUPER)
		.value("PKEY_RIGHT_SHIFT", dyno::PKeyboardType::PKEY_RIGHT_SHIFT)
		.value("PKEY_RIGHT_CONTROL", dyno::PKeyboardType::PKEY_RIGHT_CONTROL)
		.value("PKEY_RIGHT_ALT", dyno::PKeyboardType::PKEY_RIGHT_ALT)
		.value("PKEY_RIGHT_SUPER", dyno::PKeyboardType::PKEY_RIGHT_SUPER);

	py::enum_<typename dyno::PModifierBits>(m, "PModifierBits")
		.value("MB_NO_MODIFIER", dyno::PModifierBits::MB_NO_MODIFIER)
		.value("MB_SHIFT", dyno::PModifierBits::MB_SHIFT)
		.value("MB_CONTROL", dyno::PModifierBits::MB_CONTROL)
		.value("MB_ALT", dyno::PModifierBits::MB_ALT)
		.value("MB_SUPER", dyno::PModifierBits::MB_SUPER)
		.value("MB_CAPS_LOCK", dyno::PModifierBits::MB_CAPS_LOCK)
		.value("MB_NUM_LOCK", dyno::PModifierBits::MB_NUM_LOCK);

	py::class_<dyno::PKeyboardEvent>(m, "PKeyboardEvent")
		.def(py::init<>())
		.def_readwrite("key", &dyno::PKeyboardEvent::key)
		.def_readwrite("action", &dyno::PKeyboardEvent::action)
		.def_readwrite("mods", &dyno::PKeyboardEvent::mods)
		.def_property_readonly("shiftKeyPressed", &dyno::PKeyboardEvent::shiftKeyPressed)
		.def_property_readonly("controlKeyPressed", &dyno::PKeyboardEvent::controlKeyPressed)
		.def_property_readonly("altKeyPressed", &dyno::PKeyboardEvent::altKeyPressed)
		.def_property_readonly("superKeyPressed", &dyno::PKeyboardEvent::superKeyPressed)
		.def_property_readonly("capsLockEnabled", &dyno::PKeyboardEvent::capsLockEnabled)
		.def_property_readonly("numLockEnabled", &dyno::PKeyboardEvent::numLockEnabled);

	py::class_<dyno::PMouseEvent>(m, "PMouseEvent")
		.def(py::init<>())
		.def_readwrite("buttonType", &dyno::PMouseEvent::buttonType)
		.def_readwrite("actionType", &dyno::PMouseEvent::actionType)
		.def_readwrite("mods", &dyno::PMouseEvent::mods)
		.def_readwrite("ray", &dyno::PMouseEvent::ray)
		.def_readwrite("camera", &dyno::PMouseEvent::camera)
		.def_readwrite("x", &dyno::PMouseEvent::x)
		.def_readwrite("y", &dyno::PMouseEvent::y)
		.def_property_readonly("shiftKeyPressed", &dyno::PMouseEvent::shiftKeyPressed)
		.def_property_readonly("controlKeyPressed", &dyno::PMouseEvent::controlKeyPressed)
		.def_property_readonly("altKeyPressed", &dyno::PMouseEvent::altKeyPressed)
		.def_property_readonly("superKeyPressed", &dyno::PMouseEvent::superKeyPressed)
		.def_property_readonly("capsLockEnabled", &dyno::PMouseEvent::capsLockEnabled)
		.def_property_readonly("numLockEnabled", &dyno::PMouseEvent::numLockEnabled)
		.def("__ne__", [](dyno::PMouseEvent& self, const dyno::PMouseEvent& other) {
		return self != other;
			})
		.def("__eq__", [](dyno::PMouseEvent& self, const dyno::PMouseEvent& other) {
				return self == other;
			});

			py::class_<InputModule, Module, std::shared_ptr<InputModule>>(m, "InputModule")
				.def(py::init<>())
				.def("getModuleType", &InputModule::getModuleType);

			class KeyboardInputModuleTrampoline : public KeyboardInputModule
			{
			public:
				void onEvent(dyno::PKeyboardEvent event) override
				{
					PYBIND11_OVERRIDE_PURE(
						void,
						KeyboardInputModule,
						onEvent,
						event
					);
				}

				bool requireUpdate() override
				{
					PYBIND11_OVERRIDE(
						bool,
						KeyboardInputModule,
						requireUpdate
					);
				}
			};

			class KeyboardInputModulePublicist : public KeyboardInputModule
			{
			public:
				using KeyboardInputModule::onEvent;
				using KeyboardInputModule::updateImpl;
				using KeyboardInputModule::requireUpdate;
			};

			py::class_<KeyboardInputModule, InputModule, KeyboardInputModuleTrampoline, std::shared_ptr<KeyboardInputModule>>(m, "KeyboardInputModule")
				.def(py::init<>())
				.def("varCacheEvent", &KeyboardInputModule::varCacheEvent, py::return_value_policy::reference)
				.def("enqueueEvent", &KeyboardInputModule::enqueueEvent, py::return_value_policy::reference)
				.def("onEvent", &KeyboardInputModulePublicist::onEvent)
				.def("updateImpl", &KeyboardInputModulePublicist::updateImpl)
				.def("requireUpdate", &KeyboardInputModulePublicist::requireUpdate);

			class MouseInputModuleTrampoline : public MouseInputModule
			{
			public:

				void onEvent(dyno::PMouseEvent event) override
				{
					PYBIND11_OVERRIDE_PURE(
						void,
						MouseInputModule,
						onEvent,
						event
					);
				}

				bool requireUpdate() override
				{
					PYBIND11_OVERRIDE(
						bool,
						MouseInputModule,
						requireUpdate
					);
				}
			};

			class MouseInputModulePublicist : public MouseInputModule
			{
			public:
				using MouseInputModule::onEvent;
				using MouseInputModule::updateImpl;
				using MouseInputModule::requireUpdate;
			};

			py::class_<MouseInputModule, InputModule, MouseInputModuleTrampoline,std::shared_ptr<MouseInputModule>>(m, "MouseInputModule")
				.def(py::init<>())
				.def("enqueueEvent", &MouseInputModule::enqueueEvent)
				.def("varCacheEvent", &MouseInputModule::varCacheEvent)
				// protected
				.def("onEvent", &MouseInputModulePublicist::onEvent)
				.def("updateImpl", &MouseInputModulePublicist::updateImpl)
				.def("requireUpdate", &MouseInputModulePublicist::requireUpdate);

			class OutputModulePublicist : public OutputModule
			{
			public:
				using OutputModule::updateImpl;
				using OutputModule::output;
				using OutputModule::constructFileName;
			};

			py::class_<OutputModule, Module, std::shared_ptr<OutputModule>>(m, "OutputModule")
				.def(py::init<>())
				.def("varOutputPath", &OutputModule::varOutputPath, py::return_value_policy::reference)
				.def("varPrefix", &OutputModule::varPrefix, py::return_value_policy::reference)
				.def("varStartFrame", &OutputModule::varStartFrame, py::return_value_policy::reference)
				.def("varEndFrame", &OutputModule::varEndFrame, py::return_value_policy::reference)
				.def("varStride", &OutputModule::varStride, py::return_value_policy::reference)
				.def("varReordering", &OutputModule::varReordering, py::return_value_policy::reference)
				.def("inFrameNumber", &OutputModule::inFrameNumber, py::return_value_policy::reference)
				.def("getModuleType", &OutputModule::getModuleType)
				// protected
				.def("updateImpl", &OutputModulePublicist::updateImpl)
				.def("output", &OutputModulePublicist::output)
				.def("constructFileName", &OutputModulePublicist::constructFileName);

			py::class_<DataSource, Module, std::shared_ptr<DataSource>>(m, "DataSource")
				.def(py::init<>())
				.def("captionVisible", &DataSource::captionVisible)
				.def("getModuleType", &DataSource::getModuleType);

			//pipeline
			py::class_<GraphicsPipeline, Pipeline, std::shared_ptr<GraphicsPipeline>>(m, "GraphicsPipeline", py::buffer_protocol(), py::dynamic_attr())
				.def(py::init<Node*>());

			class AnimationPipelineTrampoline : public AnimationPipeline
			{
			public:
				using AnimationPipeline::AnimationPipeline;

				bool printDebugInfo() override
				{
					PYBIND11_OVERRIDE(
						bool,
						dyno::AnimationPipeline,
						printDebugInfo
					);
				}
			};

			class AnimationPipelinePublicist : public AnimationPipeline
			{
			public:
				using AnimationPipeline::printDebugInfo;
			};

			py::class_<AnimationPipeline, Pipeline, AnimationPipelineTrampoline, std::shared_ptr<AnimationPipeline>>(m, "AnimationPipeline", py::buffer_protocol(), py::dynamic_attr())
				.def(py::init<Node*>())
				.def("printDebugInfo", &AnimationPipelinePublicist::printDebugInfo);

			class TopologyModuleTrampoline : public TopologyModule
			{
			public:
				void updateTopology() override
				{
					PYBIND11_OVERRIDE_PURE(
						void,
						dyno::TopologyModule,
						updateTopology
					);
				}
			};

			class TopologyModulePublicist : public TopologyModule
			{
			public:
				using TopologyModule::updateTopology;
			};

			py::class_<TopologyModule, OBase, TopologyModuleTrampoline, std::shared_ptr<TopologyModule>>(m, "TopologyModule")
				.def(py::init<>())
				.def("getDOF", &TopologyModule::getDOF)
				.def("tagAsChanged", &TopologyModule::tagAsChanged)
				.def("tagAsUnchanged", &TopologyModule::tagAsUnchanged)
				.def("isTopologyChanged", &TopologyModule::isTopologyChanged)
				.def("update", &TopologyModule::update)
				.def("updateTopology", &TopologyModulePublicist::updateTopology);

			class SceneGraphPublicist : public SceneGraph
			{
			public:
				using SceneGraph::updateExecutionQueue;
			};

			py::class_<SceneGraph, OBase, std::shared_ptr<SceneGraph>>SG(m, "SceneGraph");
			SG.def(py::init<>())
				.def("advance", &SceneGraph::advance)
				.def("takeOneFrame", &SceneGraph::takeOneFrame)
				.def("updateGraphicsContext", &SceneGraph::updateGraphicsContext)
				.def("run", &SceneGraph::run)
				.def("boundingBox", &SceneGraph::boundingBox)
				.def("reset", py::overload_cast<>(&SceneGraph::reset))
				.def("reset", py::overload_cast<std::shared_ptr<Node>>(&SceneGraph::reset))
				.def("printNodeInfo", &SceneGraph::printNodeInfo)
				.def("printSimulationInfo", &SceneGraph::printSimulationInfo)
				.def("printRenderingInfo", &SceneGraph::printRenderingInfo)
				.def("printValidationInfo", &SceneGraph::printValidationInfo)
				.def("isValidationInfoPrintable", &SceneGraph::isValidationInfoPrintable)
				.def("isNodeInfoPrintable", &SceneGraph::isNodeInfoPrintable)
				.def("isSimulationInfoPrintable", &SceneGraph::isSimulationInfoPrintable)
				.def("isRenderingInfoPrintable", &SceneGraph::isRenderingInfoPrintable)
				.def("load", &SceneGraph::load)
				.def("invoke", &SceneGraph::invoke)
				//createNewScene
				//.def("createNewScene", py::overload_cast<>()(&createNewScene<>))
				.def("deleteNode", &SceneGraph::deleteNode)
				.def("propagateNode", &SceneGraph::propagateNode)
				.def("isEmpty", &SceneGraph::isEmpty)
				.def("getWorkMode", &SceneGraph::getWorkMode)
				.def("getInstance", &SceneGraph::getInstance)
				.def("setTotalTime", &SceneGraph::setTotalTime)
				.def("getTotalTime", &SceneGraph::getTotalTime)
				.def("setFrameRate", &SceneGraph::setFrameRate)
				.def("getFrameRate", &SceneGraph::getFrameRate)
				.def("getTimeCostPerFrame", &SceneGraph::getTimeCostPerFrame)
				.def("getFrameInterval", &SceneGraph::getFrameInterval)
				.def("getFrameNumber", &SceneGraph::getFrameNumber)
				.def("isIntervalAdaptive", &SceneGraph::isIntervalAdaptive)
				.def("setAdaptiveInterval", &SceneGraph::setAdaptiveInterval)
				.def("setGravity", &SceneGraph::setGravity)
				.def("getGravity", &SceneGraph::getGravity)
				.def("setUpperBound", &SceneGraph::setUpperBound)
				.def("getUpperBound", &SceneGraph::getUpperBound)
				.def("setLowerBound", &SceneGraph::setLowerBound)
				.def("getLowerBound", &SceneGraph::getLowerBound)
				.def("begin", &SceneGraph::begin)
				.def("end", &SceneGraph::end)
				.def("markQueueUpdateRequired", &SceneGraph::markQueueUpdateRequired)
				.def("onMouseEvent", py::overload_cast<dyno::PMouseEvent>(&SceneGraph::onMouseEvent))
				.def("onMouseEvent", py::overload_cast<dyno::PMouseEvent, std::shared_ptr<Node>>(&SceneGraph::onMouseEvent))
				.def("onKeyboardEvent", &SceneGraph::onKeyboardEvent)
				//.def("traverse_backward", py::overload_cast<dyno::Action*>(&SceneGraph::traverseBackward))
				//.def("traverseForward", &SceneGraph::traverseForward)
				//.def("traverse_forward_with_auth_sync", &SceneGraph::traverseForwardWithAutoSync)
				.def("addNode", static_cast<std::shared_ptr<Node>(SceneGraph::*)(std::shared_ptr<Node>)>(&SceneGraph::addNode))
				.def("addNode", static_cast<std::shared_ptr<dyno::StaticMeshLoader<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::StaticMeshLoader<dyno::DataType3f>>)>(&SceneGraph::addNode))
				.def("addNode", static_cast<std::shared_ptr<dyno::VolumeBoundary<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::VolumeBoundary<dyno::DataType3f>>)>(&SceneGraph::addNode))
				.def("addNode", static_cast<std::shared_ptr<dyno::ParticleFluid<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::ParticleFluid<dyno::DataType3f>>)>(&SceneGraph::addNode))
				.def("addNode", static_cast<std::shared_ptr<dyno::PointsLoader<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::PointsLoader<dyno::DataType3f>>)>(&SceneGraph::addNode))
				.def("addNode", static_cast<std::shared_ptr<dyno::MakeParticleSystem<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::MakeParticleSystem<dyno::DataType3f>>)>(&SceneGraph::addNode))
				.def("addNode", static_cast<std::shared_ptr<dyno::GltfLoader<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::GltfLoader<dyno::DataType3f>>)>(&SceneGraph::addNode))
				.def("addNode", static_cast<std::shared_ptr<dyno::ParametricModel<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::ParametricModel<dyno::DataType3f>>)>(&SceneGraph::addNode))
				.def("addNode", static_cast<std::shared_ptr<dyno::GeometryLoader<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::GeometryLoader<dyno::DataType3f>>)>(&SceneGraph::addNode))
				// protected
				.def("updateExecutionQueue", &SceneGraphPublicist::updateExecutionQueue);

			py::enum_<typename SceneGraph::EWorkMode>(m, "EWorkMode")
				.value("EDIT_MODE", SceneGraph::EWorkMode::EDIT_MODE)
				.value("RUNNING_MODE", SceneGraph::EWorkMode::RUNNING_MODE);

			//py::class_<dyno::SceneGraphFactory>(m, "SceneGraphFactory");
				//.def("instance", &dyno::SceneGraphFactory::instance, py::return_value_policy::reference)
				//.def("active", &dyno::SceneGraphFactory::active)
				//.def("createNewScene", &dyno::SceneGraphFactory::createNewScene)
				//.def("createDefaultScene", &dyno::SceneGraphFactory::createDefaultScene)
				//.def("setDefaultCreator", &dyno::SceneGraphFactory::setDefaultCreator)
				//.def("pushScene", &dyno::SceneGraphFactory::pushScene)
				//.def("popScene", &dyno::SceneGraphFactory::popScene)
				//.def("popAllScenes", &dyno::SceneGraphFactory::popAllScenes);

			py::class_<dyno::SceneLoader>(m, "SceneLoader")
				.def("load", &dyno::SceneLoader::load)
				.def("save", &dyno::SceneLoader::save)
				.def("canLoadFileByName", &dyno::SceneLoader::canLoadFileByName)
				.def("canLoadFileByExtension", &dyno::SceneLoader::canLoadFileByExtension);

			py::class_<dyno::SceneLoaderFactory>(m, "SceneLoaderFactory")
				.def("getInstance", &dyno::SceneLoaderFactory::getInstance, py::return_value_policy::reference)
				.def("getEntryByFileExtension", &dyno::SceneLoaderFactory::getEntryByFileExtension)
				.def("getEntryByFileName", &dyno::SceneLoaderFactory::getEntryByFileName)
				.def("addEntry", &dyno::SceneLoaderFactory::addEntry)
				.def("getEntryList", &dyno::SceneLoaderFactory::getEntryList);

			py::class_<dyno::SceneLoaderXML, dyno::SceneLoader>(m, "SceneLoaderXML")
				.def("load", &dyno::SceneLoaderXML::load)
				.def("save", &dyno::SceneLoaderXML::save);

			//------------------------- New ------------------------------2024

			declare_p_enum(m);

			//declare_var<float>(m, "f");
			declare_var<bool>(m, "Bool");
			declare_var<uint>(m, "Uint");
			declare_var<int>(m, "Int");
			declare_var<Real>(m, "Real");
			//declare_var<Coord>(m, "Coord");
			declare_var<std::string>(m, "S");
			declare_var<dyno::Vec3f>(m, "3f");
			declare_var<dyno::Vec3d>(m, "3d");
			declare_var<dyno::Vec3i>(m, "3i");
			declare_var<dyno::Vec3u>(m, "3u");
			declare_var<dyno::Vec3c>(m, "3c");
			declare_var<dyno::Vec3uc>(m, "3uc");
			declare_var<dyno::TOrientedBox3D<Real>>(m, "TOrientedBox3D");
			declare_var<dyno::FilePath>(m, "FilePath");
			declare_var<dyno::Color>(m, "Color");
			//declare_var<dyno::RigidBody<dyno::DataType3f>>(m, "RigidBody3f");
			declare_var<dyno::Quat<Real>>(m, "QuatReal");
			declare_var<dyno::Curve>(m, "Curve");
			declare_var<dyno::Ramp>(m, "Ramp");
			declare_var<dyno::Key2HingeConfig>(m, "Key2HingeConfig");

			py::class_<dyno::FVar<dyno::PEnum>, FBase, std::shared_ptr<dyno::FVar<dyno::PEnum>>>(m, "FVarPEnum")
				.def(py::init<>())
				.def("size", &dyno::FVar<dyno::PEnum>::size)
				.def("setValue", &dyno::FVar<dyno::PEnum>::setValue)
				.def("getValue", &dyno::FVar<dyno::PEnum>::getValue)
				.def("currentKey", &dyno::FVar<dyno::PEnum>::currentKey)
				.def("setCurrentKey", &dyno::FVar<dyno::PEnum>::setCurrentKey)
				.def("serialize", &dyno::FVar<dyno::PEnum>::serialize)
				//.def("deserialize", &dyno::FVar<dyno::PEnum>::deserialize)
				.def("isEmpty", &dyno::FVar<dyno::PEnum>::isEmpty)
				.def("getDataPtr", &dyno::FVar<dyno::PEnum>::getDataPtr);

			declare_farray<int, DeviceType::GPU>(m, "1D");
			declare_farray<float, DeviceType::GPU>(m, "1fD");
			declare_farray<Vec3f, DeviceType::GPU>(m, "3fD");
			declare_farray<CollisionMask, DeviceType::GPU>(m, "CollisionMask");
			declare_farray<dyno::TContactPair<float>, DeviceType::GPU>(m, "TContactPair");
			declare_farray<dyno::Attribute, DeviceType::GPU>(m, "Attribute");

			declare_array_list<int, DeviceType::GPU>(m, "1D");
			declare_array_list<float, DeviceType::GPU>(m, "1fD");
			declare_array_list<Vec3f, DeviceType::GPU>(m, "3fD");
			declare_array_list<dyno::TBond<dyno::DataType3f>, DeviceType::GPU>(m, "TBondD3f");

			declare_instance<TopologyModule>(m, "");
			declare_instance<dyno::PointSet<dyno::DataType3f>>(m, "PointSet3f");
			declare_instance<dyno::EdgeSet<dyno::DataType3f>>(m, "EdgeSet3f");
			declare_instance<dyno::TriangleSet<dyno::DataType3f>>(m, "TriangleSet3f");
			declare_instance<dyno::DiscreteElements<dyno::DataType3f>>(m, "DiscreteElements3f");
			declare_instance<dyno::HeightField<dyno::DataType3f>>(m, "HeightField3f");
			declare_instance<dyno::TextureMesh>(m, "TextureMesh");
			declare_instance<dyno::LevelSet<dyno::DataType3f>>(m, "LevelSet3f");
			declare_instance<dyno::QuadSet<dyno::DataType3f>>(m, "QuadSet3f");
			declare_instance<dyno::PolygonSet<dyno::DataType3f>>(m, "PolygonSet3f");

			declare_instances<TopologyModule>(m, "");
			declare_instances<dyno::PointSet<dyno::DataType3f>>(m, "PointSet3f");
			declare_instances<dyno::EdgeSet<dyno::DataType3f>>(m, "EdgeSet3f");
			declare_instances<dyno::TriangleSet<dyno::DataType3f>>(m, "TriangleSet3f");
			declare_instances<dyno::DiscreteElements<dyno::DataType3f>>(m, "DiscreteElements3f");
			declare_instances<dyno::HeightField<dyno::DataType3f>>(m, "HeightField3f");
			declare_instances<dyno::TextureMesh>(m, "TextureMesh");
			declare_instances<dyno::LevelSet<dyno::DataType3f>>(m, "LevelSet3f");
			declare_instances<dyno::QuadSet<dyno::DataType3f>>(m, "QuadSet3f");
			declare_instances<dyno::PolygonSet<dyno::DataType3f>>(m, "PolygonSet3f");

			// New
			declare_parametric_model<dyno::DataType3f>(m, "3f");
			//import
			declare_multi_node_port<dyno::ParticleEmitter<dyno::DataType3f>>(m, "ParticleEmitter3f");
			declare_multi_node_port<dyno::ParticleSystem<dyno::DataType3f>>(m, "ParticleSystem3f");
			declare_multi_node_port<dyno::TriangularSystem<dyno::DataType3f>>(m, "TriangularSystem3f");
			declare_multi_node_port<dyno::CapillaryWave<dyno::DataType3f>>(m, "CapillaryWave3f");
			declare_multi_node_port<dyno::Volume<dyno::DataType3f>>(m, "Volume3f");
			declare_multi_node_port<dyno::RigidBodySystem<dyno::DataType3f>>(m, "RigidBodySystem3f");
			declare_multi_node_port<dyno::Vessel<dyno::DataType3f>>(m, "Vessel3f");

			declare_single_node_port<dyno::OceanBase<dyno::DataType3f>>(m, "OceanBase3f");
			declare_single_node_port<dyno::Ocean<dyno::DataType3f>>(m, "Ocean3f");
			declare_single_node_port<dyno::OceanPatch<dyno::DataType3f>>(m, "OceanPatch3f");
			declare_single_node_port<dyno::GranularMedia<dyno::DataType3f>>(m, "GranularMedia3f");
			declare_single_node_port<dyno::BasicShape<dyno::DataType3f>>(m, "BasicShape3f");
			declare_single_node_port<dyno::RigidBodySystem<dyno::DataType3f>>(m, "RigidBodySystem3f");
			declare_single_node_port<dyno::Vessel<dyno::DataType3f>>(m, "Vessel3f");
			declare_single_node_port<dyno::GhostParticles<dyno::DataType3f>>(m, "GhostParticles3f");
			declare_single_node_port<dyno::ParticleSystem<dyno::DataType3f>>(m, "ParticleSystem3f");
			//declare_semi_analytical_sfi_node<dyno::DataType3f>(m, "3f");

			declare_floating_number<dyno::DataType3f>(m, "3f");

			declare_camera(m);
			declare_act_node_info(m);
			declare_post_processing(m);
			declare_reset_act(m);
			declare_add_real_and_real<dyno::DataType3f>(m, "3f");
			declare_divide_real_and_real<dyno::DataType3f>(m, "3f");
			declare_multiply_real_and_real<dyno::DataType3f>(m, "3f");
			declare_subtract_real_and_real<dyno::DataType3f>(m, "3f");

			declare_vehicle_rigid_body_info(m);
			declare_vehicle_joint_info(m);
			declare_vehicle_bind(m);
			declare_animation_2_joint_config(m);
			declare_hinge_action(m);
			declare_key_2_hinge_config(m);

			declare_f_list<dyno::DataType3f>(m, "3f");
}