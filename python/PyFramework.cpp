#include "PyFramework.h"

#include "RigidBody/RigidBodySystem.h"
#include "HeightField/Vessel.h"

#include "Camera.h"
void declare_camera(py::module& m)
{
	using Class = dyno::Camera;
	std::string pyclass_name = std::string("TopologyMapping");
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
void declare_reset_act(py::module& m)
{
	using Class = dyno::ResetAct;
	using Parent = dyno::Action;
	std::string pyclass_name = std::string("ResetAct");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str())
		.def(py::init<>());
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
		.def("set_scnen_graph", &Node::setSceneGraph)
		.def("getSceneGraph", &Node::getSceneGraph, py::return_value_policy::reference)
		.def("getImportNodes", &Node::getImportNodes)
		.def("getExportNodes", &Node::getExportNodes)
		.def("add_module", static_cast<bool(Node::*)(std::shared_ptr<Module>)>(&Node::addModule))
		.def("delete_module", static_cast<bool(Node::*)(std::shared_ptr<Module>)>(&Node::deleteModule))
		.def("getModuleList", &Node::getModuleList)
		.def("hasModule", &Node::hasModule)
		.def("get_module", static_cast<std::shared_ptr<Module>(Node::*)(std::string)> (&Node::getModule))
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
		.def("stateFrameNumber", &Node::stateFrameNumber, py::return_value_policy::reference);

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

	py::class_<dyno::Action, std::shared_ptr<dyno::Action>>(m, "Action")
		.def(py::init<>())
		.def("start", &dyno::Action::start)
		.def("process", &dyno::Action::process)
		.def("end", &dyno::Action::end);

	py::class_<dyno::TimeStamp, std::shared_ptr<dyno::TimeStamp>>(m, "TimeStamp")
		.def(py::init<>())
		.def("mark", &dyno::TimeStamp::mark);

	py::class_<dyno::DirectedAcyclicGraph, std::shared_ptr<dyno::DirectedAcyclicGraph>>(m, "DirectedAcyclicGraph")
		.def(py::init<>())
		.def("add_edge", &dyno::DirectedAcyclicGraph::addEdge)
		//.def("add_edge", &dyno::DirectedAcyclicGraph::topologicalSort)
		//.def("add_edge", &dyno::DirectedAcyclicGraph::topologicalSortUtil)
		.def("sizeOfVertex", &dyno::DirectedAcyclicGraph::sizeOfVertex)
		.def("OtherVerticesSize", &dyno::DirectedAcyclicGraph::OtherVerticesSize)
		.def("getOtherVertices", &dyno::DirectedAcyclicGraph::getOtherVertices)
		.def("vertices", &dyno::DirectedAcyclicGraph::vertices)
		.def("edges", &dyno::DirectedAcyclicGraph::edges, py::return_value_policy::reference)
		.def("reverse_edges", &dyno::DirectedAcyclicGraph::reverseEdges, py::return_value_policy::reference)
		.def("addOtherVertices", &dyno::DirectedAcyclicGraph::addOtherVertices)
		.def("addtoRemoveList", &dyno::DirectedAcyclicGraph::addtoRemoveList)
		.def("remove_id", &dyno::DirectedAcyclicGraph::removeID);

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

	py::class_<dyno::PluginEntry, std::shared_ptr<dyno::PluginEntry >>(m, "PluginEntry")
		.def(py::init<>())
		.def("name", &dyno::PluginEntry::name)
		.def("version", &dyno::PluginEntry::version)
		.def("description", &dyno::PluginEntry::description)
		.def("setName", &dyno::PluginEntry::setName, py::arg("pluginName"))
		.def("setVersion", &dyno::PluginEntry::setVersion, py::arg("pluginVersion"))
		.def("setDescription", &dyno::PluginEntry::setDescription, py::arg("desc"))
		.def("initialize", &dyno::PluginEntry::initialize);

	py::enum_<typename dyno::NodePortType>(m, "NodePortType")
		.value("Single", dyno::NodePortType::Single)
		.value("Multiple", dyno::NodePortType::Multiple)
		.value("Unknown", dyno::NodePortType::Unknown);

	py::class_<NodePort, std::shared_ptr<dyno::NodePort>>(m, "NodePort")
		//.def(py::init<>())
		.def("getPortName", &NodePort::getPortName)
		.def("getPortType", &NodePort::getPortType)
		.def("setPortType", &NodePort::setPortType)
		.def("getNodes", &NodePort::getNodes)
		.def("isKindOf", &NodePort::isKindOf)
		.def("hasNode", &NodePort::hasNode)
		.def("getParent", &NodePort::getParent, py::return_value_policy::reference)
		.def("attach", &NodePort::attach);

	py::enum_<typename dyno::FieldTypeEnum>(m, "FieldTypeEnum")
		.value("In", dyno::FieldTypeEnum::In)
		.value("Out", dyno::FieldTypeEnum::Out)
		.value("IO", dyno::FieldTypeEnum::IO)
		.value("Param", dyno::FieldTypeEnum::Param)
		.value("State", dyno::FieldTypeEnum::State)
		.value("Next", dyno::FieldTypeEnum::Next);

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
		.def("detach", &FBase::detach);

	py::class_<dyno::FCallBackFunc, std::shared_ptr<dyno::FCallBackFunc>>(m, "FCallBackFunc")
		.def("update", &dyno::FCallBackFunc::update)
		.def("add_inputA", &dyno::FCallBackFunc::addInput);

	py::class_<Color>(m, "Color")
		.def(py::init<float, float, float>());

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

	class PyModule : public Module
	{
	public:
		using Module::Module;

		void updateImpl() override
		{
			PYBIND11_OVERRIDE(void, Module, updateImpl);
		}

		bool initializeImpl() override
		{
			PYBIND11_OVERRIDE(bool, Module, initializeImpl);
		}
	};

	//module
	py::class_<Module, PyModule, OBase, std::shared_ptr<Module>>(m, "Module")
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
		.def("updateImpl", &Module::updateImpl);

	//py::class_<DebugInfo, Module, std::shared_ptr<DebugInfo>>(m, "DebugInfo")
	//	.def("print", &DebugInfo::print)
	//	.def("varPrefix", &DebugInfo::varPrefix)
	//	.def("getModuleType", &DebugInfo::getModuleType);

	//py::class_<PrintInt, Module, std::shared_ptr<PrintInt>>(m, "PrintInt")
	//	.def("print", &PrintInt::print)
	//	.def("inInt", &PrintInt::inInt);

	//py::class_<PrintUnsigned, Module, std::shared_ptr<PrintUnsigned>>(m, "PrintUnsigned")
	//	.def("print", &PrintUnsigned::print)
	//	.def("inUnsigned", &PrintUnsigned::inUnsigned);

	//py::class_<PrintFloat, Module, std::shared_ptr<PrintFloat>>(m, "PrintFloat")
	//	.def("print", &PrintFloat::print)
	//	.def("inFloat", &PrintFloat::inFloat);

	//py::class_<PrintVector, Module, std::shared_ptr<PrintVector>>(m, "PrintVector")
	//	.def("print", &PrintVector::print)
	//	.def("inVector", &PrintVector::inVector);

	py::class_<VisualModule, Module, std::shared_ptr<VisualModule>>(m, "VisualModule")
		.def(py::init<>())
		.def("setVisible", &VisualModule::setVisible)
		.def("isVisible", &VisualModule::isVisible)
		.def("getModuleType", &VisualModule::getModuleType)
		.def("varVisible", &VisualModule::varVisible, py::return_value_policy::reference);

	py::class_<ComputeModule, Module, std::shared_ptr<ComputeModule>>(m, "ComputeModule")
		.def("get_module_type", &dyno::ComputeModule::getModuleType);

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

	py::class_<Pipeline, Module, std::shared_ptr<Pipeline>>(m, "Pipeline")
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
		.def("demoteOutputFromNode", &Pipeline::demoteOutputFromNode);

	py::class_<ConstraintModule, Module, std::shared_ptr<ConstraintModule>>(m, "ConstraintModule")
		.def("constrain", &dyno::ConstraintModule::constrain)
		.def("get_module_type", &dyno::ConstraintModule::getModuleType);

	py::class_<GroupModule, Module, std::shared_ptr<GroupModule>>(m, "GroupModule")
		.def("pushModule", &GroupModule::pushModule)
		.def("moduleList", &GroupModule::moduleList, py::return_value_policy::reference)
		.def("setParentNode", &GroupModule::setParentNode);

	py::class_<TopologyMappingdyno, Module, std::shared_ptr< TopologyMappingdyno>>(m, "TopologyMappingdyno");

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

			py::class_<KeyboardInputModule, InputModule, std::shared_ptr<KeyboardInputModule>>(m, "KeyboardInputModule")
				.def("enqueueEvent", &KeyboardInputModule::enqueueEvent)
				.def("varCacheEvent", &KeyboardInputModule::varCacheEvent);

			class PyMouseInputModule : public MouseInputModule
			{
			public:
				using MouseInputModule::MouseInputModule;

				void onEvent(dyno::PMouseEvent event) override
				{
					PYBIND11_OVERRIDE(void, MouseInputModule, onEvent, event);
				}

				void updateImpl()
				{
					PYBIND11_OVERRIDE(void, MouseInputModule, updateImpl);
				}
			};

			py::class_<MouseInputModule, PyMouseInputModule, InputModule, std::shared_ptr<MouseInputModule>>(m, "MouseInputModule")
				.def(py::init<>())
				.def("enqueueEvent", &MouseInputModule::enqueueEvent)
				.def("varCacheEvent", &MouseInputModule::varCacheEvent)
				.def("onEvent", &MouseInputModule::onEvent)
				.def("updateImpl", &MouseInputModule::updateImpl)
				.def("requireUpdate", &MouseInputModule::requireUpdate);

			py::class_<OutputModule, Module, std::shared_ptr<OutputModule>>(m, "OutputModule")
				.def(py::init<>())
				.def("varOutputPath", &OutputModule::varOutputPath, py::return_value_policy::reference)
				.def("varPrefix", &OutputModule::varPrefix, py::return_value_policy::reference)
				.def("varStartFrame", &OutputModule::varStartFrame, py::return_value_policy::reference)
				.def("varEndFrame", &OutputModule::varEndFrame, py::return_value_policy::reference)
				.def("varStride", &OutputModule::varStride, py::return_value_policy::reference)
				.def("varReordering", &OutputModule::varReordering, py::return_value_policy::reference)
				.def("inFrameNumber", &OutputModule::inFrameNumber, py::return_value_policy::reference)
				.def("getModuleType", &OutputModule::getModuleType);

			py::class_<DataSource, Module, std::shared_ptr<DataSource>>(m, "DataSource")
				.def(py::init<>())
				.def("captionVisible", &DataSource::captionVisible)
				.def("getModuleType", &DataSource::getModuleType);

			//pipeline
			py::class_<GraphicsPipeline, Pipeline, std::shared_ptr<GraphicsPipeline>>(m, "GraphicsPipeline", py::buffer_protocol(), py::dynamic_attr())
				.def(py::init<Node*>());

			py::class_<AnimationPipeline, Pipeline, std::shared_ptr<AnimationPipeline>>(m, "AnimationPipeline", py::buffer_protocol(), py::dynamic_attr())
				.def(py::init<Node*>());

			py::class_<TopologyModule, OBase, std::shared_ptr<TopologyModule>>(m, "TopologyModule")
				.def(py::init<>())
				.def("getDOF", &TopologyModule::getDOF)
				.def("tagAsChanged", &TopologyModule::tagAsChanged)
				.def("tagAsUnchanged", &TopologyModule::tagAsUnchanged)
				.def("isTopologyChanged", &TopologyModule::isTopologyChanged)
				.def("update", &TopologyModule::update);

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
				.def("addNode", static_cast<std::shared_ptr<dyno::GeometryLoader<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::GeometryLoader<dyno::DataType3f>>)>(&SceneGraph::addNode));

			py::enum_<typename SceneGraph::EWorkMode>(m, "EWorkMode")
				.value("EDIT_MODE", SceneGraph::EWorkMode::EDIT_MODE)
				.value("RUNNING_MODE", SceneGraph::EWorkMode::RUNNING_MODE);

			//py::class_<dyno::SceneGraphFactory>(m, "SceneGraphFactory");
			//.def("instance", &dyno::SceneGraphFactory::instance, py::return_value_policy::reference)
			//.def("active", &dyno::SceneGraphFactory::active)
			//.def("create_new_scene", &dyno::SceneGraphFactory::createNewScene)
			//.def("pop_scnene", &dyno::SceneGraphFactory::popScene)
			//.def("pop_all_scene", &dyno::SceneGraphFactory::popAllScenes);

			py::class_<dyno::SceneLoader>(m, "SceneLoader")
				.def("load", &dyno::SceneLoader::load)
				.def("save", &dyno::SceneLoader::save)
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

			declare_var<float>(m, "f");
			declare_var<bool>(m, "b");
			declare_var<uint>(m, "uint");
			declare_var<int>(m, "int");
			declare_var<std::string>(m, "s");
			declare_var<dyno::Vec3f>(m, "3f");
			declare_var<dyno::Vec3d>(m, "3d");
			declare_var<dyno::Vec3i>(m, "3i");
			declare_var<dyno::Vec3u>(m, "3u");
			declare_var<dyno::Vec3c>(m, "3c");
			declare_var<dyno::Vec3uc>(m, "3uc");
			declare_var<dyno::TOrientedBox3D<Real>>(m, "TOrientedBox3D");
			declare_var<dyno::FilePath>(m, "FilePath");
			declare_var<dyno::Color>(m, "Color");
			declare_var<dyno::RigidBody<dyno::DataType3f>>(m, "RigidBody3f");
			declare_var<dyno::Quat<float>>(m, "QuatFloat");
			declare_var<dyno::Curve>(m, "Curve");
			declare_var<dyno::Ramp>(m, "Ramp");

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
			declare_single_node_port<dyno::VolumeOctree<dyno::DataType3f>>(m, "VolumeOctree3f");
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
}