#include "PyFramework.h"

#include "Camera.h"
void declare_camera(py::module& m)
{
	using Class = dyno::Camera;
	std::string pyclass_name = std::string("TopologyMapping");
	py::class_<Class, std::shared_ptr<Class>>camera(m, pyclass_name.c_str());
	camera.def("get_view_mat", &Class::getViewMat) // 绑定 getViewMat 方法
		.def("get_proj_mat", &Class::getProjMat) // 绑定 getProjMat 方法
		.def("rotate_to_point", &Class::rotateToPoint) // 绑定 rotateToPoint 方法
		.def("translate_to_point", &Class::translateToPoint) // 绑定 translateToPoint 方法
		.def("zoom", &Class::zoom) // 绑定 zoom 方法
		.def("register_point", &Class::registerPoint) // 绑定 registerPoint 方法
		.def("set_width", &Class::setWidth) // 绑定 setWidth 方法
		.def("set_height", &Class::setHeight) // 绑定 setHeight 方法
		.def("set_clip_near", &Class::setClipNear) // 绑定 setClipNear 方法
		.def("set_clip_far", &Class::setClipFar) // 绑定 setClipFar 方法
		.def("viewport_width", &Class::viewportWidth) // 绑定 viewportWidth 方法
		.def("viewport_height", &Class::viewportHeight) // 绑定 viewportHeight 方法
		.def("clip_near", &Class::clipNear) // 绑定 clipNear 方法
		.def("clip_far", &Class::clipFar) // 绑定 clipFar 方法
		.def("set_eye_pos", &Class::setEyePos) // 绑定 setEyePos 方法
		.def("set_target_pos", &Class::setTargetPos) // 绑定 setTargetPos 方法
		.def("get_eye_pos", &Class::getEyePos) // 绑定 getEyePos 方法
		.def("get_target_pos", &Class::getTargetPos) // 绑定 getTargetPos 方法
		.def("cast_ray_in_world_space", &Class::castRayInWorldSpace) // 绑定 castRayInWorldSpace 方法
		.def("set_unit_scale", &Class::setUnitScale) // 绑定 setUnitScale 方法
		.def("unit_scale", &Class::unitScale) // 绑定 unitScale 方法
		.def("set_projection_type", &Class::setProjectionType) // 绑定 setProjectionType 方法
		.def("projection_type", &Class::projectionType); // 绑定 projectionType 方法;

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
		.def("current_key", &Class::currentKey)
		.def("current_string", &Class::currentString)
		.def("set_current_key", &Class::setCurrentKey)
		.def("enum_map", &Class::enumMap);
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



void pybind_log(py::module& m)
{

	//py::class_<Log>LOG(m, "Log");
	//LOG.def(py::init<>())
	//	.def("instance", &Log::instance, py::return_value_policy::reference)
	//	.def("send_message", &Log::sendMessage)
	//	.def("set_user_receiver", &Log::setUserReceiver)
	//	.def("set_level", &Log::setLevel)
	//	.def("set_output", &Log::setOutput)
	//	.def("get_output", &Log::getOutput, py::return_value_policy::reference);

	//py::enum_<typename Log::MessageType>(LOG, "MessageType")
	//	.value("DebugInfo", Log::DebugInfo)
	//	.value("Info", Log::Info)
	//	.value("Warning", Log::Warning)
	//	.value("Error", Log::Error)
	//	.value("User", Log::User);

}

void pybind_framework(py::module& m)
{
	pybind_log(m);

	py::class_<dyno::ClassInfo, std::shared_ptr<dyno::ClassInfo>>(m, "ClassInfo")
		.def("create_object", &dyno::ClassInfo::createObject, py::return_value_policy::reference)
		.def("is_dynamic", &dyno::ClassInfo::isDynamic)
		.def("get_class_name", &dyno::ClassInfo::getClassName)
		.def("get_constructor", &dyno::ClassInfo::getConstructor);



	//basic
	py::class_<Object, std::shared_ptr<Object>>(m, "Object")
		.def(py::init<>())
		.def("register_class", &Object::registerClass)
		//.def("create_object", &Object::createObject, py::return_value_policy::reference)
		.def("get_class_map", &Object::getClassMap, py::return_value_policy::reference)
		.def("base_id", &Object::baseId)
		.def("object_id", &Object::objectId);

	py::class_<Node, std::shared_ptr<Node>>(m, "Node", py::buffer_protocol(), py::dynamic_attr())
		.def("set_name", &Node::setName)
		.def("get_name", &Node::getName)
		.def("get_node_type", &Node::getNodeType)
		.def("is_auto_sync", &Node::isAutoSync)
		.def("set_auto_sync", &Node::setAutoSync)
		.def("is_active", &Node::isActive)
		.def("set_active", &Node::setActive)
		.def("is_visible", &Node::isVisible)
		.def("set_visible", &Node::setVisible)
		.def("get_dt", &Node::getDt)
		.def("set_dt", &Node::setDt)
		.def("set_scnen_graph", &Node::setSceneGraph)
		.def("get_scene_graph", &Node::getSceneGraph, py::return_value_policy::reference)
		.def("get_import_nodes", &Node::getImportNodes)
		.def("get_export_nodes", &Node::getExportNodes)
		.def("add_module", static_cast<bool(Node::*)(std::shared_ptr<Module>)>(&Node::addModule))
		.def("delete_module", static_cast<bool(Node::*)(std::shared_ptr<Module>)>(&Node::deleteModule))
		.def("get_module_list", &Node::getModuleList)
		.def("has_module", &Node::hasModule)
		.def("get_module", static_cast<std::shared_ptr<Module>(Node::*)(std::string)> (&Node::getModule))
		.def("reset_pipeline", &Node::resetPipeline)
		.def("graphics_pipeline", &Node::graphicsPipeline)
		.def("animation_pipeline", &Node::animationPipeline)
		.def("update", &Node::update)
		.def("update_graphics_context", &Node::updateGraphicsContext)
		.def("reset", &Node::reset)
		.def("bounding_box", &Node::boundingBox)
		.def("connect", &Node::connect)
		.def("disconnect", &Node::disconnect)
		.def("attach_field", &Node::attachField)
		.def("get_all_node_ports", &Node::getAllNodePorts)
		.def("size_of_node_ports", &Node::sizeOfNodePorts)
		.def("size_of_import_nodes", &Node::sizeOfImportNodes)
		.def("size_of_export_nodes", &Node::sizeOfExportNodes)
		.def("state_elapsed_time", &Node::stateElapsedTime, py::return_value_policy::reference)
		.def("state_time_step", &Node::stateTimeStep, py::return_value_policy::reference)
		.def("state_frame_number", &Node::stateFrameNumber, py::return_value_policy::reference);

	py::class_<dyno::NBoundingBox>(m, "NBoundingBox")
		.def(py::init<>())
		.def(py::init<dyno::Vec3f, dyno::Vec3f>())
		.def_readwrite("lower", &dyno::NBoundingBox::lower)
		.def_readwrite("upper", &dyno::NBoundingBox::upper)
		.def("join", &dyno::NBoundingBox::join, py::return_value_policy::reference)
		.def("intersect", &dyno::NBoundingBox::intersect, py::return_value_policy::reference)
		.def("max_length", &dyno::NBoundingBox::maxLength);

	py::class_<dyno::NodeAction, std::shared_ptr<dyno::NodeAction>>(m, "NodeAction")
		.def("icon", &dyno::NodeAction::icon)
		.def("caption", &dyno::NodeAction::caption)
		.def("action", &dyno::NodeAction::action);

	py::class_<dyno::NodeGroup, std::shared_ptr<dyno::NodeGroup>>(m, "NodeGroup")
		//.def("add_action", &dyno::NodeAction::addAction)
		.def("actions", &dyno::NodeGroup::actions, py::return_value_policy::reference)
		.def("caption", &dyno::NodeGroup::caption);

	py::class_<dyno::NodePage, std::shared_ptr<dyno::NodePage>>(m, "NodePage")
		.def("add_group", &dyno::NodePage::addGroup)
		.def("add_group", &dyno::NodePage::hasGroup)
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

	py::class_<dyno::Canvas, std::shared_ptr<dyno::Canvas>>(m, "Canvas")
		.def(py::init<>());

	py::class_<dyno::TimeStamp, std::shared_ptr<dyno::TimeStamp>>(m, "TimeStamp")
		.def(py::init<>())
		.def("mark", &dyno::TimeStamp::mark);

	py::class_<dyno::DirectedAcyclicGraph, std::shared_ptr<dyno::DirectedAcyclicGraph>>(m, "DirectedAcyclicGraph")
		.def(py::init<>())
		.def("add_edge", &dyno::DirectedAcyclicGraph::addEdge)
		//.def("add_edge", &dyno::DirectedAcyclicGraph::topologicalSort)
		//.def("add_edge", &dyno::DirectedAcyclicGraph::topologicalSortUtil)
		.def("size_of_vertex", &dyno::DirectedAcyclicGraph::sizeOfVertex)
		.def("other_vertices_size", &dyno::DirectedAcyclicGraph::OtherVerticesSize)
		.def("get_other_vertices", &dyno::DirectedAcyclicGraph::getOtherVertices)
		.def("vertices", &dyno::DirectedAcyclicGraph::vertices)
		.def("edges", &dyno::DirectedAcyclicGraph::edges, py::return_value_policy::reference)
		.def("reverse_edges", &dyno::DirectedAcyclicGraph::reverseEdges, py::return_value_policy::reference)
		.def("add_other_vertices", &dyno::DirectedAcyclicGraph::addOtherVertices)
		.def("add_to_remove_list", &dyno::DirectedAcyclicGraph::addtoRemoveList)
		.def("remove_id", &dyno::DirectedAcyclicGraph::removeID);

	py::class_<dyno::Plugin>(m, "Plugin")
		.def(py::init<>())
		.def("get_info", &dyno::Plugin::getInfo, py::return_value_policy::reference)
		.def("is_loaded", &dyno::Plugin::isLoaded)
		.def("unload", &dyno::Plugin::unload)
		.def("load", &dyno::Plugin::load);

	py::class_<dyno::PluginManager>(m, "PluginManager")
		.def("instance", &dyno::PluginManager::instance, py::return_value_policy::reference)
		.def("get_extension", &dyno::PluginManager::getExtension)
		.def("load_plugin", &dyno::PluginManager::loadPlugin)
		.def("load_plugin_by_path", &dyno::PluginManager::loadPluginByPath)
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
		.def("get_port_name", &NodePort::getPortName)
		.def("get_port_type", &NodePort::getPortType)
		.def("set_port_type", &NodePort::setPortType)
		.def("get_nodes", &NodePort::getNodes)
		.def("is_kind_of", &NodePort::isKindOf)
		.def("had_node", &NodePort::hasNode)
		.def("get_parent", &NodePort::getParent, py::return_value_policy::reference)
		.def("attach", &NodePort::attach);

	py::enum_<typename dyno::FieldTypeEnum>(m, "FieldTypeEnum")
		.value("In", dyno::FieldTypeEnum::In)
		.value("Out", dyno::FieldTypeEnum::Out)
		.value("IO", dyno::FieldTypeEnum::IO)
		.value("Param", dyno::FieldTypeEnum::Param)
		.value("State", dyno::FieldTypeEnum::State)
		.value("Next", dyno::FieldTypeEnum::Next);

	py::class_<FBase, std::shared_ptr<FBase>>(m, "FBase")
		.def("get_template_name", &FBase::getTemplateName)
		.def("get_class_name", &FBase::getClassName)
		.def("get_object_name", &FBase::getObjectName)
		.def("get_description", &FBase::getDescription)
		.def("get_device_type", &FBase::getDeviceType)
		.def("set_object_name", &FBase::setObjectName)
		.def("set_parent", &FBase::setParent)
		.def("is_derived", &FBase::isDerived)
		.def("is_auto_destroyable", &FBase::isAutoDestroyable)
		.def("set_auto_destroy", &FBase::setAutoDestroy)
		.def("set_derived", &FBase::setDerived)
		.def("is_modified", &FBase::isModified)
		.def("tick", &FBase::tick)
		.def("tack", &FBase::tack)
		.def("is_optional", &FBase::isOptional)
		.def("tag_optional", &FBase::tagOptional)
		.def("get_min", &FBase::getMin)
		.def("set_min", &FBase::setMin)
		.def("get_max", &FBase::getMax)
		.def("set_max", &FBase::setMax)
		.def("set_range", &FBase::setRange)
		.def("connect", &FBase::connect)
		.def("disconnect", &FBase::disconnect)
		.def("serialize", &FBase::serialize)
		//.def("deserialize", &FBase::deserialize)
		.def("get_top_field", &FBase::getTopField)
		.def("get_source", &FBase::getSource)
		.def("promote_output", &FBase::promoteOuput)
		.def("promote_input", &FBase::promoteInput)
		.def("demote_ouput", &FBase::demoteOuput)
		.def("demote_input", &FBase::demoteInput)
		.def("is_empty", &FBase::isEmpty)
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
		.def("can_be_connected_by", &InstanceBase::canBeConnectedBy)
		.def("set_object_pointer", &InstanceBase::setObjectPointer)
		.def("object_pointer", &InstanceBase::objectPointer)
		.def("standard_object_pointer", &InstanceBase::standardObjectPointer)
		.def("class_name", &InstanceBase::className);

	//module
	py::class_<Module, std::shared_ptr<Module>>(m, "Module")
		.def(py::init<>())
		.def("initialize", &Module::initialize)
		.def("update", &Module::update)
		.def("set_name", &Module::initialize)
		.def("get_name", &Module::initialize)
		.def("set_parent_node", &Module::initialize)
		.def("get_parent_node", &Module::initialize)
		.def("get_scene_graph", &Module::initialize)
		.def("is_initialized", &Module::isInitialized)
		.def("get_module_type", &Module::getModuleType)
		.def("attach_field", &Module::attachField)
		.def("is_input_complete", &Module::isInputComplete)
		.def("is_output_complete", &Module::isOutputCompete)
		.def("var_force_update", &Module::varForceUpdate, py::return_value_policy::reference)
		.def("set_update_always", &Module::setUpdateAlways);

	py::class_<DebugInfo, Module, std::shared_ptr<DebugInfo>>(m, "DebugInfo")
		.def("print", &DebugInfo::print)
		.def("var_prefix", &DebugInfo::varPrefix)
		.def("get_module_type", &DebugInfo::getModuleType);

	py::class_<PrintInt, Module, std::shared_ptr<PrintInt>>(m, "PrintInt")
		.def("print", &PrintInt::print)
		.def("in_int", &PrintInt::inInt);

	py::class_<PrintUnsigned, Module, std::shared_ptr<PrintUnsigned>>(m, "PrintUnsigned")
		.def("print", &PrintUnsigned::print)
		.def("in_unsigned", &PrintUnsigned::inUnsigned);

	py::class_<PrintFloat, Module, std::shared_ptr<PrintFloat>>(m, "PrintFloat")
		.def("print", &PrintFloat::print)
		.def("in_float", &PrintFloat::inFloat);

	py::class_<PrintVector, Module, std::shared_ptr<PrintVector>>(m, "PrintVector")
		.def("print", &PrintVector::print)
		.def("in_vector", &PrintVector::inVector);

	py::class_<VisualModule, Module, std::shared_ptr<VisualModule>>(m, "VisualModule")
		.def(py::init<>())
		.def("set_visible", &VisualModule::setVisible)
		.def("is_visible", &VisualModule::isVisible)
		.def("get_module_type", &VisualModule::getModuleType);

	py::class_<ComputeModule, Module, std::shared_ptr<ComputeModule>>(m, "ComputeModule")
		.def("get_module_type", &dyno::ComputeModule::getModuleType);

	py::class_<Add, ComputeModule, std::shared_ptr<Add>>(m, "Add")
		.def("caption", &Add::caption)
		.def("get_module_type", &Add::getModuleType);

	py::class_<Divide, ComputeModule, std::shared_ptr<Divide>>(m, "Divide")
		.def("caption", &Divide::caption)
		.def("get_module_type", &Divide::getModuleType);

	py::class_<Multiply, ComputeModule, std::shared_ptr<Multiply>>(m, "Multiply")
		.def("caption", &Multiply::caption)
		.def("get_module_type", &Multiply::getModuleType);

	py::class_<Subtract, ComputeModule, std::shared_ptr<Subtract>>(m, "Subtract")
		.def("caption", &Subtract::caption)
		.def("get_module_type", &Subtract::getModuleType);

	py::class_<Pipeline, Module, std::shared_ptr<Pipeline>>(m, "Pipeline")
		.def("size_of_dynamic_modules", &Pipeline::sizeOfDynamicModules)
		.def("size_of_persistent_modules", &Pipeline::sizeOfPersistentModules)
		.def("push_module", &Pipeline::pushModule)
		.def("pop_module", &Pipeline::popModule)
		.def("create_modules", static_cast<std::shared_ptr<Module>(Pipeline::*)()>(&Pipeline::createModule))
		.def("find_first_module", static_cast<std::shared_ptr<Module>(Pipeline::*)()>(&Pipeline::findFirstModule))
		.def("clear", &Pipeline::clear)
		.def("push_persiistent_module", &Pipeline::pushPersistentModule)
		.def("active_moduiles", &Pipeline::activeModules, py::return_value_policy::reference)
		.def("all_modules", &Pipeline::allModules, py::return_value_policy::reference)
		.def("enable", &Pipeline::enable)
		.def("disable", &Pipeline::disable)
		.def("update_execution_queue", &Pipeline::updateExecutionQueue)
		.def("force_update", &Pipeline::forceUpdate)
		.def("promote_putput_to_node", &Pipeline::promoteOutputToNode, py::return_value_policy::reference)
		.def("demote_output_from_node", &Pipeline::demoteOutputFromNode);

	py::class_<ConstraintModule, Module, std::shared_ptr<ConstraintModule>>(m, "ConstraintModule")
		.def("constrain", &dyno::ConstraintModule::constrain)
		.def("get_module_type", &dyno::ConstraintModule::getModuleType);

	py::class_<GroupModule, Module, std::shared_ptr<GroupModule>>(m, "GroupModule")
		.def("push_module", &GroupModule::pushModule)
		.def("module_list", &GroupModule::moduleList, py::return_value_policy::reference)
		.def("set_parent_node", &GroupModule::setParentNode);

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
		.def(pybind11::init<>())
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
		.def_property_readonly("numLockEnabled", &dyno::PMouseEvent::numLockEnabled);
	//.def(py::self == py::self&);
//.def(py::self != py::self);

	py::class_<InputModule, Module, std::shared_ptr<InputModule>>(m, "InputModule")
		.def("get_module_type", &InputModule::getModuleType);

	py::class_<KeyboardInputModule, InputModule, std::shared_ptr<KeyboardInputModule>>(m, "KeyboardInputModule")
		.def("enqueue_event", &KeyboardInputModule::enqueueEvent)
		.def("var_cache_event", &KeyboardInputModule::varCacheEvent);

	py::class_<MouseInputModule, InputModule, std::shared_ptr<MouseInputModule>>(m, "MouseInputModule")
		.def("enqueue_event", &MouseInputModule::enqueueEvent)
		.def("var_cache_event", &MouseInputModule::varCacheEvent);

	py::class_<OutputModule, Module, std::shared_ptr<OutputModule>>(m, "OutputModule")
		.def("var_output_path", &OutputModule::varOutputPath, py::return_value_policy::reference)
		.def("var_prefix", &OutputModule::varPrefix, py::return_value_policy::reference)
		.def("var_start_frame", &OutputModule::varStartFrame, py::return_value_policy::reference)
		.def("var_end_frame", &OutputModule::varEndFrame, py::return_value_policy::reference)
		.def("var_stride", &OutputModule::varStride, py::return_value_policy::reference)
		.def("var_reordering", &OutputModule::varReordering, py::return_value_policy::reference)
		.def("in_frame_number", &OutputModule::inFrameNumber, py::return_value_policy::reference)
		.def("get_module_type", &OutputModule::getModuleType);

	py::class_<DataSource, Module, std::shared_ptr<DataSource>>(m, "DataSource")
		.def(py::init<>())
		.def("captionVisible", &DataSource::captionVisible)
		.def("get_module_type", &DataSource::getModuleType);

	//pipeline
	py::class_<GraphicsPipeline, Pipeline, std::shared_ptr<GraphicsPipeline>>(m, "GraphicsPipeline", py::buffer_protocol(), py::dynamic_attr());

	py::class_<AnimationPipeline, Pipeline, std::shared_ptr<AnimationPipeline>>(m, "AnimationPipeline", py::buffer_protocol(), py::dynamic_attr());

	//OBase
	typedef std::map<dyno::FieldID, FBase*> FieldMap;
	py::class_<OBase, Object, std::shared_ptr<OBase>>(m, "OBase")
		.def("caption", &OBase::caption)
		.def("caption_visible", &OBase::captionVisible)
		.def("description", &OBase::description)
		.def("get_name", &OBase::getName)
		.def("add_field", py::overload_cast<dyno::FBase*>(&OBase::addField))
		.def("add_field", py::overload_cast<dyno::FieldID, dyno::FBase*>(&OBase::addField))
		.def("add_field_alias", py::overload_cast<dyno::FieldID, dyno::FBase*>(&OBase::addFieldAlias))
		.def("add_field_alias", py::overload_cast<dyno::FieldID, dyno::FBase*, FieldMap&>(&OBase::addFieldAlias))
		.def("find_field", &OBase::findField)
		.def("find_field_alias", py::overload_cast<dyno::FieldID, FieldMap&>(&OBase::findFieldAlias))
		.def("find_field_alias", py::overload_cast<dyno::FieldID>(&OBase::findFieldAlias))
		.def("remove_field", &OBase::removeField)
		.def("remove_field_alias", py::overload_cast<dyno::FieldID>(&OBase::removeFieldAlias))
		.def("remove_field_alias", py::overload_cast<dyno::FieldID, FieldMap&>(&OBase::removeFieldAlias))
		/*.def("get_field", py::overload_cast<dyno::FieldID>(&OBase::getField), py::return_value_policy::reference)*/
		.def("get_all_fields", &OBase::getAllFields, py::return_value_policy::reference)
		.def("attach_field", &OBase::attachField)
		.def("is_all_fields_ready", &OBase::isAllFieldsReady)
		.def("get_field_alias", &OBase::getFieldAlias)
		.def("get_field_alias_count", &OBase::getFieldAliasCount)
		.def("set_block_coord", &OBase::setBlockCoord)
		.def("bx", &OBase::bx)
		.def("by", &OBase::by)
		.def("find_input_field", &OBase::findInputField)
		.def("add_input_field", &OBase::addInputField)
		.def("remove_input_field", &OBase::removeInputField)
		.def("get_input_fields", &OBase::getInputFields, py::return_value_policy::reference)
		.def("find_output_field", &OBase::findOutputField)
		.def("add_output_field", &OBase::addOutputField)
		.def("add_to_output", &OBase::addToOutput)
		.def("remove_output_field", &OBase::removeOutputField)
		.def("remove_from_output", &OBase::removeFromOutput)
		.def("get_output_fields", &OBase::getOutputFields, py::return_value_policy::reference)
		.def("find_parameter", &OBase::findParameter)
		.def("add_parameter", &OBase::addParameter)
		.def("remove_parameter", &OBase::removeParameter)
		.def("get_parameters", &OBase::getParameters, py::return_value_policy::reference);

	py::class_<TopologyModule, OBase, std::shared_ptr<TopologyModule>>(m, "TopologyModule")
		.def(py::init<>())
		.def("get_dof", &TopologyModule::getDOF)
		.def("tag_as_changed", &TopologyModule::tagAsChanged)
		.def("tag_as_unchanged", &TopologyModule::tagAsUnchanged)
		.def("is_topology_changed", &TopologyModule::isTopologyChanged)
		.def("update", &TopologyModule::update);

	py::class_<SceneGraph, OBase, std::shared_ptr<SceneGraph>>SG(m, "SceneGraph");
	SG.def(py::init<>())
		.def("advance", &SceneGraph::advance)
		.def("take_one_frame", &SceneGraph::takeOneFrame)
		.def("update_graphics_context", &SceneGraph::updateGraphicsContext)
		.def("run", &SceneGraph::run)
		.def("bounding_box", &SceneGraph::boundingBox)
		.def("reset", py::overload_cast<>(&SceneGraph::reset))
		.def("reset", py::overload_cast<std::shared_ptr<Node>>(&SceneGraph::reset))
		.def("print_node_info", &SceneGraph::printNodeInfo)
		.def("print_simulation_info", &SceneGraph::printSimulationInfo)
		.def("print_rendering_info", &SceneGraph::printRenderingInfo)
		.def("print_validation_info", &SceneGraph::printValidationInfo)
		.def("is_validation_info_printable", &SceneGraph::isValidationInfoPrintable)
		.def("is_node_info_printable", &SceneGraph::isNodeInfoPrintable)
		.def("is_simulation_info_printable", &SceneGraph::isSimulationInfoPrintable)
		.def("is_rendering_info_printable", &SceneGraph::isRenderingInfoPrintable)
		.def("load", &SceneGraph::load)
		.def("invoke", &SceneGraph::invoke)
		//createNewScene
		//.def("createNewScene", py::overload_cast<>()(&createNewScene<>))
		.def("delete_node", &SceneGraph::deleteNode)
		.def("propagate_node", &SceneGraph::propagateNode)
		.def("is_empty", &SceneGraph::isEmpty)
		.def("get_work_mode", &SceneGraph::getWorkMode)
		.def("get_instance", &SceneGraph::getInstance)
		.def("set_total_time", &SceneGraph::setTotalTime)
		.def("get_total_time", &SceneGraph::getTotalTime)
		.def("set_frame_rate", &SceneGraph::setFrameRate)
		.def("get_frame_rate", &SceneGraph::getFrameRate)
		.def("get_timecost_perframe", &SceneGraph::getTimeCostPerFrame)
		.def("get_frame_interval", &SceneGraph::getFrameInterval)
		.def("get_frame_number", &SceneGraph::getFrameNumber)
		.def("is_interval_adaptive", &SceneGraph::isIntervalAdaptive)
		.def("set_adaptive_interval", &SceneGraph::setAdaptiveInterval)
		.def("set_gravity", &SceneGraph::setGravity)
		.def("get_gravity", &SceneGraph::getGravity)
		.def("set_upper_bound", &SceneGraph::setUpperBound)
		.def("get_upper_bound", &SceneGraph::getUpperBound)
		.def("set_lower_bound", &SceneGraph::setLowerBound)
		.def("get_lower_bound", &SceneGraph::getLowerBound)
		.def("begin", &SceneGraph::begin)
		.def("end", &SceneGraph::end)
		.def("mark_queue_update_required", &SceneGraph::markQueueUpdateRequired)
		.def("on_mouse_event", py::overload_cast<dyno::PMouseEvent>(&SceneGraph::onMouseEvent))
		.def("on_mouse_event", py::overload_cast<dyno::PMouseEvent, std::shared_ptr<Node>>(&SceneGraph::onMouseEvent))
		.def("on_key_board_event", &SceneGraph::onKeyboardEvent)
		//.def("traverse_backward", py::overload_cast<dyno::Action*>(&SceneGraph::traverseBackward))
		//.def("traverse_forward", &SceneGraph::traverseForward)
		//.def("traverse_forward_with_auth_sync", &SceneGraph::traverseForwardWithAutoSync)
		.def("add_node", static_cast<std::shared_ptr<Node>(SceneGraph::*)(std::shared_ptr<Node>)>(&SceneGraph::addNode))
		.def("add_node", static_cast<std::shared_ptr<dyno::StaticTriangularMesh<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::StaticTriangularMesh<dyno::DataType3f>>)>(&SceneGraph::addNode))
		.def("add_node", static_cast<std::shared_ptr<dyno::VolumeBoundary<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::VolumeBoundary<dyno::DataType3f>>)>(&SceneGraph::addNode))
		.def("add_node", static_cast<std::shared_ptr<dyno::ParticleFluid<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::ParticleFluid<dyno::DataType3f>>)>(&SceneGraph::addNode))
		.def("add_node", static_cast<std::shared_ptr<dyno::PointsLoader<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::PointsLoader<dyno::DataType3f>>)>(&SceneGraph::addNode))
		.def("add_node", static_cast<std::shared_ptr<dyno::MakeParticleSystem<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::MakeParticleSystem<dyno::DataType3f>>)>(&SceneGraph::addNode))
		.def("add_node", static_cast<std::shared_ptr<dyno::GltfLoader<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::GltfLoader<dyno::DataType3f>>)>(&SceneGraph::addNode))
		.def("add_node", static_cast<std::shared_ptr<dyno::ParametricModel<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::ParametricModel<dyno::DataType3f>>)>(&SceneGraph::addNode))
		.def("add_node", static_cast<std::shared_ptr<dyno::GeometryLoader<dyno::DataType3f>>(SceneGraph::*)(std::shared_ptr<dyno::GeometryLoader<dyno::DataType3f>>)>(&SceneGraph::addNode));


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
		.def("can_load_file_by_extension", &dyno::SceneLoader::canLoadFileByExtension);


	py::class_<dyno::SceneLoaderFactory>(m, "SceneLoaderFactory")
		.def("get_instance", &dyno::SceneLoaderFactory::getInstance, py::return_value_policy::reference)
		.def("get_entry_by_file_extension", &dyno::SceneLoaderFactory::getEntryByFileExtension)
		.def("get_entry_by_file_name", &dyno::SceneLoaderFactory::getEntryByFileName)
		.def("add_entry", &dyno::SceneLoaderFactory::addEntry)
		.def("get_entry_list", &dyno::SceneLoaderFactory::getEntryList);

	py::class_<dyno::SceneLoaderXML, dyno::SceneLoader>(m, "SceneLoaderXML")
		.def("load", &dyno::SceneLoaderXML::load)
		.def("save", &dyno::SceneLoaderXML::save);

	//------------------------- New ------------------------------2024

	declare_p_enum(m);

	declare_var<float>(m, "f");
	declare_var<bool>(m, "b");
	declare_var<uint>(m, "uint");
	declare_var<std::string>(m, "s");
	declare_var<dyno::Vec3f>(m, "3f");
	declare_var<dyno::TOrientedBox3D<Real>>(m, "TOrientedBox3D");
	declare_var<dyno::FilePath>(m, "FilePath");
	declare_var<dyno::Color>(m, "Color");
	declare_var<dyno::RigidBody<dyno::DataType3f>>(m, "RigidBody3f");
	declare_var < dyno::Quat<float>>(m, "QuatFloat");
	py::class_<dyno::FVar<dyno::PEnum>, FBase, std::shared_ptr<dyno::FVar<dyno::PEnum>>>(m, "FVarPEnum")
		.def(py::init<>())
		.def("size", &dyno::FVar<dyno::PEnum>::size)
		.def("set_value", &dyno::FVar<dyno::PEnum>::setValue)
		.def("get_value", &dyno::FVar<dyno::PEnum>::getValue)
		.def("current_key", &dyno::FVar<dyno::PEnum>::currentKey)
		.def("set_current_key", &dyno::FVar<dyno::PEnum>::setCurrentKey)
		.def("serialize", &dyno::FVar<dyno::PEnum>::serialize)
		//.def("deserialize", &dyno::FVar<dyno::PEnum>::deserialize)
		.def("is_empty", &dyno::FVar<dyno::PEnum>::isEmpty)
		.def("get_data_ptr", &dyno::FVar<dyno::PEnum>::getDataPtr);

	declare_array<int, DeviceType::GPU>(m, "1D");
	declare_array<float, DeviceType::GPU>(m, "1fD");
	declare_array<Vec3f, DeviceType::GPU>(m, "3fD");
	declare_array<CollisionMask, DeviceType::GPU>(m, "CollisionMask");
	declare_array<dyno::TContactPair<float>, DeviceType::GPU>(m, "TContactPair");
	declare_array<dyno::Attribute, DeviceType::GPU>(m, "Attribute");

	declare_array_list<int, DeviceType::GPU>(m, "1D");
	declare_array_list<float, DeviceType::GPU>(m, "1fD");
	declare_array_list<Vec3f, DeviceType::GPU>(m, "3fD");

	declare_instance<TopologyModule>(m, "");
	declare_instance<dyno::PointSet<dyno::DataType3f>>(m, "PointSet3f");
	declare_instance<dyno::EdgeSet<dyno::DataType3f>>(m, "EdgeSet3f");
	declare_instance<dyno::TriangleSet<dyno::DataType3f>>(m, "TriangleSet3f");
	declare_instance<dyno::DiscreteElements<dyno::DataType3f>>(m, "DiscreteElements3f");
	declare_instance<dyno::HeightField<dyno::DataType3f>>(m, "HeightField3f");
	declare_instance<dyno::TextureMesh>(m, "TextureMesh");

	// New
	declare_parametric_model<dyno::DataType3f>(m, "3f");
	//import
	declare_multi_node_port<dyno::ParticleEmitter<dyno::DataType3f>>(m, "ParticleEmitter3f");
	declare_multi_node_port<dyno::ParticleSystem<dyno::DataType3f>>(m, "ParticleSystem3f");
	declare_multi_node_port<dyno::TriangularSystem<dyno::DataType3f>>(m, "TriangularSystem3f");
	declare_multi_node_port<dyno::CapillaryWave<dyno::DataType3f>>(m, "CapillaryWave3f");

	declare_single_node_port<dyno::Ocean<dyno::DataType3f>>(m, "Ocean3f");
	declare_single_node_port<dyno::OceanPatch<dyno::DataType3f>>(m, "OceanPatch3f");
	declare_single_node_port<dyno::GranularMedia<dyno::DataType3f>>(m, "GranularMedia3f");
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