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

void pybind_log(py::module& m)
{
	//TODO: Log is updated, update the python binding as well
// 	py::class_<Log>(m, "Log")
// 		.def_static("set_output", &Log::setOutput)
// 		.def_static("get_output", &Log::getOutput)
// 		.def_static("set_level", &Log::setLevel);
}

void pybind_framework(py::module& m)
{
	pybind_log(m);

	//basic
	py::class_<Object, std::shared_ptr<Object>>(m, "Object")
		.def(py::init<>())
		.def("register_class", &Object::registerClass)
		//.def("create_object", &Object::createObject, py::return_value_policy::reference)
		.def("get_class_map", &Object::getClassMap, py::return_value_policy::reference)
		.def("base_id", &Object::baseId)
		.def("object_id", &Object::objectId);

	py::class_<Node, std::shared_ptr<Node>>(m, "Node")
		.def(py::init<>())
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
		.def("get_scene_graph", &Node::getSceneGraph)
		.def("get_import_nodes", &Node::getImportNodes)
		.def("get_export_nodes", &Node::getExportNodes)
		//.def("add_module", &Node::addModule)
		//.def("delete_module", &Node::deleteModule)
		.def("get_module_list", &Node::getModuleList)
		.def("has_module", &Node::hasModule)
		//.def("get_module", &Node::getModule)
		.def("reset_pipeline", &Node::resetPipeline, py::return_value_policy::reference)
		.def("graphics_pipeline", &Node::graphicsPipeline, py::return_value_policy::reference)
		.def("animation_pipeline", &Node::animationPipeline, py::return_value_policy::reference)
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

	py::class_<dyno::PluginEntry, std::shared_ptr<dyno::PluginEntry >>(m, "PluginEntry")
		.def(py::init<>())
		.def("name", &dyno::PluginEntry::name)
		.def("version", &dyno::PluginEntry::version)
		.def("description", &dyno::PluginEntry::description)
		.def("setName", &dyno::PluginEntry::setName, py::arg("pluginName"))
		.def("setVersion", &dyno::PluginEntry::setVersion, py::arg("pluginVersion"))
		.def("setDescription", &dyno::PluginEntry::setDescription, py::arg("desc"))
		.def("initialize", &dyno::PluginEntry::initialize);

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
		.def("deserialize", &FBase::deserialize)

		.def("get_top_field", &FBase::getTopField)
		.def("get_source", &FBase::getSource)
		.def("promote_ouput", &FBase::promoteOuput)
		.def("promote_input", &FBase::promoteInput)
		.def("demote_ouput", &FBase::demoteOuput)
		.def("demote_input", &FBase::demoteInput)

		.def("is_empty", &FBase::isEmpty)
		.def("update", &FBase::update)
		.def("attach", &FBase::attach)
		.def("detach", &FBase::detach);

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

	py::class_<VisualModule, Module, std::shared_ptr<VisualModule>>(m, "VisualModule")
		.def(py::init<>())
		.def("set_visible", &VisualModule::setVisible)
		.def("is_visible", &VisualModule::isVisible)
		.def("get_module_type", &VisualModule::getModuleType);

	py::class_<Pipeline, Module, std::shared_ptr<Pipeline>>(m, "Pipeline")
		.def("size_of_dynamic_modules", &Pipeline::sizeOfDynamicModules)
		.def("size_of_persistent_modules", &Pipeline::sizeOfPersistentModules)
		.def("push_module", &Pipeline::pushModule)
		.def("pop_module", &Pipeline::popModule)
		//.def("create_modules", &Pipeline::createModule)
		//.def("find_first_module", &Pipeline::findFirstModule)
		.def("clear", &Pipeline::clear)
		.def("push_persiistent_module", &Pipeline::pushPersistentModule)
		.def("active_moduiles", &Pipeline::activeModules, py::return_value_policy::reference)
		.def("all_modules", &Pipeline::allModules, py::return_value_policy::reference)
		.def("enable", &Pipeline::enable)
		.def("disable", &Pipeline::disable)
		.def("update_execution_queue", &Pipeline::updateExecutionQueue)
		.def("print_module_info", &Pipeline::printModuleInfo)
		.def("force_update", &Pipeline::forceUpdate)
		.def("promote_putput_to_node", &Pipeline::promoteOutputToNode, py::return_value_policy::reference)
		.def("demote_output_from_node", &Pipeline::demoteOutputFromNode);


	py::class_<ComputeModule, Module, std::shared_ptr<ComputeModule>>(m, "ComputeModule")
		.def("get_module_type", &dyno::ComputeModule::getModuleType);

	py::class_<ConstraintModule, Module, std::shared_ptr<ConstraintModule>>(m, "ConstraintModule")
		.def("constrain", &dyno::ConstraintModule::constrain)
		.def("get_module_type", &dyno::ConstraintModule::getModuleType);

	py::class_<GroupModule, Module, std::shared_ptr<GroupModule>>(m, "GroupModule")
		.def("push_module", &GroupModule::pushModule)
		.def("module_list", &GroupModule::moduleList)
		.def("set_parent_node", &GroupModule::setParentNode);

	py::class_<NumericalIntegrator, Module, std::shared_ptr<NumericalIntegrator>>(m, "NumericalIntegrator")
		.def(py::init<>()) // 绑定默认构造函数
		.def("begin", &NumericalIntegrator::begin) // 绑定 begin 方法
		.def("end", &NumericalIntegrator::end) // 绑定 end 方法
		.def("integrate", &NumericalIntegrator::integrate) // 绑定 integrate 方法
		.def("set_mass_id", &NumericalIntegrator::setMassID) // 绑定 setMassID 方法
		.def("set_force_id", &NumericalIntegrator::setForceID) // 绑定 setForceID 方法
		.def("set_torque_id", &NumericalIntegrator::setTorqueID) // 绑定 setTorqueID 方法
		.def("set_position_id", &NumericalIntegrator::setPositionID) // 绑定 setPositionID 方法
		.def("set_velocity_id", &NumericalIntegrator::setVelocityID) // 绑定 setVelocityID 方法
		.def("set_position_pre_id", &NumericalIntegrator::setPositionPreID) // 绑定 setPositionPreID 方法
		.def("set_velocity_pre_id", &NumericalIntegrator::setVelocityPreID) // 绑定 setVelocityPreID 方法
		.def("get_module_type", &NumericalIntegrator::getModuleType); // 绑定 getModuleType 方法

	py::class_<TopologyMappingdyno, Module, std::shared_ptr< TopologyMappingdyno>>(m, "TopologyMappingdyno");

	py::class_<InputModule, Module, std::shared_ptr<InputModule>>(m, "InputModule")
		.def("get_module_type", &InputModule::getModuleType);

	py::class_<OutputModule, Module, std::shared_ptr<OutputModule>>(m, "OutputModule")
		.def("var_output_path", &OutputModule::varOutputPath, py::return_value_policy::reference)
		.def("var_prefix", &OutputModule::varPrefix, py::return_value_policy::reference)
		.def("var_start_frame", &OutputModule::varStartFrame, py::return_value_policy::reference)
		.def("var_end_frame", &OutputModule::varEndFrame, py::return_value_policy::reference)
		.def("var_stride", &OutputModule::varStride, py::return_value_policy::reference)
		.def("var_reordering", &OutputModule::varReordering, py::return_value_policy::reference)
		.def("in_frame_number", &OutputModule::inFrameNumber, py::return_value_policy::reference)
		.def("get_module_type", &OutputModule::getModuleType);

	py::class_<MouseInputModule, InputModule, std::shared_ptr<MouseInputModule>>(m, "MouseInputModule")
		.def("enqueue_event", &MouseInputModule::enqueueEvent)
		.def("var_cache_event", &MouseInputModule::varCacheEvent);

	//pipeline

	py::class_<GraphicsPipeline, Pipeline, std::shared_ptr<GraphicsPipeline>>(m, "GraphicsPipeline", py::buffer_protocol(), py::dynamic_attr());

	py::class_<AnimationPipeline, Pipeline, std::shared_ptr<AnimationPipeline>>(m, "AnimationPipeline", py::buffer_protocol(), py::dynamic_attr());

	//OBase

	py::class_<OBase, Object, std::shared_ptr<OBase>>(m, "OBase")
		.def("caption", &OBase::caption)
		.def("caption_visible", &OBase::captionVisible)
		.def("description", &OBase::description)
		.def("get_name", &OBase::getName)
		/*.def("add_field", &OBase::addField)
		.def("add_field", &OBase::addField)
		.def("add_field_alias", &OBase::addFieldAlias)
		.def("add_field_alias", &OBase::addFieldAlias)*/
		.def("find_field", &OBase::findField)
		//.def("find_field_alias", &OBase::findFieldAlias)
		//.def("find_field_alias", &OBase::findFieldAlias)
		.def("remove_field", &OBase::removeField)
		//.def("remove_field_alias", &OBase::removeFieldAlias)
		//.def("remove_field_alias", &OBase::removeFieldAlias)
		//.def("get_field", &OBase::getField, py::return_value_policy::reference)
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
		//.def("reset", &SceneGraph::reset)
		.def("print_node_info", &SceneGraph::printNodeInfo)
		.def("print_module_info", &SceneGraph::printModuleInfo)
		.def("is_node_info_printable", &SceneGraph::isNodeInfoPrintable)
		.def("is_module_info_printable", &SceneGraph::isModuleInfoPrintable)
		.def("load", &SceneGraph::load)
		.def("invoke", &SceneGraph::invoke)
		//.def("create_new_scene", &SceneGraph::createNewScene)
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
		//.def("on_mouse_event", &SceneGraph::onMouseEvent)
		//.def("traverse_backward", &SceneGraph::traverseBackward)
		//.def("traverse_forward", &SceneGraph::traverseForward)
		//.def("traverse_forward_with_auth_sync", &SceneGraph::traverseForwardWithAutoSync)
		.def("add_node", static_cast<std::shared_ptr<Node>(SceneGraph::*)(std::shared_ptr<Node>)>(&SceneGraph::addNode));

	py::enum_<typename SceneGraph::EWorkMode>(m, "EWorkMode")
		.value("EDIT_MODE", SceneGraph::EWorkMode::EDIT_MODE)
		.value("RUNNING_MODE", SceneGraph::EWorkMode::RUNNING_MODE);

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
	py::class_<dyno::FVar<dyno::PEnum>, FBase, std::shared_ptr<dyno::FVar<dyno::PEnum>>>(m, "FVarPEnum")
		.def(py::init<>())
		.def("size", &dyno::FVar<dyno::PEnum>::size)
		.def("set_value", &dyno::FVar<dyno::PEnum>::setValue)
		.def("get_value", &dyno::FVar<dyno::PEnum>::getValue)
		.def("current_key", &dyno::FVar<dyno::PEnum>::currentKey)
		.def("set_current_key", &dyno::FVar<dyno::PEnum>::setCurrentKey)
		.def("serialize", &dyno::FVar<dyno::PEnum>::serialize)
		.def("deserialize", &dyno::FVar<dyno::PEnum>::deserialize)
		.def("is_empty", &dyno::FVar<dyno::PEnum>::isEmpty);

	declare_array<float, DeviceType::GPU>(m, "1fD");
	declare_array<dyno::Vec3f, DeviceType::GPU>(m, "3fD");

	declare_instance<TopologyModule>(m, "");
	declare_instance<dyno::PointSet<dyno::DataType3f>>(m, "PointSet3f");
	declare_instance<dyno::EdgeSet<dyno::DataType3f>>(m, "EdgeSet3f");
	declare_instance<dyno::TriangleSet<dyno::DataType3f>>(m, "TriangleSet3f");
	declare_instance<dyno::DiscreteElements<dyno::DataType3f>>(m, "DiscreteElements3f");
	declare_instance<dyno::HeightField<dyno::DataType3f>>(m, "HeightField3f");

	// New
	declare_parametric_model<dyno::DataType3f>(m, "3f");
	//import
	declare_multi_node_port<dyno::ParticleEmitter<dyno::DataType3f>>(m, "ParticleEmitter3f");
	declare_multi_node_port<dyno::ParticleSystem<dyno::DataType3f>>(m, "ParticleSystem3f");
	declare_multi_node_port<dyno::TriangularSystem<dyno::DataType3f>>(m, "TriangularSystem3f");
	declare_multi_node_port<dyno::CapillaryWave<dyno::DataType3f>>(m, "CapillaryWave3f");

	declare_single_node_port<dyno::Ocean<dyno::DataType3f>>(m, "Ocean3f");
	declare_single_node_port<dyno::OceanPatch<dyno::DataType3f>>(m, "OceanPatch3f");

	declare_camera(m);
}