#include "PyFramework.h"

void declare_modeling_init_static_plugin(py::module& m, std::string typestr) {
	using Class = dyno::ModelingInitializer;
	using Parent = dyno::PluginEntry;
	std::string pyclass_name = std::string("ModelingInitializer" + typestr);
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("modeling_init_static_plugin", &Modeling::initStaticPlugin);
}

void declare_paticleSystem_init_static_plugin(py::module& m, std::string typestr) {
	using Class = dyno::ParticleSystemInitializer;
	using Parent = dyno::PluginEntry;
	std::string pyclass_name = std::string("ParticleSystemInitializer" + typestr);
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("paticleSystem_init_static_plugin", &PaticleSystem::initStaticPlugin);
}

void declare_discrete_topology_mapping(py::module& m, std::string typestr)
{
	using Class = dyno::TopologyMapping;
	using Parent = dyno::Module;
	std::string pyclass_name = std::string("TopologyMapping") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str());
}

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

	py::class_<Node, std::shared_ptr<Node>>(m, "Node")
		.def(py::init<>())
		.def("set_name", &Node::setName)
		.def("is_active", &Node::isActive)
		.def("connect", &Node::connect)
		.def("set_visible", &Node::setVisible)
		.def("disconnect", &Node::disconnect)
		.def("graphics_pipeline", &Node::graphicsPipeline, py::return_value_policy::reference)
		.def("animation_pipeline", &Node::animationPipeline, py::return_value_policy::reference);

	py::class_<NodePort>(m, "NodePort")
		//.def(py::init<>())
		.def("get_port_name", &NodePort::getPortName)
		.def("get_port_type", &NodePort::getPortType)
		.def("set_port_type", &NodePort::setPortType)
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

	py::class_<dyno::PluginEntry>(m, "PluginEntry")
		.def(py::init<>())
		.def("name", &dyno::PluginEntry::name)
		.def("version", &dyno::PluginEntry::version)
		.def("description", &dyno::PluginEntry::description)
		.def("setName", &dyno::PluginEntry::setName, py::arg("pluginName"))
		.def("setVersion", &dyno::PluginEntry::setVersion, py::arg("pluginVersion"))
		.def("setDescription", &dyno::PluginEntry::setDescription, py::arg("desc"))
		.def("initialize", &dyno::PluginEntry::initialize);

	//FBase
	py::class_<InstanceBase, FBase, std::shared_ptr<InstanceBase>>(m, "FInstance");

	//module

	py::class_<Module, std::shared_ptr<Module>>(m, "Module")
		.def(py::init<>());

	py::class_<VisualModule, Module, std::shared_ptr<VisualModule>>(m, "VisualModule")
		.def(py::init<>())
		.def("set_visible", &VisualModule::setVisible)
		.def("is_visible", &VisualModule::isVisible)
		.def("get_module_type", &VisualModule::getModuleType);

	py::class_<Pipeline, Module, std::shared_ptr<Pipeline>>(m, "Pipeline")
		.def("push_module", &Pipeline::pushModule)
		.def("disable", &Pipeline::disable)
		.def("enable", &Pipeline::enable);

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

	py::class_<InputModule, Module, std::shared_ptr<InputModule>>(m, "InputModule")
		.def("get_module_type", &InputModule::getModuleType);

	py::class_<MouseInputModule, InputModule, std::shared_ptr<MouseInputModule>>(m, "MouseInputModule")
		.def("enqueue_event", &MouseInputModule::enqueueEvent)
		.def("var_cache_event", &MouseInputModule::varCacheEvent);


	//pipeline

	py::class_<GraphicsPipeline, Pipeline, std::shared_ptr<GraphicsPipeline>>(m, "GraphicsPipeline", py::buffer_protocol(), py::dynamic_attr());

	py::class_<AnimationPipeline, Pipeline, std::shared_ptr<AnimationPipeline>>(m, "AnimationPipeline", py::buffer_protocol(), py::dynamic_attr());

	//OBase

	py::class_<OBase, std::shared_ptr<OBase>>(m, "OBase");

	py::class_<TopologyModule, OBase, std::shared_ptr<TopologyModule>>(m, "TopologyModule")
		.def(py::init<>());

	py::class_<SceneGraph, OBase, std::shared_ptr<SceneGraph>>(m, "SceneGraph")
		.def(py::init<>())
		.def("bounding_box", &SceneGraph::boundingBox)
		.def("print_node_info", &SceneGraph::printNodeInfo)
		.def("print_module_info", &SceneGraph::printModuleInfo)
		.def("is_node_info_printable", &SceneGraph::isNodeInfoPrintable)
		.def("is_module_info_printable", &SceneGraph::isModuleInfoPrintable)
		.def("load", &SceneGraph::load)
		.def("invoke", &SceneGraph::invoke)
		.def("delete_node", &SceneGraph::deleteNode)
		.def("propagate_node", &SceneGraph::propagateNode)
		.def("is_empty", &SceneGraph::isEmpty)
		.def("set_total_time", &SceneGraph::setTotalTime)
		.def("get_total_time", &SceneGraph::getTotalTime)
		.def("set_frame_rate", &SceneGraph::setFrameRate)
		.def("get_frame_rate", &SceneGraph::getFrameRate)
		.def("get_timecost_perframe", &SceneGraph::getTimeCostPerFrame)
		.def("get_frame_interval", &SceneGraph::getFrameInterval)
		.def("get_frame_number", &SceneGraph::getFrameNumber)
		.def("set_gravity", &SceneGraph::setGravity)
		.def("get_gravity", &SceneGraph::getGravity)
		.def("set_upper_bound", &SceneGraph::setUpperBound)
		.def("get_upper_bound", &SceneGraph::getUpperBound)
		.def("set_lower_bound", &SceneGraph::setLowerBound)
		.def("get_lower_bound", &SceneGraph::getLowerBound)
		.def("add_node", static_cast<std::shared_ptr<Node>(SceneGraph::*)(std::shared_ptr<Node>)>(&SceneGraph::addNode));

	//------------------------- New ------------------------------2024

	declare_calculate_norm<dyno::DataType3f>(m, "3f");

	declare_pointset<dyno::DataType3f>(m, "3f");
	declare_edgeSet<dyno::DataType3f>(m, "3f");
	declare_triangleSet<dyno::DataType3f>(m, "3f");

	declare_discrete_topology_mapping(m, "3f");
	declare_discrete_elements_to_triangle_set<dyno::DataType3f>(m, "3f");

	declare_var<float>(m, "f");
	declare_var<bool>(m, "b");
	declare_var<uint>(m, "uint");
	declare_var<std::string>(m, "s");
	declare_var<dyno::Vec3f>(m, "3f");
	declare_var<dyno::TOrientedBox3D<Real>>(m, "TOrientedBox3D");
	declare_var<dyno::FilePath>(m, "FilePath");

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

	declare_merge_triangle_set<dyno::DataType3f>(m, "3f");

	declare_semiAnalyticalSFI_node<dyno::DataType3f>(m, "3f");

	declare_modeling_init_static_plugin(m, "");
	declare_paticleSystem_init_static_plugin(m, "");

	//import
	declare_multi_node_port<dyno::ParticleEmitter<dyno::DataType3f>>(m, "ParticleEmitter3f");
	declare_multi_node_port<dyno::ParticleSystem<dyno::DataType3f>>(m, "ParticleSystem3f");
	declare_multi_node_port<dyno::TriangularSystem<dyno::DataType3f>>(m, "TriangularSystem3f");
	declare_multi_node_port<dyno::CapillaryWave<dyno::DataType3f>>(m, "CapillaryWave3f");

	declare_single_node_port<dyno::Ocean<dyno::DataType3f>>(m, "Ocean3f");
	declare_single_node_port<dyno::OceanPatch<dyno::DataType3f>>(m, "OceanPatch3f");

	declare_camera(m);

	//declare_semiAnalyticalScheme_init_static_plugin(m, "");
}