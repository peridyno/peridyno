#include "PyRigidBody.h"

#include "RigidBody/Module/SharedFuncsForRigidBody.h"

void declare_rigid_body_info(py::module& m, std::string typestr) {
	using Class = dyno::RigidBodyInfo;
	std::string pyclass_name = std::string("RigidBodyInfo") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init < dyno::Vector<float, 3>, dyno::Quat<Real>>())
		.def_readwrite("linear_velocity", &Class::linearVelocity)
		.def_readwrite("angular_velocity", &Class::angularVelocity)
		.def_readwrite("position", &Class::position)
		.def_readwrite("inertia", &Class::inertia)
		.def_readwrite("mass", &Class::mass)
		.def_readwrite("friction", &Class::friction)
		.def_readwrite("restitution", &Class::restitution)
		.def_readwrite("motion_type", &Class::motionType)
		.def_readwrite("shape_type", &Class::shapeType)
		.def_readwrite("collision_mask", &Class::collisionMask)
		.def_readwrite("angle", &Class::angle);
}

void declare_box_info(py::module& m, std::string typestr) {
	using Class = dyno::BoxInfo;

	std::string pyclass_name = std::string("BoxInfo") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def_readwrite("center", &Class::center)
		.def_readwrite("half_length", &Class::halfLength)
		.def_readwrite("rot", &Class::rot);
}

void declare_sphere_info(py::module& m, std::string typestr) {
	using Class = dyno::SphereInfo;

	std::string pyclass_name = std::string("SphereInfo") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def_readwrite("center", &Class::center)
		.def_readwrite("radius", &Class::radius)
		.def_readwrite("rot", &Class::rot);
}

// class: TetInfo      - For Examples_1: QT_Bricks
void declare_tet_info(py::module& m, std::string typestr) {
	using Class = dyno::TetInfo;

	std::string pyclass_name = std::string("TetInfo") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//		.def_readwrite("v", &Class::v); // "def_readwrite" is not applicable to fixed arrays,so replace it with the get-set method.
		.def_property("v", &dyno::get_v, &dyno::set_v);
}

void declare_capsule_info(py::module& m, std::string typestr) {
	using Class = dyno::CapsuleInfo;
	std::string pyclass_name = std::string("CapsuleInfo") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def_readwrite("center", &Class::center)
		.def_readwrite("radius", &Class::radius)
		.def_readwrite("rot", &Class::rot)
		.def_readwrite("half_length", &Class::halfLength);
}

#include "RigidBody/Module/SimpleVechicleDriver.h"
void declare_simple_vechicle_driver(py::module& m) {
	using Class = dyno::SimpleVechicleDriver;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("SimpleVechicleDriver");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("in_frame_number", &Class::inFrameNumber, py::return_value_policy::reference)
		.def("in_instance_transform", &Class::inInstanceTransform, py::return_value_policy::reference);
}

void pybind_rigid_body(py::module& m) {
	// Module
	declare_animation_driver<dyno::DataType3f>(m, "3f");
	declare_car_driver<dyno::DataType3f>(m, "3f");
	declare_contacts_union<dyno::DataType3f>(m, "3f");
	declare_instance_transform<dyno::DataType3f>(m, "3f");
	declare_pcg_constraint_solver<dyno::DataType3f>(m, "3f");
	declare_pjs_constraint_solver<dyno::DataType3f>(m, "3f");
	declare_pjsnj_constraint_solver<dyno::DataType3f>(m, "3f");
	declare_pj_soft_constraint_solver<dyno::DataType3f>(m, "3f");
	declare_simple_vechicle_driver(m);
	declare_tj_constraint_solver<dyno::DataType3f>(m, "3f");
	declare_tj_soft_constraint_solver<dyno::DataType3f>(m, "3f");
	declare_rigid_body_system<dyno::DataType3f>(m, "3f");
	declare_articulated_body<dyno::DataType3f>(m, "3f");
	declare_configurable_body<dyno::DataType3f>(m, "3f");
	declare_gear<dyno::DataType3f>(m, "3f");
	declare_multibody_system<dyno::DataType3f>(m, "3f");
	declare_rigid_body<dyno::DataType3f>(m, "3f");
	declare_rigid_mesh<dyno::DataType3f>(m, "3f");
	declare_jeep<dyno::DataType3f>(m, "3f");
	declare_tank<dyno::DataType3f>(m, "3f");
	declare_tracked_tank<dyno::DataType3f>(m, "3f");
	declare_uav<dyno::DataType3f>(m, "3f");
	declare_uuv<dyno::DataType3f>(m, "3f");

	declare_rigid_body_info(m, "");
	declare_box_info(m, "");
	declare_sphere_info(m, "");
	declare_tet_info(m, "");
	declare_capsule_info(m, "");
}