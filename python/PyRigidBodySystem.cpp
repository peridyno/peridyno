#include "PyRigidBodySystem.h"
#include "RigidBody/RigidBody.h"
#include "RigidBody/RigidBodySystem.h"
#include "RigidBody/RigidBodyShared.h"

using InstanceBase = dyno::InstanceBase;


template <typename TDataType>
void declare_rigid_body_system(py::module& m, std::string typestr) {
	using Class = dyno::RigidBodySystem<TDataType>;
	using Parent = dyno::Node;

	std::string pyclass_name = std::string("RigidBodySystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("add_box", &Class::addBox)
		.def("add_sphere", &Class::addSphere)
		.def("add_tet", &Class::addTet);
}



void declare_rigid_body_info(py::module& m, std::string typestr) {
	using Class = dyno::RigidBodyInfo;

	std::string pyclass_name = std::string("RigidBodyInfo") + typestr;
	py::class_<Class, std::shared_ptr<Class>> (m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def_readwrite("linear_velocity", &Class::linearVelocity)
		.def_readwrite("angular_velocity", &Class::angularVelocity)
		.def_readwrite("position", &Class::position)
		.def_readwrite("inertia", &Class::inertia)
		.def_readwrite("mass", &Class::mass)
		.def_readwrite("friction", &Class::friction)
		.def_readwrite("restitution", &Class::restitution)
		.def_readwrite("motionType", &Class::motionType)
		.def_readwrite("shapeType", &Class::shapeType)
		.def_readwrite("collisionMask", &Class::collisionMask);
}


void declare_box_info(py::module& m, std::string typestr) {
	using Class = dyno::BoxInfo;

	std::string pyclass_name = std::string("BoxInfo") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def_readwrite("center", &Class::center)
		.def_readwrite("halfLength", &Class::halfLength);

}
void pybind_rigid_body_system(py::module& m) {
	declare_rigid_body_system<dyno::DataType3f>(m, "3f");

	declare_rigid_body_info(m, "");
	declare_box_info(m,"");

}