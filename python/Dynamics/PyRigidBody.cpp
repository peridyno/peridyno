#include "PyRigidBody.h"

void declare_rigid_body_info(py::module& m, std::string typestr) {
	using Class = dyno::RigidBodyInfo;

	std::string pyclass_name = std::string("RigidBodyInfo") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
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

void declare_sphere_info(py::module& m, std::string typestr) {
	using Class = dyno::SphereInfo;

	std::string pyclass_name = std::string("SphereInfo") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def_readwrite("center", &Class::center)
		.def_readwrite("radius", &Class::radius);
}

// class: TetInfo      - For Examples_1: QT_Bricks
void declare_tet_info(py::module& m, std::string typestr) {
	using Class = dyno::TetInfo;

	std::string pyclass_name = std::string("TetInfo") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//		.def_readwrite("v", &Class::v); // "def_readwrite" is not applicable to fixed arrays,so replace it with the get-set method.
		.def_property("v", &get_v, &set_v);
}

void pybind_rigid_body_system(py::module& m) {
	declare_rigid_body<dyno::DataType3f>(m, "3f");
	declare_rigid_body_system<dyno::DataType3f>(m, "3f");

	declare_rigid_body_info(m, "");
	declare_box_info(m, "");

	declare_sphere_info(m, "");
	declare_tet_info(m, "");

	declare_neighbor_element_query<dyno::DataType3f>(m, "3f");
	declare_contacts_to_edge_set<dyno::DataType3f>(m, "3f");
	declare_contacts_to_point_set<dyno::DataType3f>(m, "3f");
}