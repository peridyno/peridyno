#pragma once
#include "../PyCommon.h"

#include "RigidBody/RigidBody.h"
#include "RigidBody/RigidBodySystem.h"
#include "RigidBody/RigidBodyShared.h"

using InstanceBase = dyno::InstanceBase;

#include "RigidBody/ContactsUnion.h"
template <typename TDataType>
void declare_contacts_union(py::module& m, std::string typestr) {
	using Class = dyno::ContactsUnion<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("ContactsUnion") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("compute", &Class::compute)
		.def("in_contacts_a", &Class::inContactsA, py::return_value_policy::reference)
		.def("in_contacts_b", &Class::inContactsB, py::return_value_policy::reference)
		.def("out_contacts", &Class::outContacts, py::return_value_policy::reference);
}

#include "RigidBody/IterativeConstraintSolver.h"
template <typename TDataType>
void declare_iterative_constraint_solver(py::module& m, std::string typestr) {
	using Class = dyno::IterativeConstraintSolver<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("IterativeConstraintSolver") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_friction_enabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("var_gravity_enabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("var_gravity_value", &Class::varGravityValue, py::return_value_policy::reference)
		.def("var_friction_coefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("var_slop", &Class::varSlop, py::return_value_policy::reference)
		.def("var_iteration_number", &Class::varIterationNumber, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_mass", &Class::inMass, py::return_value_policy::reference)
		.def("in_center", &Class::inCenter, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_angular_velocity", &Class::inAngularVelocity, py::return_value_policy::reference)
		.def("in_rotation_matrix", &Class::inRotationMatrix, py::return_value_policy::reference)
		.def("in_inertia", &Class::inInertia, py::return_value_policy::reference)
		.def("in_initial_inertia", &Class::inInitialInertia, py::return_value_policy::reference)
		.def("in_quaternion", &Class::inQuaternion, py::return_value_policy::reference)
		.def("in_contacts", &Class::inContacts, py::return_value_policy::reference)
		.def("in_ball_and_socket_joints", &Class::inBallAndSocketJoints, py::return_value_policy::reference)
		.def("in_slider_joints", &Class::inSliderJoints, py::return_value_policy::reference)
		.def("in_hinge_joints", &Class::inHingeJoints, py::return_value_policy::reference)
		.def("in_fixed_joints", &Class::inFixedJoints, py::return_value_policy::reference)
		.def("in_point_joints", &Class::inPointJoints, py::return_value_policy::reference);
	;
}

template <typename TDataType>
void declare_rigid_body(py::module& m, std::string typestr) {
	using Class = dyno::RigidBody<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("RigidBody") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_gravity", &Class::varGravity, py::return_value_policy::reference)
		.def("state_mass", &Class::stateMass, py::return_value_policy::reference)
		.def("state_center", &Class::stateCenter, py::return_value_policy::reference)
		.def("state_velocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("state_angular_velocity", &Class::stateAngularVelocity, py::return_value_policy::reference)
		.def("state_rotation_matrix", &Class::stateRotationMatrix, py::return_value_policy::reference)
		.def("state_inertia", &Class::stateInertia, py::return_value_policy::reference)
		.def("state_quaternion", &Class::stateQuaternion, py::return_value_policy::reference)
		.def("state_initial_inertia", &Class::stateInitialInertia, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_rigid_body_system(py::module& m, std::string typestr) {
	using Class = dyno::RigidBodySystem<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("RigidBodySystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("add_box", &Class::addBox, py::arg("box"), py::arg("bodyDef"), py::arg("density") = TDataType::Real(100))
		.def("add_sphere", &Class::addSphere, py::arg("sphere"), py::arg("bodyDef"), py::arg("density") = TDataType::Real(100))
		.def("add_tet", &Class::addTet, py::arg("tet"), py::arg("bodyDef"), py::arg("density") = TDataType::Real(100))
		.def("add_capsule", &Class::addCapsule, py::arg("capsule"), py::arg("bodyDef"), py::arg("density") = TDataType::Real(100))
		.def("add_ball_and_socket_joint", &Class::addBallAndSocketJoint)
		.def("add_slider_joint", &Class::addSliderJoint)
		.def("add_hinge_joint", &Class::addHingeJoint)
		.def("add_fixed_joint", &Class::addFixedJoint)
		.def("add_point_joint", &Class::addPointJoint)
		.def("point_inertia", &Class::pointInertia)
		.def("get_dt", &Class::getDt)
		.def("var_friction_enabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("var_gravity_enabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("var_gravity_value", &Class::varGravityValue, py::return_value_policy::reference)
		.def("var_friction_coefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("var_slop", &Class::varSlop, py::return_value_policy::reference)
		.def("state_topology", &Class::stateTopology, py::return_value_policy::reference)
		.def("state_mass", &Class::stateMass, py::return_value_policy::reference)
		.def("state_center", &Class::stateCenter, py::return_value_policy::reference)
		.def("state_offset", &Class::stateOffset, py::return_value_policy::reference)
		.def("state_velocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("state_angular_velocity", &Class::stateAngularVelocity, py::return_value_policy::reference)
		.def("state_rotation_matrix", &Class::stateRotationMatrix, py::return_value_policy::reference)
		.def("state_inertia", &Class::stateInertia, py::return_value_policy::reference)
		.def("state_quaternion", &Class::stateQuaternion, py::return_value_policy::reference)
		.def("state_collision_mask", &Class::stateCollisionMask, py::return_value_policy::reference)
		.def("state_initial_inertia", &Class::stateInitialInertia, py::return_value_policy::reference)
		.def("state_ball_and_socket_joints", &Class::stateBallAndSocketJoints, py::return_value_policy::reference)
		.def("state_slider_joints", &Class::stateSliderJoints, py::return_value_policy::reference)
		.def("state_hinge_joints", &Class::stateHingeJoints, py::return_value_policy::reference)
		.def("state_fixed_joints", &Class::stateFixedJoints, py::return_value_policy::reference)
		.def("state_point_joints", &Class::statePointJoints, py::return_value_policy::reference);
}

#include "RigidBody/RigidMesh.h"
template <typename TDataType>
void declare_rigid_mesh(py::module& m, std::string typestr) {
	using Class = dyno::RigidMesh<TDataType>;
	using Parent = dyno::RigidBody<TDataType>;
	std::string pyclass_name = std::string("RigidMesh") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_envelope_name", &Class::varEnvelopeName, py::return_value_policy::reference)
		.def("var_mesh_name", &Class::varMeshName, py::return_value_policy::reference)
		.def("var_density", &Class::varDensity, py::return_value_policy::reference)
		.def("state_initial_envelope", &Class::stateInitialEnvelope, py::return_value_policy::reference)
		.def("state_envelope", &Class::stateEnvelope, py::return_value_policy::reference)
		.def("state_initial_mesh", &Class::stateInitialMesh, py::return_value_policy::reference)
		.def("state_mesh", &Class::stateMesh, py::return_value_policy::reference);
}

#include "RigidBody/Vechicle.h"
template <typename TDataType>
void declare_vechicle(py::module& m, std::string typestr) {
	using Class = dyno::Vechicle<TDataType>;
	using Parent = dyno::RigidBodySystem<TDataType>;
	std::string pyclass_name = std::string("Vechicle") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("bind", &Class::bind)
		.def("in_texture_mesh", &Class::inTextureMesh, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("state_binding", &Class::stateBinding, py::return_value_policy::reference)
		.def("state_binding_tag", &Class::stateBindingTag, py::return_value_policy::reference)
		.def("state_instance_transform", &Class::stateInstanceTransform, py::return_value_policy::reference);
}

// class: TContactPair      - For Examples_1: QT_Bricks
template<typename Real>
void declare_collision_data_t_contact_pair(py::module& m, std::string typestr) {
	using Class = dyno::TContactPair<Real>;

	std::string pyclass_name = std::string("TContactPair") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<int, int, dyno::ContactType, dyno::Vector<Real, 3>, dyno::Vector<Real, 3>, dyno::Vector<Real, 3>, dyno::Vector<Real, 3>>())
		.def_readwrite("bodyId1", &Class::bodyId1)
		.def_readwrite("bodyId2", &Class::bodyId2)
		.def_readwrite("interpenetration", &Class::interpenetration)
		.def_readwrite("pos1", &Class::pos1)
		.def_readwrite("pos2", &Class::pos2)
		.def_readwrite("normal1", &Class::normal1)
		.def_readwrite("normal2", &Class::normal2)
		.def_readwrite("contactType", &Class::contactType);
}

void declare_simple_vechicle_driver(py::module& m);

void declare_rigid_body_info(py::module& m, std::string typestr);

void declare_box_info(py::module& m, std::string typestr);

void declare_sphere_info(py::module& m, std::string typestr);

// class: TetInfo      - For Examples_1: QT_Bricks
void declare_tet_info(py::module& m, std::string typestr);

void declare_capsule_info(py::module& m, std::string typestr);

void pybind_rigid_body(py::module& m);
