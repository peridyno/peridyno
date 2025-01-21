#pragma once
#include "../PyCommon.h"

#include "RigidBody/RigidBodyShared.h"

using InstanceBase = dyno::InstanceBase;

#include "RigidBody/Module/AnimationDriver.h"
template <typename TDataType>
void declare_animation_driver(py::module& m, std::string typestr) {
	using Class = dyno::AnimationDriver<TDataType>;
	using Parent = dyno::KeyboardInputModule;
	std::string pyclass_name = std::string("AnimationDriver") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_speed", &Class::varSpeed, py::return_value_policy::reference)
		.def("var_driver_name", &Class::varDriverName, py::return_value_policy::reference)
		.def("in_topology", &Class::inTopology, py::return_value_policy::reference)
		.def("in_hierarchical_scene", &Class::inHierarchicalScene, py::return_value_policy::reference)
		.def("in_delta_time", &Class::inDeltaTime, py::return_value_policy::reference);
}

#include "RigidBody/Module/CarDriver.h"
template <typename TDataType>
void declare_car_driver(py::module& m, std::string typestr) {
	using Class = dyno::CarDriver<TDataType>;
	using Parent = dyno::KeyboardInputModule;
	std::string pyclass_name = std::string("CarDriver") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_topology", &Class::inTopology, py::return_value_policy::reference);
}

#include "RigidBody/Module/ContactsUnion.h"
template <typename TDataType>
void declare_contacts_union(py::module& m, std::string typestr) {
	using Class = dyno::ContactsUnion<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("ContactsUnion") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_contacts_a", &Class::inContactsA, py::return_value_policy::reference)
		.def("in_contacts_b", &Class::inContactsB, py::return_value_policy::reference)
		.def("out_contacts", &Class::outContacts, py::return_value_policy::reference);
}

#include "RigidBody/Module/InstanceTransform.h"
template <typename TDataType>
void declare_instance_transform(py::module& m, std::string typestr) {
	using Class = dyno::InstanceTransform<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("InstanceTransform") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_center", &Class::inCenter, py::return_value_policy::reference)
		.def("in_rotation_matrix", &Class::inRotationMatrix, py::return_value_policy::reference)
		.def("in_binding_pair", &Class::inBindingPair, py::return_value_policy::reference)
		.def("in_binding_tag", &Class::inBindingTag, py::return_value_policy::reference)
		.def("in_instance_transform", &Class::inInstanceTransform, py::return_value_policy::reference)
		.def("out_instance_transform", &Class::outInstanceTransform, py::return_value_policy::reference);
}

#include "RigidBody/Module/PCGConstraintSolver.h"
template <typename TDataType>
void declare_pcg_constraint_solver(py::module& m, std::string typestr) {
	using Class = dyno::PCGConstraintSolver<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("PCGConstraintSolver") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_friction_enabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("var_gravity_enabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("var_gravity_value", &Class::varGravityValue, py::return_value_policy::reference)
		.def("var_friction_coefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("var_slop", &Class::varSlop, py::return_value_policy::reference)
		.def("var_frequency", &Class::varFrequency, py::return_value_policy::reference)
		.def("var_damping_ratio", &Class::varDampingRatio, py::return_value_policy::reference)
		.def("var_iteration_number_for_velocity_solver_cg", &Class::varIterationNumberForVelocitySolverCG, py::return_value_policy::reference)
		.def("var_iteration_number_for_velocity_solver_jacobi", &Class::varIterationNumberForVelocitySolverJacobi, py::return_value_policy::reference)
		.def("var_linear_damping", &Class::varLinearDamping, py::return_value_policy::reference)
		.def("var_angular_damping", &Class::varAngularDamping, py::return_value_policy::reference)
		.def("var_tolerance", &Class::varTolerance, py::return_value_policy::reference)

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
		.def("in_discrete_elements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("in_attribute", &Class::inAttribute, py::return_value_policy::reference);
}

#include "RigidBody/Module/PJSConstraintSolver.h"
template <typename TDataType>
void declare_pjs_constraint_solver(py::module& m, std::string typestr) {
	using Class = dyno::PJSConstraintSolver<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("PJSConstraintSolver") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_friction_enabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("var_gravity_enabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("var_gravity_value", &Class::varGravityValue, py::return_value_policy::reference)
		.def("var_friction_coefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("var_slop", &Class::varSlop, py::return_value_policy::reference)
		.def("vat_baumgarte_rate", &Class::varBaumgarteRate, py::return_value_policy::reference)
		.def("var_iteration_number_for_velocity_solver", &Class::varIterationNumberForVelocitySolver, py::return_value_policy::reference)
		.def("var_linear_damping", &Class::varLinearDamping, py::return_value_policy::reference)
		.def("var_angular_damping", &Class::varAngularDamping, py::return_value_policy::reference)

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
		.def("in_discrete_elements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("in_attribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("in_friction_coefficients", &Class::inFrictionCoefficients, py::return_value_policy::reference);
}

#include "RigidBody/Module/PJSNJSConstraintSolver.h"
template <typename TDataType>
void declare_pjsnj_constraint_solver(py::module& m, std::string typestr) {
	using Class = dyno::PJSNJSConstraintSolver<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("PJSNJSConstraintSolver") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_friction_enabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("var_gravity_enabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("var_gravity_value", &Class::varGravityValue, py::return_value_policy::reference)
		.def("var_friction_coefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("var_slop", &Class::varSlop, py::return_value_policy::reference)
		.def("vat_baumgarte_bias", &Class::varBaumgarteBias, py::return_value_policy::reference)
		.def("var_iteration_number_for_velocity_solver", &Class::varIterationNumberForVelocitySolver, py::return_value_policy::reference)
		.def("var_iteration_number_for_position_solver", &Class::varIterationNumberForPositionSolver, py::return_value_policy::reference)
		.def("var_linear_damping", &Class::varLinearDamping, py::return_value_policy::reference)
		.def("var_angular_damping", &Class::varAngularDamping, py::return_value_policy::reference)

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
		.def("in_discrete_elements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("in_friction_coefficients", &Class::inFrictionCoefficients, py::return_value_policy::reference);
}

#include "RigidBody/Module/PJSoftConstraintSolver.h"
template <typename TDataType>
void declare_pj_soft_constraint_solver(py::module& m, std::string typestr) {
	using Class = dyno::PJSoftConstraintSolver<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("PJSoftConstraintSolver") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_friction_enabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("var_gravity_enabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("var_gravity_value", &Class::varGravityValue, py::return_value_policy::reference)
		.def("var_friction_coefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("var_slop", &Class::varSlop, py::return_value_policy::reference)
		.def("var_iteration_number_for_velocity_solver", &Class::varIterationNumberForVelocitySolver, py::return_value_policy::reference)
		.def("var_linear_damping", &Class::varLinearDamping, py::return_value_policy::reference)
		.def("var_angular_damping", &Class::varAngularDamping, py::return_value_policy::reference)
		.def("var_damping_ratio", &Class::varDampingRatio, py::return_value_policy::reference)
		.def("var_hertz", &Class::varHertz, py::return_value_policy::reference)

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
		.def("in_discrete_elements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("in_friction_coefficients", &Class::inFrictionCoefficients, py::return_value_policy::reference);
}

#include "RigidBody/Module/TJConstraintSolver.h"
template <typename TDataType>
void declare_tj_constraint_solver(py::module& m, std::string typestr) {
	using Class = dyno::TJConstraintSolver<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("TJConstraintSolver") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_friction_enabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("var_gravity_enabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("var_gravity_value", &Class::varGravityValue, py::return_value_policy::reference)
		.def("var_friction_coefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("var_slop", &Class::varSlop, py::return_value_policy::reference)
		.def("vat_baumgarte_bias", &Class::varBaumgarteBias, py::return_value_policy::reference)
		.def("var_sub_stepping", &Class::varSubStepping, py::return_value_policy::reference)
		.def("var_iteration_number_for_velocity_solver", &Class::varIterationNumberForVelocitySolver, py::return_value_policy::reference)
		.def("var_linear_damping", &Class::varLinearDamping, py::return_value_policy::reference)
		.def("var_angular_damping", &Class::varAngularDamping, py::return_value_policy::reference)

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
		.def("in_discrete_elements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("in_friction_coefficients", &Class::inFrictionCoefficients, py::return_value_policy::reference);
}

#include "RigidBody/Module/TJSoftConstraintSolver.h"
template <typename TDataType>
void declare_tj_soft_constraint_solver(py::module& m, std::string typestr) {
	using Class = dyno::TJSoftConstraintSolver<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("TJSoftConstraintSolver") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_friction_enabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("var_gravity_enabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("var_gravity_value", &Class::varGravityValue, py::return_value_policy::reference)
		.def("var_friction_coefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("var_slop", &Class::varSlop, py::return_value_policy::reference)
		.def("var_iteration_number_for_velocity_solver", &Class::varIterationNumberForVelocitySolver, py::return_value_policy::reference)
		.def("var_sub_stepping", &Class::varSubStepping, py::return_value_policy::reference)
		.def("var_linear_damping", &Class::varLinearDamping, py::return_value_policy::reference)
		.def("var_angular_damping", &Class::varAngularDamping, py::return_value_policy::reference)
		.def("var_damping_ratio", &Class::varDampingRatio, py::return_value_policy::reference)
		.def("var_hertz", &Class::varHertz, py::return_value_policy::reference)

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
		.def("in_discrete_elements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("in_friction_coefficients", &Class::inFrictionCoefficients, py::return_value_policy::reference);
}

#include "RigidBody/RigidBodySystem.h"
template <typename TDataType>
void declare_rigid_body_system(py::module& m, std::string typestr) {
	using Class = dyno::RigidBodySystem<TDataType>;
	using Parent = dyno::Node;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename dyno::Quat<Real> TQuat;
	std::string pyclass_name = std::string("RigidBodySystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("add_box", &Class::addBox, py::arg("box"), py::arg("bodyDef"), py::arg("density") = TDataType::Real(100))
		.def("add_sphere", &Class::addSphere, py::arg("sphere"), py::arg("bodyDef"), py::arg("density") = TDataType::Real(100))
		.def("add_tet", &Class::addTet, py::arg("tet"), py::arg("bodyDef"), py::arg("density") = TDataType::Real(100))
		.def("add_capsule", &Class::addCapsule, py::arg("capsule"), py::arg("bodyDef"), py::arg("density") = TDataType::Real(100))
		.def("create_rigid_body", py::overload_cast<const Coord&, const TQuat&>(&Class::createRigidBody))
		.def("create_rigid_body", py::overload_cast<const dyno::RigidBodyInfo&>(&Class::createRigidBody))
		.def("bind_box", &Class::bindBox)
		.def("bind_sphere", &Class::bindSphere)
		.def("bind_capsule", &Class::bindCapsule)
		.def("create_ball_and_socket_joint", &Class::createBallAndSocketJoint, py::return_value_policy::reference)
		.def("create_slider_joint", &Class::createSliderJoint, py::return_value_policy::reference)
		.def("create_hinge_joint", &Class::createHingeJoint, py::return_value_policy::reference)
		.def("create_fixed_joint", &Class::createFixedJoint, py::return_value_policy::reference)
		.def("create_unilateral_fixed_joint", &Class::createUnilateralFixedJoint, py::return_value_policy::reference)
		.def("create_point_joint", &Class::createPointJoint, py::return_value_policy::reference)
		.def("point_inertia", &Class::pointInertia)
		.def("get_node_type", &Class::getNodeType)

		.def("var_friction_enabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("var_gravity_enabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("var_gravity_value", &Class::varGravityValue, py::return_value_policy::reference)
		.def("var_friction_coefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("var_slop", &Class::varSlop, py::return_value_policy::reference)
		.def("state_topology", &Class::stateTopology, py::return_value_policy::reference)
		.def("state_friction_coefficents", &Class::stateFrictionCoefficients, py::return_value_policy::reference)
		.def("state_mass", &Class::stateMass, py::return_value_policy::reference)
		.def("state_center", &Class::stateCenter, py::return_value_policy::reference)
		.def("state_velocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("state_angular_velocity", &Class::stateAngularVelocity, py::return_value_policy::reference)
		.def("state_rotation_matrix", &Class::stateRotationMatrix, py::return_value_policy::reference)
		.def("state_inertia", &Class::stateInertia, py::return_value_policy::reference)
		.def("state_quaternion", &Class::stateQuaternion, py::return_value_policy::reference)
		.def("state_collision_mask", &Class::stateCollisionMask, py::return_value_policy::reference)
		.def("state_attribute", &Class::stateAttribute, py::return_value_policy::reference)
		.def("state_initial_inertia", &Class::stateInitialInertia, py::return_value_policy::reference);
}

#include "RigidBody/ArticulatedBody.h"
template <typename TDataType>
void declare_articulated_body(py::module& m, std::string typestr) {
	using Class = dyno::ArticulatedBody<TDataType>;
	using Parent1 = dyno::ParametricModel<TDataType>;
	using Parent2 = dyno::RigidBodySystem<TDataType>;
	std::string pyclass_name = std::string("ArticulatedBody") + typestr;
	py::class_<Class, Parent1, Parent2, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("bind", &Class::bind)
		.def("var_file_path", &Class::varFilePath, py::return_value_policy::reference)
		.def("var_vehicles_transform", &Class::varVehiclesTransform, py::return_value_policy::reference)
		.def("state_texture_mesh", &Class::stateTextureMesh, py::return_value_policy::reference)
		.def("state_instance_transform", &Class::stateInstanceTransform, py::return_value_policy::reference)
		.def("state_binding_pair", &Class::stateBindingPair, py::return_value_policy::reference)
		.def("state_binding_tag", &Class::stateBindingTag, py::return_value_policy::reference);
}

#include "RigidBody/ConfigurableBody.h"
template <typename TDataType>
void declare_configurable_body(py::module& m, std::string typestr) {
	using Class = dyno::ConfigurableBody<TDataType>;
	using Parent = dyno::ArticulatedBody<TDataType>;
	std::string pyclass_name = std::string("ConfigurableBody") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_vehicle_configuration", &Class::varVehicleConfiguration, py::return_value_policy::reference)
		.def("in_texture_mesh", &Class::inTextureMesh, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference);
}

#include "RigidBody/Gear.h"
template <typename TDataType>
void declare_gear(py::module& m, std::string typestr) {
	using Class = dyno::Gear<TDataType>;
	using Parent = dyno::ArticulatedBody<TDataType>;
	std::string pyclass_name = std::string("Gear") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

#include "RigidBody/MultibodySystem.h"
template <typename TDataType>
void declare_multibody_system(py::module& m, std::string typestr) {
	using Class = dyno::MultibodySystem<TDataType>;
	using Parent = dyno::RigidBodySystem<TDataType>;
	std::string pyclass_name = std::string("MultibodySystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_vehicles_transform", &Class::varVehiclesTransform, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("import_vehicles", &Class::importVehicles, py::return_value_policy::reference)
		.def("get_vehicles", &Class::getVehicles)
		.def("add_vehicle", &Class::addVehicle)
		.def("remove_vehicle", &Class::removeVehicle)
		.def("state_instance_transform", &Class::stateInstanceTransform, py::return_value_policy::reference);
}

#include "RigidBody/RigidBody.h"
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

#include "RigidBody/Vehicle.h"
template <typename TDataType>
void declare_jeep(py::module& m, std::string typestr) {
	using Class = dyno::Jeep<TDataType>;
	using Parent = dyno::ArticulatedBody<TDataType>;
	std::string pyclass_name = std::string("Jeep") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

template <typename TDataType>
void declare_tank(py::module& m, std::string typestr) {
	using Class = dyno::Tank<TDataType>;
	using Parent = dyno::ArticulatedBody<TDataType>;
	std::string pyclass_name = std::string("Tank") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

template <typename TDataType>
void declare_tracked_tank(py::module& m, std::string typestr) {
	using Class = dyno::TrackedTank<TDataType>;
	using Parent = dyno::ArticulatedBody<TDataType>;
	std::string pyclass_name = std::string("TrackedTank") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("state_caterpillar_track", &Class::statecaterpillarTrack, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_uav(py::module& m, std::string typestr) {
	using Class = dyno::UAV<TDataType>;
	using Parent = dyno::ArticulatedBody<TDataType>;
	std::string pyclass_name = std::string("UAV") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

template <typename TDataType>
void declare_uuv(py::module& m, std::string typestr) {
	using Class = dyno::UUV<TDataType>;
	using Parent = dyno::ArticulatedBody<TDataType>;
	std::string pyclass_name = std::string("UUV") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
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
