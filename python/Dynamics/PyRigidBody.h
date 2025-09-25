#pragma once
#include "../PyCommon.h"

#include "RigidBody/RigidBodyShared.h"

using InstanceBase = dyno::InstanceBase;

#include "RigidBody/Module/AnimationDriver.h"
template <typename TDataType>
void declare_animation_driver(py::module& m, std::string typestr) {
	using Class = dyno::AnimationDriver<TDataType>;
	using Parent = dyno::KeyboardInputModule;

	class AnimationDriverTrampoline : public Class
	{
	public:
		using Class::Class;

		void onEvent(dyno::PKeyboardEvent event) override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::AnimationDriver<TDataType>,
				onEvent,
				event
			);
		}
	};

	class AnimationDriverPublicist : public Class
	{
	public:
		using Class::onEvent;
		using Class::getFrameAndWeight;
	};

	std::string pyclass_name = std::string("AnimationDriver") + typestr;
	py::class_<Class, Parent, AnimationDriverTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varSpeed", &Class::varSpeed, py::return_value_policy::reference)
		.def("varBindingConfiguration", &Class::varBindingConfiguration, py::return_value_policy::reference)
		.def("inTopology", &Class::inTopology, py::return_value_policy::reference)
		.def("inJointAnimationInfo", &Class::inJointAnimationInfo, py::return_value_policy::reference)
		.def("inDeltaTime", &Class::inDeltaTime, py::return_value_policy::reference)
		// protected
		.def("onEvent", &AnimationDriverPublicist::onEvent, py::return_value_policy::reference)
		.def("getFrameAndWeight", &AnimationDriverPublicist::getFrameAndWeight, py::return_value_policy::reference);
}

#include "RigidBody/Module/CarDriver.h"
template <typename TDataType>
void declare_car_driver(py::module& m, std::string typestr) {
	using Class = dyno::CarDriver<TDataType>;
	using Parent = dyno::KeyboardInputModule;

	class CarDriverTrampoline : public Class
	{
	public:
		using Class::Class;

		void onEvent(dyno::PKeyboardEvent event) override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::CarDriver<TDataType>,
				onEvent,
				event
			);
		}
	};

	class CarDriverPublicist : public Class
	{
	public:
		using Class::onEvent;
	};

	std::string pyclass_name = std::string("CarDriver") + typestr;
	py::class_<Class, Parent, CarDriverTrampoline,std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inTopology", &Class::inTopology, py::return_value_policy::reference)
		// protected
		.def("onEvent", &CarDriverPublicist::onEvent, py::return_value_policy::reference);
}

#include "RigidBody/Module/ContactsUnion.h"
template <typename TDataType>
void declare_contacts_union(py::module& m, std::string typestr) {
	using Class = dyno::ContactsUnion<TDataType>;
	using Parent = dyno::ComputeModule;

	class ContactsUnionTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ContactsUnion<TDataType>,
				compute,
			);
		}

		bool validateInputs() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::ContactsUnion<TDataType>,
				validateInputs,
				);
		}
	};

	class ContactsUnionPublicist : public Class
	{
	public:
		using Class::compute;
		using Class::validateInputs;
	};

	std::string pyclass_name = std::string("ContactsUnion") + typestr;
	py::class_<Class, Parent, ContactsUnionTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inContactsA", &Class::inContactsA, py::return_value_policy::reference)
		.def("inContactsB", &Class::inContactsB, py::return_value_policy::reference)
		.def("outContacts", &Class::outContacts, py::return_value_policy::reference)
		// protected
		.def("compute", &ContactsUnionPublicist::compute, py::return_value_policy::reference)
		.def("validateInputs", &ContactsUnionPublicist::validateInputs, py::return_value_policy::reference);
}

#include "RigidBody/Module/InstanceTransform.h"
template <typename TDataType>
void declare_instance_transform(py::module& m, std::string typestr) {
	using Class = dyno::InstanceTransform<TDataType>;
	using Parent = dyno::ComputeModule;

	class InstanceTransformTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::InstanceTransform<TDataType>,
				compute,
				);
		}
	};

	class InstanceTransformPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("InstanceTransform") + typestr;
	py::class_<Class, Parent, InstanceTransformTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inCenter", &Class::inCenter, py::return_value_policy::reference)
		.def("inRotationMatrix", &Class::inRotationMatrix, py::return_value_policy::reference)
		.def("inBindingPair", &Class::inBindingPair, py::return_value_policy::reference)
		.def("inBindingTag", &Class::inBindingTag, py::return_value_policy::reference)
		.def("inInstanceTransform", &Class::inInstanceTransform, py::return_value_policy::reference)
		.def("outInstanceTransform", &Class::outInstanceTransform, py::return_value_policy::reference)
		// protected
		.def("compute", &InstanceTransformPublicist::compute, py::return_value_policy::reference);
}

#include "RigidBody/Module/PCGConstraintSolver.h"
template <typename TDataType>
void declare_pcg_constraint_solver(py::module& m, std::string typestr) {
	using Class = dyno::PCGConstraintSolver<TDataType>;
	using Parent = dyno::ConstraintModule;

	class PCGConstraintSolverTrampoline : public Class
	{
	public:
		using Class::Class;

		void constrain() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PCGConstraintSolver<TDataType>,
				constrain,
				);
		}
	};

	class PCGConstraintSolverPublicist : public Class
	{
	public:
		using Class::constrain;
	};

	std::string pyclass_name = std::string("PCGConstraintSolver") + typestr;
	py::class_<Class, Parent, PCGConstraintSolverTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varFrictionEnabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("varGravityEnabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("varGravityValue", &Class::varGravityValue, py::return_value_policy::reference)
		.def("varFrictionCoefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("varSlop", &Class::varSlop, py::return_value_policy::reference)
		.def("varFrequency", &Class::varFrequency, py::return_value_policy::reference)
		.def("varDampingRatio", &Class::varDampingRatio, py::return_value_policy::reference)
		.def("varIterationNumberForVelocitySolverCG", &Class::varIterationNumberForVelocitySolverCG, py::return_value_policy::reference)
		.def("varIterationNumberForVelocitySolverJacobi", &Class::varIterationNumberForVelocitySolverJacobi, py::return_value_policy::reference)
		.def("varLinearDamping", &Class::varLinearDamping, py::return_value_policy::reference)
		.def("varAngularDamping", &Class::varAngularDamping, py::return_value_policy::reference)
		.def("varTolerance", &Class::varTolerance, py::return_value_policy::reference)

		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inMass", &Class::inMass, py::return_value_policy::reference)
		.def("inCenter", &Class::inCenter, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inAngularVelocity", &Class::inAngularVelocity, py::return_value_policy::reference)
		.def("inRotationMatrix", &Class::inRotationMatrix, py::return_value_policy::reference)
		.def("inInertia", &Class::inInertia, py::return_value_policy::reference)
		.def("inInitialInertia", &Class::inInitialInertia, py::return_value_policy::reference)
		.def("inQuaternion", &Class::inQuaternion, py::return_value_policy::reference)
		.def("inContacts", &Class::inContacts, py::return_value_policy::reference)
		.def("inDiscreteElements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference)
		// protected
		.def("constrain", &PCGConstraintSolverPublicist::constrain, py::return_value_policy::reference);
}

#include "RigidBody/Module/PJSConstraintSolver.h"
template <typename TDataType>
void declare_pjs_constraint_solver(py::module& m, std::string typestr) {
	using Class = dyno::PJSConstraintSolver<TDataType>;
	using Parent = dyno::ConstraintModule;

	class PJSConstraintSolverTrampoline : public Class
	{
	public:
		using Class::Class;

		void constrain() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PJSConstraintSolver<TDataType>,
				constrain,
				);
		}
	};

	class PJSConstraintSolverPublicist : public Class
	{
	public:
		using Class::constrain;
	};

	std::string pyclass_name = std::string("PJSConstraintSolver") + typestr;
	py::class_<Class, Parent, PJSConstraintSolverTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varFrictionEnabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("varGravityEnabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("varGravityValue", &Class::varGravityValue, py::return_value_policy::reference)
		.def("varFrictionCoefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("varSlop", &Class::varSlop, py::return_value_policy::reference)
		.def("vat_baumgarte_rate", &Class::varBaumgarteRate, py::return_value_policy::reference)
		.def("varIterationNumberForVelocitySolver", &Class::varIterationNumberForVelocitySolver, py::return_value_policy::reference)
		.def("varLinearDamping", &Class::varLinearDamping, py::return_value_policy::reference)
		.def("varAngularDamping", &Class::varAngularDamping, py::return_value_policy::reference)

		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inMass", &Class::inMass, py::return_value_policy::reference)
		.def("inCenter", &Class::inCenter, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inAngularVelocity", &Class::inAngularVelocity, py::return_value_policy::reference)
		.def("inRotationMatrix", &Class::inRotationMatrix, py::return_value_policy::reference)
		.def("inInertia", &Class::inInertia, py::return_value_policy::reference)
		.def("inInitialInertia", &Class::inInitialInertia, py::return_value_policy::reference)
		.def("inQuaternion", &Class::inQuaternion, py::return_value_policy::reference)
		.def("inContacts", &Class::inContacts, py::return_value_policy::reference)
		.def("inDiscreteElements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("inFrictionCoefficients", &Class::inFrictionCoefficients, py::return_value_policy::reference)
		// protected
		.def("constrain", &PJSConstraintSolverPublicist::constrain, py::return_value_policy::reference);
}

#include "RigidBody/Module/PJSNJSConstraintSolver.h"
template <typename TDataType>
void declare_pjsnj_constraint_solver(py::module& m, std::string typestr) {
	using Class = dyno::PJSNJSConstraintSolver<TDataType>;
	using Parent = dyno::ConstraintModule;

	class PJSNJSConstraintSolverTrampoline : public Class
	{
	public:
		using Class::Class;

		void constrain() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PJSNJSConstraintSolver<TDataType>,
				constrain,
				);
		}
	};

	class PJSNJSConstraintSolverPublicist : public Class
	{
	public:
		using Class::constrain;
	};

	std::string pyclass_name = std::string("PJSNJSConstraintSolver") + typestr;
	py::class_<Class, Parent, PJSNJSConstraintSolverTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varFrictionEnabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("varGravityEnabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("varGravityValue", &Class::varGravityValue, py::return_value_policy::reference)
		.def("varFrictionCoefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("varSlop", &Class::varSlop, py::return_value_policy::reference)
		.def("vat_baumgarte_bias", &Class::varBaumgarteBias, py::return_value_policy::reference)
		.def("varIterationNumberForVelocitySolver", &Class::varIterationNumberForVelocitySolver, py::return_value_policy::reference)
		.def("varIterationNumberForPositionSolver", &Class::varIterationNumberForPositionSolver, py::return_value_policy::reference)
		.def("varLinearDamping", &Class::varLinearDamping, py::return_value_policy::reference)
		.def("varAngularDamping", &Class::varAngularDamping, py::return_value_policy::reference)

		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inMass", &Class::inMass, py::return_value_policy::reference)
		.def("inCenter", &Class::inCenter, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inAngularVelocity", &Class::inAngularVelocity, py::return_value_policy::reference)
		.def("inRotationMatrix", &Class::inRotationMatrix, py::return_value_policy::reference)
		.def("inInertia", &Class::inInertia, py::return_value_policy::reference)
		.def("inInitialInertia", &Class::inInitialInertia, py::return_value_policy::reference)
		.def("inQuaternion", &Class::inQuaternion, py::return_value_policy::reference)
		.def("inContacts", &Class::inContacts, py::return_value_policy::reference)
		.def("inDiscreteElements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("inFrictionCoefficients", &Class::inFrictionCoefficients, py::return_value_policy::reference)
		// protected
		.def("constrain", &PJSNJSConstraintSolverPublicist::constrain, py::return_value_policy::reference);
}

#include "RigidBody/Module/PJSoftConstraintSolver.h"
template <typename TDataType>
void declare_pj_soft_constraint_solver(py::module& m, std::string typestr) {
	using Class = dyno::PJSoftConstraintSolver<TDataType>;
	using Parent = dyno::ConstraintModule;

	class PJSoftConstraintSolverTrampoline : public Class
	{
	public:
		using Class::Class;

		void constrain() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PJSoftConstraintSolver<TDataType>,
				constrain,
				);
		}
	};

	class PJSoftConstraintSolverPublicist : public Class
	{
	public:
		using Class::constrain;
	};

	std::string pyclass_name = std::string("PJSoftConstraintSolver") + typestr;
	py::class_<Class, Parent, PJSoftConstraintSolverTrampoline,std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varFrictionEnabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("varGravityEnabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("varGravityValue", &Class::varGravityValue, py::return_value_policy::reference)
		.def("varFrictionCoefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("varSlop", &Class::varSlop, py::return_value_policy::reference)
		.def("varIterationNumberForVelocitySolver", &Class::varIterationNumberForVelocitySolver, py::return_value_policy::reference)
		.def("varLinearDamping", &Class::varLinearDamping, py::return_value_policy::reference)
		.def("varAngularDamping", &Class::varAngularDamping, py::return_value_policy::reference)
		.def("varDampingRatio", &Class::varDampingRatio, py::return_value_policy::reference)
		.def("varHertz", &Class::varHertz, py::return_value_policy::reference)

		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inMass", &Class::inMass, py::return_value_policy::reference)
		.def("inCenter", &Class::inCenter, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inAngularVelocity", &Class::inAngularVelocity, py::return_value_policy::reference)
		.def("inRotationMatrix", &Class::inRotationMatrix, py::return_value_policy::reference)
		.def("inInertia", &Class::inInertia, py::return_value_policy::reference)
		.def("inInitialInertia", &Class::inInitialInertia, py::return_value_policy::reference)
		.def("inQuaternion", &Class::inQuaternion, py::return_value_policy::reference)
		.def("inContacts", &Class::inContacts, py::return_value_policy::reference)
		.def("inDiscreteElements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("inFrictionCoefficients", &Class::inFrictionCoefficients, py::return_value_policy::reference)
		// protected
		.def("constrain", &PJSoftConstraintSolverPublicist::constrain, py::return_value_policy::reference);
}

#include "RigidBody/Module/TJConstraintSolver.h"
template <typename TDataType>
void declare_tj_constraint_solver(py::module& m, std::string typestr) {
	using Class = dyno::TJConstraintSolver<TDataType>;
	using Parent = dyno::ConstraintModule;

	class TJConstraintSolverTrampoline : public Class
	{
	public:
		using Class::Class;

		void constrain() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TJConstraintSolver<TDataType>,
				constrain,
				);
		}
	};

	class TJConstraintSolverPublicist : public Class
	{
	public:
		using Class::constrain;
	};

	std::string pyclass_name = std::string("TJConstraintSolver") + typestr;
	py::class_<Class, Parent, TJConstraintSolverTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varFrictionEnabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("varGravityEnabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("varGravityValue", &Class::varGravityValue, py::return_value_policy::reference)
		.def("varFrictionCoefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("varSlop", &Class::varSlop, py::return_value_policy::reference)
		.def("vat_baumgarte_bias", &Class::varBaumgarteBias, py::return_value_policy::reference)
		.def("varSubStepping", &Class::varSubStepping, py::return_value_policy::reference)
		.def("varIterationNumberForVelocitySolver", &Class::varIterationNumberForVelocitySolver, py::return_value_policy::reference)
		.def("varLinearDamping", &Class::varLinearDamping, py::return_value_policy::reference)
		.def("varAngularDamping", &Class::varAngularDamping, py::return_value_policy::reference)

		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inMass", &Class::inMass, py::return_value_policy::reference)
		.def("inCenter", &Class::inCenter, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inAngularVelocity", &Class::inAngularVelocity, py::return_value_policy::reference)
		.def("inRotationMatrix", &Class::inRotationMatrix, py::return_value_policy::reference)
		.def("inInertia", &Class::inInertia, py::return_value_policy::reference)
		.def("inInitialInertia", &Class::inInitialInertia, py::return_value_policy::reference)
		.def("inQuaternion", &Class::inQuaternion, py::return_value_policy::reference)
		.def("inContacts", &Class::inContacts, py::return_value_policy::reference)
		.def("inDiscreteElements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("inFrictionCoefficients", &Class::inFrictionCoefficients, py::return_value_policy::reference)
		// protected
		.def("constrain", &TJConstraintSolverPublicist::constrain, py::return_value_policy::reference);

}

#include "RigidBody/Module/TJSoftConstraintSolver.h"
template <typename TDataType>
void declare_tj_soft_constraint_solver(py::module& m, std::string typestr) {
	using Class = dyno::TJSoftConstraintSolver<TDataType>;
	using Parent = dyno::ConstraintModule;

	class TJSoftConstraintSolverTrampoline : public Class
	{
	public:
		using Class::Class;

		void constrain() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TJSoftConstraintSolver<TDataType>,
				constrain,
				);
		}
	};

	class TJSoftConstraintSolverPublicist : public Class
	{
	public:
		using Class::constrain;
	};

	std::string pyclass_name = std::string("TJSoftConstraintSolver") + typestr;
	py::class_<Class, Parent, TJSoftConstraintSolverTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varFrictionEnabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("varGravityEnabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("varGravityValue", &Class::varGravityValue, py::return_value_policy::reference)
		.def("varFrictionCoefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("varSlop", &Class::varSlop, py::return_value_policy::reference)
		.def("varIterationNumberForVelocitySolver", &Class::varIterationNumberForVelocitySolver, py::return_value_policy::reference)
		.def("varSubStepping", &Class::varSubStepping, py::return_value_policy::reference)
		.def("varLinearDamping", &Class::varLinearDamping, py::return_value_policy::reference)
		.def("varAngularDamping", &Class::varAngularDamping, py::return_value_policy::reference)
		.def("varDampingRatio", &Class::varDampingRatio, py::return_value_policy::reference)
		.def("varHertz", &Class::varHertz, py::return_value_policy::reference)

		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inMass", &Class::inMass, py::return_value_policy::reference)
		.def("inCenter", &Class::inCenter, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inAngularVelocity", &Class::inAngularVelocity, py::return_value_policy::reference)
		.def("inRotationMatrix", &Class::inRotationMatrix, py::return_value_policy::reference)
		.def("inInertia", &Class::inInertia, py::return_value_policy::reference)
		.def("inInitialInertia", &Class::inInitialInertia, py::return_value_policy::reference)
		.def("inQuaternion", &Class::inQuaternion, py::return_value_policy::reference)
		.def("inContacts", &Class::inContacts, py::return_value_policy::reference)
		.def("inDiscreteElements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("inFrictionCoefficients", &Class::inFrictionCoefficients, py::return_value_policy::reference)
		// protected
		.def("constrain", &TJSoftConstraintSolverPublicist::constrain, py::return_value_policy::reference);
}


#include "RigidBody/Module/KeyDriver.h"
template <typename TDataType>
void declare_key_driver(py::module& m, std::string typestr) {
	using Class = dyno::KeyDriver<TDataType>;
	using Parent = dyno::KeyboardInputModule;

	class KeyDriverTrampoline : public Class
	{
	public:
		using Class::Class;

		void onEvent(dyno::PKeyboardEvent event) override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::KeyDriver<TDataType>,
				onEvent,
				event
			);
		}
	};

	class KeyDriverPublicist : public Class
	{
	public:
		using Class::onEvent;
	};

	std::string pyclass_name = std::string("KeyDriver") + typestr;
	py::class_<Class, Parent, KeyDriverTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varHingeKeyConfig", &Class::varHingeKeyConfig, py::return_value_policy::reference)
		.def("inReset", &Class::inReset, py::return_value_policy::reference)
		.def("inTopology", &Class::inTopology, py::return_value_policy::reference)
		// protected
		.def("onEvent", &KeyDriverPublicist::onEvent, py::return_value_policy::reference);
}

#include "RigidBody/RigidBodySystem.h"
template <typename TDataType>
void declare_rigid_body_system(py::module& m, std::string typestr) {
	using Class = dyno::RigidBodySystem<TDataType>;
	using Parent = dyno::Node;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename dyno::Quat<Real> TQuat;

	class RigidBodySystemTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::RigidBodySystem<TDataType>,
				resetStates,
			);
		}

		void postUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::RigidBodySystem<TDataType>,
				postUpdateStates,
				);
		}
	};

	class RigidBodySystemPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::postUpdateStates;
		using Class::clearRigidBodySystem;
	};

	std::string pyclass_name = std::string("RigidBodySystem") + typestr;
	py::class_<Class, Parent, RigidBodySystemTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("addBox", &Class::addBox, py::arg("box"), py::arg("bodyDef"), py::arg("density") = TDataType::Real(100))
		.def("addSphere", &Class::addSphere, py::arg("sphere"), py::arg("bodyDef"), py::arg("density") = TDataType::Real(100))
		.def("addTet", &Class::addTet, py::arg("tet"), py::arg("bodyDef"), py::arg("density") = TDataType::Real(100))
		.def("addCapsule", &Class::addCapsule, py::arg("capsule"), py::arg("bodyDef"), py::arg("density") = TDataType::Real(100))
		.def("create_rigid_body", py::overload_cast<const Coord&, const TQuat&>(&Class::createRigidBody))
		.def("create_rigid_body", py::overload_cast<const dyno::RigidBodyInfo&>(&Class::createRigidBody))
		.def("bindBox", &Class::bindBox)
		.def("bindSphere", &Class::bindSphere)
		.def("bindCapsule", &Class::bindCapsule)
		.def("bindTet", &Class::bindTet)
		.def("createBallAndSocketJoint", &Class::createBallAndSocketJoint, py::return_value_policy::reference)
		.def("createSliderJoint", &Class::createSliderJoint, py::return_value_policy::reference)
		.def("createHingeJoint", &Class::createHingeJoint, py::return_value_policy::reference)
		.def("createFixedJoint", &Class::createFixedJoint, py::return_value_policy::reference)
		.def("createUnilateralFixedJoint", &Class::createUnilateralFixedJoint, py::return_value_policy::reference)
		.def("createPointJoint", &Class::createPointJoint, py::return_value_policy::reference)
		.def("pointInertia", &Class::pointInertia)
		.def("getNodeType", &Class::getNodeType)

		.def("varFrictionEnabled", &Class::varFrictionEnabled, py::return_value_policy::reference)
		.def("varGravityEnabled", &Class::varGravityEnabled, py::return_value_policy::reference)
		.def("varGravityValue", &Class::varGravityValue, py::return_value_policy::reference)
		.def("varFrictionCoefficient", &Class::varFrictionCoefficient, py::return_value_policy::reference)
		.def("varSlop", &Class::varSlop, py::return_value_policy::reference)
		.def("stateTopology", &Class::stateTopology, py::return_value_policy::reference)
		.def("state_friction_coefficents", &Class::stateFrictionCoefficients, py::return_value_policy::reference)
		.def("stateMass", &Class::stateMass, py::return_value_policy::reference)
		.def("stateCenter", &Class::stateCenter, py::return_value_policy::reference)
		.def("stateVelocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("stateAngularVelocity", &Class::stateAngularVelocity, py::return_value_policy::reference)
		.def("stateRotationMatrix", &Class::stateRotationMatrix, py::return_value_policy::reference)
		.def("stateInertia", &Class::stateInertia, py::return_value_policy::reference)
		.def("stateQuaternion", &Class::stateQuaternion, py::return_value_policy::reference)
		.def("stateCollisionMask", &Class::stateCollisionMask, py::return_value_policy::reference)
		.def("stateAttribute", &Class::stateAttribute, py::return_value_policy::reference)
		.def("stateInitialInertia", &Class::stateInitialInertia, py::return_value_policy::reference)

		.def_readwrite("m_numOfSamples", &Class::m_numOfSamples)
		.def_readwrite("m_deviceSamples", &Class::m_deviceSamples)
		.def_readwrite("m_deviceNormals", &Class::m_deviceNormals)
		.def_readwrite("samples", &Class::samples)
		.def_readwrite("normals", &Class::normals)
		.def("getSamplingPointSize", &Class::getSamplingPointSize)
		.def("getSamples", &Class::getSamples)
		.def("getNormals", &Class::getNormals)
		// protected
		.def("resetStates", &RigidBodySystemPublicist::resetStates, py::return_value_policy::reference)
		.def("postUpdateStates", &RigidBodySystemPublicist::postUpdateStates, py::return_value_policy::reference)
		.def("clearRigidBodySystem", &RigidBodySystemPublicist::clearRigidBodySystem, py::return_value_policy::reference);

}

#include "RigidBody/ArticulatedBody.h"
template <typename TDataType>
void declare_articulated_body(py::module& m, std::string typestr) {
	using Class = dyno::ArticulatedBody<TDataType>;
	using Parent1 = dyno::ParametricModel<TDataType>;
	using Parent2 = dyno::RigidBodySystem<TDataType>;

	class ArticulatedBodyTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ArticulatedBody<TDataType>,
				resetStates,
				);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ArticulatedBody<TDataType>,
				updateStates,
				);
		}
	};

	class ArticulatedBodyPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
		using Class::updateInstanceTransform;
		using Class::clearVechicle;
		using Class::transform;
		using Class::varChanged;
		using Class::getInstanceRotation;
	};

	std::string pyclass_name = std::string("ArticulatedBody") + typestr;
	py::class_<Class, Parent1, Parent2, ArticulatedBodyTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("bindSphere", &Class::bindSphere)

		.def("varFilePath", &Class::varFilePath, py::return_value_policy::reference)
		.def("varVehiclesTransform", &Class::varVehiclesTransform, py::return_value_policy::reference)
		.def("stateTextureMesh", &Class::stateTextureMesh, py::return_value_policy::reference)

		.def("stateInstanceTransform", &Class::stateInstanceTransform, py::return_value_policy::reference)
		.def("stateBindingPair", &Class::stateBindingPair, py::return_value_policy::reference)
		.def("stateBindingTag", &Class::stateBindingTag, py::return_value_policy::reference)
		// protected
		.def("resetStates", &ArticulatedBodyPublicist::resetStates, py::return_value_policy::reference)
		.def("updateStates", &ArticulatedBodyPublicist::updateStates, py::return_value_policy::reference)
		.def("updateInstanceTransform", &ArticulatedBodyPublicist::updateInstanceTransform, py::return_value_policy::reference)
		.def("clearVechicle", &ArticulatedBodyPublicist::clearVechicle, py::return_value_policy::reference)
		.def("transform", &ArticulatedBodyPublicist::transform, py::return_value_policy::reference)
		.def("varChanged", &ArticulatedBodyPublicist::varChanged, py::return_value_policy::reference)
		.def("getInstanceRotation", &ArticulatedBodyPublicist::getInstanceRotation, py::return_value_policy::reference);
}

#include "RigidBody/ConfigurableBody.h"
template <typename TDataType>
void declare_configurable_body(py::module& m, std::string typestr) {
	using Class = dyno::ConfigurableBody<TDataType>;
	using Parent = dyno::ArticulatedBody<TDataType>;

	class ConfigurableBodyTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ConfigurableBody<TDataType>,
				resetStates,
				);
		}
	};

	class ConfigurableBodyPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateConfig;
	};

	std::string pyclass_name = std::string("ConfigurableBody") + typestr;
	py::class_<Class, Parent, ConfigurableBodyTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varVehicleConfiguration", &Class::varVehicleConfiguration, py::return_value_policy::reference)

		.def("inTextureMesh", &Class::inTextureMesh, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &ConfigurableBodyPublicist::resetStates, py::return_value_policy::reference)
		.def("updateConfig", &ConfigurableBodyPublicist::updateConfig, py::return_value_policy::reference);
}

#include "RigidBody/Gear.h"
template <typename TDataType>
void declare_gear(py::module& m, std::string typestr) {
	using Class = dyno::Gear<TDataType>;
	using Parent = dyno::ArticulatedBody<TDataType>;

	class GearTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Gear<TDataType>,
				resetStates,
				);
		}
	};

	class GearPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("Gear") + typestr;
	py::class_<Class, Parent, GearTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		// protected
		.def("resetStates", &GearPublicist::resetStates, py::return_value_policy::reference);
}

#include "RigidBody/MultibodySystem.h"
template <typename TDataType>
void declare_multibody_system(py::module& m, std::string typestr) {
	using Class = dyno::MultibodySystem<TDataType>;
	using Parent = dyno::RigidBodySystem<TDataType>;

	class MultibodySystemTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::MultibodySystem<TDataType>,
				resetStates,
				);
		}

		void postUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::MultibodySystem<TDataType>,
				postUpdateStates,
				);
		}
	};

	class MultibodySystemPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::postUpdateStates;
		using Class::validateInputs;
	};

	std::string pyclass_name = std::string("MultibodySystem") + typestr;
	py::class_<Class, Parent, MultibodySystemTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varVehiclesTransform", &Class::varVehiclesTransform, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)

		.def("importVehicles", &Class::importVehicles, py::return_value_policy::reference)
		.def("getVehicles", &Class::getVehicles)
		.def("addVehicle", &Class::addVehicle)
		.def("removeVehicle", &Class::removeVehicle)

		.def("stateInstanceTransform", &Class::stateInstanceTransform, py::return_value_policy::reference)
		// protected
		.def("resetStates", &MultibodySystemPublicist::resetStates, py::return_value_policy::reference)
		.def("postUpdateStates", &MultibodySystemPublicist::postUpdateStates, py::return_value_policy::reference)
		.def("validateInputs", &MultibodySystemPublicist::validateInputs, py::return_value_policy::reference);
}

#include "RigidBody/RigidBody.h"
template <typename TDataType>
void declare_rigid_body(py::module& m, std::string typestr) {
	using Class = dyno::RigidBody<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;

	class RigidBodyTrampoline : public Class
	{
	public:
		using Class::Class;

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::RigidBody<TDataType>,
				updateStates,
				);
		}
	};

	class RigidBodyPublicist : public Class
	{
	public:
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("RigidBody") + typestr;
	py::class_<Class, Parent, RigidBodyTrampoline,std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)

		.def("varGravity", &Class::varGravity, py::return_value_policy::reference)
		.def("stateMass", &Class::stateMass, py::return_value_policy::reference)
		.def("stateCenter", &Class::stateCenter, py::return_value_policy::reference)
		.def("stateVelocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("stateAngularVelocity", &Class::stateAngularVelocity, py::return_value_policy::reference)
		.def("stateRotationMatrix", &Class::stateRotationMatrix, py::return_value_policy::reference)
		.def("stateInertia", &Class::stateInertia, py::return_value_policy::reference)
		.def("stateQuaternion", &Class::stateQuaternion, py::return_value_policy::reference)
		.def("stateInitialInertia", &Class::stateInitialInertia, py::return_value_policy::reference)
		// protected
		.def("updateStates", &RigidBodyPublicist::updateStates, py::return_value_policy::reference);
}

#include "RigidBody/RigidMesh.h"
template <typename TDataType>
void declare_rigid_mesh(py::module& m, std::string typestr) {
	using Class = dyno::RigidMesh<TDataType>;
	using Parent = dyno::RigidBody<TDataType>;

	class RigidMeshTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::RigidMesh<TDataType>,
				resetStates,
				);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::RigidMesh<TDataType>,
				updateStates,
				);
		}
	};

	class RigidMeshPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("RigidMesh") + typestr;
	py::class_<Class, Parent, RigidMeshTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varEnvelopeName", &Class::varEnvelopeName, py::return_value_policy::reference)
		.def("varMeshName", &Class::varMeshName, py::return_value_policy::reference)
		.def("varDensity", &Class::varDensity, py::return_value_policy::reference)

		.def("stateInitialEnvelope", &Class::stateInitialEnvelope, py::return_value_policy::reference)
		.def("stateEnvelope", &Class::stateEnvelope, py::return_value_policy::reference)

		.def("stateInitialMesh", &Class::stateInitialMesh, py::return_value_policy::reference)
		.def("stateMesh", &Class::stateMesh, py::return_value_policy::reference)
		// protected
		.def("resetStates", &RigidMeshPublicist::resetStates, py::return_value_policy::reference)
		.def("updateStates", &RigidMeshPublicist::updateStates, py::return_value_policy::reference);
}

#include "RigidBody/Vehicle.h"
template <typename TDataType>
void declare_jeep(py::module& m, std::string typestr) {
	using Class = dyno::Jeep<TDataType>;
	using Parent = dyno::ArticulatedBody<TDataType>;

	class JeepTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Jeep<TDataType>,
				resetStates,
				);
		}
	};

	class JeepPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("Jeep") + typestr;
	py::class_<Class, Parent, JeepTrampoline,std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		// protected
		.def("resetStates", &JeepPublicist::resetStates, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_tank(py::module& m, std::string typestr) {
	using Class = dyno::Tank<TDataType>;
	using Parent = dyno::ArticulatedBody<TDataType>;

	class TankTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Tank<TDataType>,
				resetStates,
				);
		}
	};

	class TankPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("Tank") + typestr;
	py::class_<Class, Parent, TankTrampoline,std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		// protected
		.def("resetStates", &TankPublicist::resetStates, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_tracked_tank(py::module& m, std::string typestr) {
	using Class = dyno::TrackedTank<TDataType>;
	using Parent = dyno::ArticulatedBody<TDataType>;

	class TrackedTankTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TrackedTank<TDataType>,
				resetStates,
				);
		}
	};

	class TrackedTankPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("TrackedTank") + typestr;
	py::class_<Class, Parent, TrackedTankTrampoline,std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("statecaterpillarTrack", &Class::statecaterpillarTrack, py::return_value_policy::reference)
		// protected
		.def("resetStates", &TrackedTankPublicist::resetStates, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_uav(py::module& m, std::string typestr) {
	using Class = dyno::UAV<TDataType>;
	using Parent = dyno::ArticulatedBody<TDataType>;

	class UAVTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::UAV<TDataType>,
				resetStates,
				);
		}
	};

	class UAVPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("UAV") + typestr;
	py::class_<Class, Parent, UAVTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		// protected
		.def("resetStates", &UAVPublicist::resetStates, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_uuv(py::module& m, std::string typestr) {
	using Class = dyno::UUV<TDataType>;
	using Parent = dyno::ArticulatedBody<TDataType>;

	class UUVTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::UUV<TDataType>,
				resetStates,
				);
		}
	};

	class UUVPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("UUV") + typestr;
	py::class_<Class, Parent, UUVTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		// protected
		.def("resetStates", &UUVPublicist::resetStates, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_bicycle(py::module& m, std::string typestr) {
	using Class = dyno::Bicycle<TDataType>;
	using Parent = dyno::ArticulatedBody<TDataType>;

	class BicycleTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Bicycle<TDataType>,
				resetStates,
				);
		}
	};

	class BicyclePublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("Bicycle") + typestr;
	py::class_<Class, Parent, BicycleTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("outReset", &Class::outReset, py::return_value_policy::reference)
		// protected
		.def("resetStates", &BicyclePublicist::resetStates, py::return_value_policy::reference);
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
