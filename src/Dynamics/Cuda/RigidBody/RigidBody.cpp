#include "RigidBody.h"
#include "Quat.h"

namespace dyno
{
	IMPLEMENT_TCLASS(RigidBody, TDataType)

	template<typename TDataType>
	RigidBody<TDataType>::RigidBody()
		: ParametricModel<TDataType>()
	{
	}

	template<typename TDataType>
	RigidBody<TDataType>::~RigidBody()
	{
	}

	template<typename TDataType>
	void RigidBody<TDataType>::updateStates()
	{
		Node::updateStates();

		Real dt = this->stateTimeStep()->getValue();

		auto center = this->stateCenter()->getValue();
		auto trans_velocity = this->stateVelocity()->getValue();
		auto angular_velocity = this->stateAngularVelocity()->getValue();

		auto quat = this->stateQuaternion()->getValue();

		auto rot = this->stateRotationMatrix()->getValue();

		auto inertia = this->stateInertia()->getValue();
		auto initial_inertia = this->stateInitialInertia()->getValue();

		trans_velocity += this->varGravity()->getValue() * dt;
		center += trans_velocity * dt;

		quat = quat.normalize();
		rot = quat.toMatrix3x3();

		quat += dt * 0.5f *
			Quat(angular_velocity[0], angular_velocity[1], angular_velocity[2], 0.0)
			* quat;

		inertia = rot * initial_inertia * rot.inverse();

		this->stateCenter()->setValue(center);
		this->stateVelocity()->setValue(trans_velocity);
		this->stateQuaternion()->setValue(quat);
		this->stateRotationMatrix()->setValue(rot);
		this->stateInertia()->setValue(inertia);
	}

	DEFINE_CLASS(RigidBody);
}