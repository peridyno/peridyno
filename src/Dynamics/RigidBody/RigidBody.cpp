#include "RigidBody.h"
#include "Quat.h"

namespace dyno
{
	IMPLEMENT_TCLASS(RigidBody, TDataType)

	template<typename TDataType>
	RigidBody<TDataType>::RigidBody()
		: Node()
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

		Real dt = this->stateTimeStep()->getData();

		auto& center = this->stateCenter()->getData();
		auto& trans_velocity = this->stateVelocity()->getData();
		auto& angular_velocity = this->stateAngularVelocity()->getData();

		auto& quat = this->stateQuaternion()->getData();

		auto& rot = this->stateRotationMatrix()->getData();

		auto& inertia = this->stateInertia()->getData();
		auto& initial_inertia = this->stateInitialInertia()->getData();

		trans_velocity += this->varGravity()->getData() * dt;
		center += trans_velocity * dt;

		quat = quat.normalize();
		rot = quat.toMatrix3x3();

		quat += dt * 0.5f *
			Quat(angular_velocity[0], angular_velocity[1], angular_velocity[2], 0.0)
			* quat;

		inertia = rot * initial_inertia * rot.inverse();
	}

	DEFINE_CLASS(RigidBody);
}