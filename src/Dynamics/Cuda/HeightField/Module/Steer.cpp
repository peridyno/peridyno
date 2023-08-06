#include "Steer.h"

namespace dyno
{
	IMPLEMENT_TCLASS(Steer, TDataType);

	template<typename TDataType>
	Steer<TDataType>::Steer()
		: KeyboardInputModule()
	{

	}

	template<typename TDataType>
	void Steer<TDataType>::onEvent(PKeyboardEvent event)
	{
		auto quat = this->inQuaternion()->getData();

		Coord vel = this->inVelocity()->getData();

		Matrix rot = quat.toMatrix3x3();

		Coord vel_prime = rot.transpose() * vel;

		vel_prime[2] += 1.0f;

		this->inVelocity()->setValue(rot * vel_prime);
	}
	
	DEFINE_CLASS(Steer);
}