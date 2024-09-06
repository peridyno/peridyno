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

		Coord omega = this->inAngularVelocity()->getData();

		Matrix rot = quat.toMatrix3x3();

		Coord vel_prime = rot.transpose() * vel;
		Coord omega_prime = rot.transpose() * omega;

		switch (event.key)
		{
		case PKeyboardType::PKEY_W:
			vel_prime[2] += 0.5f;
			vel_prime[2] = vel_prime[2] > 5.0 ? 5.0 : vel_prime[2];
			vel_prime[2] = vel_prime[2] < -5.0 ? -5.0 : vel_prime[2];
			break;
		case PKeyboardType::PKEY_S:
			vel_prime[2] *= 0.95f;
			break;
		case PKeyboardType::PKEY_A:
			omega_prime.y += 0.5;
			break;
		case PKeyboardType::PKEY_D:
			omega_prime.y -= 0.5;
			break;
		default:
			break;
		}

		this->inVelocity()->setValue(rot * vel_prime);
		this->inAngularVelocity()->setValue(rot * omega_prime);
	}
	
	DEFINE_CLASS(Steer);
}