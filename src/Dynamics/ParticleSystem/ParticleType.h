#pragma once
#include <cuda_runtime.h>
#include "Platform.h"
namespace dyno
{
	/*!
	*	\class  ParticleType
	*	\brief  2 types of particle: Real particle (with pressure feild, advection.); virtual particle (with velocity field).  
	*
	*/



	class ParticleType
	{
	public:
		DYN_FUNC ParticleType() { m_type = true; }
		DYN_FUNC ~ParticleType() {};

		enum Type
		{
			VIRTUAL = false,
			REAL = true
		};

		DYN_FUNC inline void SetParticleType(Type type) { m_type = type; }
		DYN_FUNC inline Type GetParticleType() { return (Type)(m_type); }

		DYN_FUNC inline bool IsVirtual() { return  m_type == Type::VIRTUAL; }
		DYN_FUNC inline bool IsReal() { return  m_type == Type::REAL; }
	private:
		bool m_type;
	};
}

