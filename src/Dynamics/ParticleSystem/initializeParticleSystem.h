#pragma once
#include <Object.h>

namespace dyno 
{
	class ParticleSystemInitializer : public Object
	{
	public:
		ParticleSystemInitializer();
	};

	const static ParticleSystemInitializer particleSystemInitializer;
}