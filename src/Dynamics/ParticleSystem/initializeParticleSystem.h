#pragma once
#include <Object.h>

namespace dyno 
{
	class ParticleSystemInitializer : public Object
	{
	public:
		ParticleSystemInitializer();

		void initializeNodeCreators();
	};

	const static ParticleSystemInitializer particleSystemInitializer;
}