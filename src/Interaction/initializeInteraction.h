#pragma once
#include <Object.h>

namespace dyno 
{
	class InteractionInitializer : public Object
	{
	public:
		InteractionInitializer();

		void initializeNodeCreators();
	};

	const static InteractionInitializer particleSystemInitializer;
}