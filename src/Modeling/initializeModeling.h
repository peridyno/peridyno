#pragma once
#include <Object.h>

namespace dyno 
{
	class ModelingInitializer : public Object
	{
	public:
		ModelingInitializer();

		void initializeNodeCreators();
	};

	const static ModelingInitializer particleSystemInitializer;
}