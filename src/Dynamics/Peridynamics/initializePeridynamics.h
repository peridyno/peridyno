#pragma once
#include <Object.h>

namespace dyno 
{
	class PeridynamicsInitializer : public Object
	{
	public:
		PeridynamicsInitializer();
	};

	const static PeridynamicsInitializer peridynamicsInitializer;
}