#pragma once
#include <Object.h>

namespace dyno
{
	class HeightFieldInitializer : public Object
	{
	public:
		HeightFieldInitializer();

		void initializeNodeCreators();
	};

	const static HeightFieldInitializer heightFieldInitializer;
}