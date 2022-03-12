#pragma once
#include <Object.h>

namespace dyno
{
	class HeightFieldInitializer : public Object
	{
	public:
		HeightFieldInitializer();
	};

	const static HeightFieldInitializer heightFieldInitializer;
}