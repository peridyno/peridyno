#pragma once
#include <Object.h>

namespace dyno
{
	class RigidBodyInitializer : public Object
	{
	public:
		RigidBodyInitializer();

		void initializeNodeCreators();
	};

	const static RigidBodyInitializer heightFieldInitializer;
}