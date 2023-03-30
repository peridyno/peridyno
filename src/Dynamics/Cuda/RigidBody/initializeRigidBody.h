#pragma once
#include <Plugin/PluginEntry.h>

namespace dyno
{
	class RigidBodyInitializer : public PluginEntry
	{
	public:
		RigidBodyInitializer();

		void initializeNodeCreators();
	};
}

namespace RigidBody
{
	PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
}
