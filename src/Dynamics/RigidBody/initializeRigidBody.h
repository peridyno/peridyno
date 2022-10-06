#pragma once
#include <Plugin/PluginInterface.h>

namespace dyno
{
	class RigidBodyInitializer : public IPlugin
	{
	public:
		RigidBodyInitializer();

		void initializeNodeCreators();
	};
}

PERIDYNO_API
auto initDynoPlugin() -> void
{
	static dyno::RigidBodyInitializer rigidBodyInitializer;
}
