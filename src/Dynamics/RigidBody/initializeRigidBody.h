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

DYNO_PLUGIN_EXPORT
auto initDynoPlugin() -> void
{
	static dyno::RigidBodyInitializer rigidBodyInitializer;
}
