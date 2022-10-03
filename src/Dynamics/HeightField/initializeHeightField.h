#pragma once
#include <Plugin/PluginInterface.h>

namespace dyno
{
	class HeightFieldInitializer : public IPlugin
	{
	public:
		HeightFieldInitializer();

		void initializeNodeCreators();
	};
}

DYNO_PLUGIN_EXPORT
auto initDynoPlugin() -> void
{
}