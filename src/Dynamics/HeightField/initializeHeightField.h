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

PERIDYNO_API
auto initDynoPlugin() -> void
{
}