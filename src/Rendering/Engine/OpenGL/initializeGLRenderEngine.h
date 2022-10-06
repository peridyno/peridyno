#pragma once
#include "Plugin/PluginInterface.h"

namespace dyno
{
	class GLRenderEngineInitializer : public IPlugin
	{
	public:
		GLRenderEngineInitializer();

		void initializeNodeCreators();
	};

	const static GLRenderEngineInitializer renderEngineInitializer;
}

PERIDYNO_API
auto initDynoPlugin() -> void
{
}