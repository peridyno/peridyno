#pragma once
#include "Plugin/PluginEntry.h"

namespace dyno
{
	class GLRenderEngineInitializer : public PluginEntry
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