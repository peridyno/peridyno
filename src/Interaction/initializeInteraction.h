#pragma once
#include <Plugin/PluginInterface.h>

namespace dyno 
{
	class InteractionInitializer : public IPlugin
	{
	public:
		InteractionInitializer();

		void initializeNodeCreators();
	};
}


DYNO_PLUGIN_EXPORT
auto initPlugin() -> void
{
	static dyno::InteractionInitializer interactionInitializer;
}