#pragma once
#include <Plugin/PluginInterface.h>

namespace dyno 
{
	class ModelingInitializer : public IPlugin
	{
	public:
		ModelingInitializer();

		void initializeNodeCreators() override;
	};
}

DYNO_PLUGIN_EXPORT
auto initPlugin() -> void
{
	static dyno::ModelingInitializer particleSystemInitializer;
}
