#pragma once
#include <Plugin/PluginInterface.h>

namespace dyno 
{
	class ParticleSystemInitializer : public IPlugin
	{
	public:
		ParticleSystemInitializer();

		void initializeNodeCreators();
	};
}

DYNO_PLUGIN_EXPORT
auto initDynoPlugin() -> void
{
	static dyno::ParticleSystemInitializer particleSystemInitializer;
}
