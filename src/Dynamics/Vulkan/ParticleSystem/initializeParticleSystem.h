#pragma once
#include <Plugin/PluginEntry.h>

namespace dyno 
{
	class ParticleSystemInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();

		protected:
			void initializeActions() override;

	private:
		ParticleSystemInitializer();

		static std::atomic<ParticleSystemInitializer*> gInstance;
		static std::mutex gMutex;
	};
}

namespace PaticleSystem
{
	dyno::PluginEntry* initStaticPlugin();

	PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
}

