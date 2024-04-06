#pragma once
#include <Plugin/PluginEntry.h>

namespace dyno
{
	class DualParticleSystemInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();

	protected:
		void initializeActions() override;

	private:
		DualParticleSystemInitializer();

		static std::atomic<DualParticleSystemInitializer*> gInstance;
		static std::mutex gMutex;
	};
}

namespace DualParticleSystem
{
	dyno::PluginEntry* initStaticPlugin();

	PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
}
