#pragma once
#include <Plugin/PluginEntry.h>

namespace dyno
{
	class PeridynamicsInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();

	protected:
		void initializeActions() override;

	private:
		PeridynamicsInitializer();

		static std::atomic<PeridynamicsInitializer*> gInstance;
		static std::mutex gMutex;
	};
}

namespace Peridynamics
{
	dyno::PluginEntry* initStaticPlugin();

	PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
}
