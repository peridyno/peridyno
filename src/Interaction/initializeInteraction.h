#pragma once
#include <Plugin/PluginEntry.h>

namespace dyno 
{
	class InteractionInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();

	protected:
		void initializeActions() override;

	private:
		InteractionInitializer() {};

		static std::atomic<InteractionInitializer*> gInstance;
		static std::mutex gMutex;
	};
}

namespace Interaction
{
	dyno::PluginEntry* initStaticPlugin();

	PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
}
