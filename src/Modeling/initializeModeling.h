#pragma once
#include <Plugin/PluginEntry.h>

namespace dyno 
{
	class ModelingInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();

	protected:
		void initializeActions() override;

	private:
		ModelingInitializer() {};

		static std::atomic<ModelingInitializer*> gInstance;
		static std::mutex gMutex;
	};
}

namespace Modeling
{
	dyno::PluginEntry* initStaticPlugin();

	PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
}
