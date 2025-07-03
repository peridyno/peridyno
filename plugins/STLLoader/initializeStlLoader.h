#pragma once
#include <Plugin/PluginEntry.h>

namespace dyno 
{
	class STLInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();

	protected:
		void initializeActions() override;

	private:
		STLInitializer() {};

		static std::atomic<STLInitializer*> gInstance;
		static std::mutex gMutex;
	};
}

namespace StlLoader
{
	dyno::PluginEntry* initStaticPlugin();

	PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
}
