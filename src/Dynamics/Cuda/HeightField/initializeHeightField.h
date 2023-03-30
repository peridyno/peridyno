#pragma once
#include <Plugin/PluginEntry.h>

namespace dyno
{
	class HeightFieldInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();

	protected:
		void initializeNodeCreators() override;

	private:
		HeightFieldInitializer();

		static std::atomic<HeightFieldInitializer*> gInstance;
		static std::mutex gMutex;
	};
}

namespace HeightFieldLibrary
{
	dyno::PluginEntry* initStaticPlugin();

	PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
}
