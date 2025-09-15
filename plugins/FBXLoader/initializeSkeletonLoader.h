#pragma once
#include <Plugin/PluginEntry.h>

namespace dyno
{
	class FBXInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();

	protected:
		void initializeActions() override;

	private:
		FBXInitializer() {};

		static std::atomic<FBXInitializer*> gInstance;
		static std::mutex gMutex;
	};
}

namespace FBXLoader
{
	dyno::PluginEntry* initStaticPlugin();

	PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
}
