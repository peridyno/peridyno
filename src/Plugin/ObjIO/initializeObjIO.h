#pragma once
#include <Plugin/PluginEntry.h>

namespace dyno 
{
	class OBJInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();

	protected:
		void initializeActions() override;

	private:
		OBJInitializer() {};

		static std::atomic<OBJInitializer*> gInstance;
		static std::mutex gMutex;
	};
}

namespace ObjIO
{
	dyno::PluginEntry* initStaticPlugin();

	PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
}
