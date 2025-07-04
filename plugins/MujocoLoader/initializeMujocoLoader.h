#pragma once
#include <Plugin/PluginEntry.h>


namespace dyno
{
	class MujocoInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();

	protected:
		void initializeActions() override;

	private:
		MujocoInitializer() {};

		static std::atomic<MujocoInitializer*> gInstance;
		static std::mutex gMutex;
	};
}

namespace MujocoLoader
{
	dyno::PluginEntry* initStaticPlugin();

	PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
}
