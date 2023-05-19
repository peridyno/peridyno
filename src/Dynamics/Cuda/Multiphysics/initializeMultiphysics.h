#pragma once
#include <Plugin/PluginEntry.h>

namespace dyno
{
	class MultiphysicsInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();

	protected:
		void initializeActions() override;

	private:
		MultiphysicsInitializer();

		static std::atomic<MultiphysicsInitializer*> gInstance;
		static std::mutex gMutex;
	};
}

namespace Multiphysics
{
	dyno::PluginEntry* initStaticPlugin();

	PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
}
