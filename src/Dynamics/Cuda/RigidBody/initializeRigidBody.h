#pragma once
#include <Plugin/PluginEntry.h>

namespace dyno
{
	class RigidBodyInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();

	protected:
		void initializeActions() override;

	private:
		RigidBodyInitializer();

		static std::atomic<RigidBodyInitializer*> gInstance;
		static std::mutex gMutex;
	};
}

namespace RigidBody
{
	dyno::PluginEntry* initStaticPlugin();

	PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
}
