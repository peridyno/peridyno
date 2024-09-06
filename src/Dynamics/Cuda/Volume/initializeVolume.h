#pragma once
#include <Plugin/PluginEntry.h>

namespace dyno
{
	class VolumeInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();

	protected:
		void initializeActions() override;

	private:
		VolumeInitializer();

		static std::atomic<VolumeInitializer*> gInstance;
		static std::mutex gMutex;
	};
}


namespace Volume
{
	dyno::PluginEntry* initStaticPlugin();

	PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
}
