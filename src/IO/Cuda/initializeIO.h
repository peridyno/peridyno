#pragma once
#include <Plugin/PluginEntry.h>

namespace dyno
{
	class IOInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();

	protected:
		void initializeActions() override;

	private:
		IOInitializer();

		static std::atomic<IOInitializer*> gInstance;
		static std::mutex gMutex;
	};
}

namespace IO
{
	dyno::PluginEntry* initStaticPlugin();
}
