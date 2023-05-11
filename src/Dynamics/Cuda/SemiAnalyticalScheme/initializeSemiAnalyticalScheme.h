#pragma once
#include <Object.h>

#include <Plugin/PluginEntry.h>

namespace dyno 
{
	class SemiAnalyticalSchemeInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();

	protected:
		void initializeActions() override;

	private:
		SemiAnalyticalSchemeInitializer() {};

		static std::atomic<SemiAnalyticalSchemeInitializer*> gInstance;
		static std::mutex gMutex;
	};

	namespace SemiAnalyticalScheme
	{
		dyno::PluginEntry* initStaticPlugin();

		PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
	}
}