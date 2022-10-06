#pragma once
#include <Platform.h>
#include <Plugin/PluginInterface.h>

namespace dyno 
{
	class ParticleSystemInitializer : public IPlugin
	{
	public:
		static IPlugin* instance();

		void initializeNodeCreators();

	private:
		ParticleSystemInitializer();

		static std::atomic<ParticleSystemInitializer*> gInstance;
		static std::mutex gMutex;
	};
}

namespace PaticleSystem
{
	bool initStaticPlugin();

	PERIDYNO_API void initDynoPlugin();
}

