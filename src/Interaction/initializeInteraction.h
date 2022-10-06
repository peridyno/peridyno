#pragma once
#include <Plugin/PluginInterface.h>

namespace dyno 
{
	class InteractionInitializer : public IPlugin
	{
	public:
		InteractionInitializer();

		void initializeNodeCreators();
	};
}

namespace Interaction
{
	PERIDYNO_API void initDynoPlugin();
}
