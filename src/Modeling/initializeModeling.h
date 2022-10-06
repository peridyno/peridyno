#pragma once
#include <Plugin/PluginInterface.h>

namespace dyno 
{
	class ModelingInitializer : public IPlugin
	{
	public:
		ModelingInitializer();

		void initializeNodeCreators() override;
	};

	//const static ModelingInitializer modelingInitializer;
}

namespace Modeling
{
	PERIDYNO_API void initDynoPlugin();
}
