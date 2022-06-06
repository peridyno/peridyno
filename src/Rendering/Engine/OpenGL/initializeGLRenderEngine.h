#pragma once
#include <Object.h>

namespace dyno
{
	class GLRenderEngineInitializer : public Object
	{
	public:
		GLRenderEngineInitializer();

		void initializeNodeCreators();
	};

	const static GLRenderEngineInitializer renderEngineInitializer;
}