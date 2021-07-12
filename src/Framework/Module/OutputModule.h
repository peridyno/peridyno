#pragma once
#include "Module.h"

namespace dyno
{
	class OutputModule : public Module
	{
	public:
		OutputModule();
		virtual ~OutputModule();

		std::string getModuleType() override { return "OuputModule"; }
	};
}
