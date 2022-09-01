#pragma once
#include "Module.h"

namespace dyno
{
	class OutputModule : public Module
	{
	public:
		OutputModule();
		virtual ~OutputModule();

		std::string getModuleType() override { return "OutputModule"; }

		virtual void flush() {};

	protected:
		void updateImpl() override;

		DEF_VAR(std::string, OutputPath, "", "");
		DEF_VAR(std::string, Prefix, "", "");
	};
}
