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

		DEF_VAR(unsigned, Start, 1, "FramStep");
		DEF_VAR(unsigned, End, 1000, "FramStep");
		DEF_VAR(unsigned, FrameStep, 1, "FramStep");
	};
}
