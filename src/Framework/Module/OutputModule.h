#pragma once
#include "Module.h"

#include "FilePath.h"

namespace dyno
{
	class OutputModule : public Module
	{
	public:
		OutputModule();
		~OutputModule() override;

		DEF_VAR(FilePath, OutputPath, "", "OutputPath");

		DEF_VAR(std::string, Prefix, "", "Prefix of the file name");
		
		DEF_VAR(uint, StartFrame, 0, "Start frame");
		DEF_VAR(uint, EndFrame, 9999, "End frame");

		DEF_VAR(uint, Stride, 1, "Stride");

		DEF_VAR(bool, Reordering, true, "If set true, the output file name will be re-indexed in sequence starting from zero");

		DEF_VAR_IN(uint, FrameNumber, "Input FrameNumber");

		std::string getModuleType() override { return "OutputModule"; }

	protected:
		void updateImpl() final;

		virtual void output() {};

		/**
		 * construct the file name with an index appended at the end
		 */
		std::string constructFileName();
	};
}
