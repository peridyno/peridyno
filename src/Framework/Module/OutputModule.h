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

		virtual void output() {};	

		int getFrameNumber();

		void setFilePostfix(std::string postfix);


	protected:
		void updateImpl() final;

		DEF_VAR_IN(unsigned, FrameNumber, "Input FrameNumber");

		DEF_VAR(std::string, OutputPath, "", "OutputPath");
		DEF_VAR(bool,ReCount,true,"ReCount");

		DEF_VAR(unsigned, Start, 1, "Start");
		DEF_VAR(unsigned, End, 1000, "End");
		DEF_VAR(unsigned, FrameStep, 1, "FramStep");



	private:

		int updateFrameNumber();
		void updateSkipFrame();

		int mFileIndex = 0;
		int mCount = -1;
		bool mSkipFrame = false;

	};
}
