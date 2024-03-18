#include "Module/OutputModule.h"

namespace dyno
{
	OutputModule::OutputModule()
		: Module()
	{
		this->varStride()->setRange(1, 1024);

		auto path = this->varOutputPath()->getValue();
		path.set_as_path(true);
		this->varOutputPath()->setValue(path);

		this->inFrameNumber()->tagOptional(true);
	}

	OutputModule::~OutputModule()
	{
	}

	void OutputModule::updateImpl()
	{
		uint startFrame = this->varStartFrame()->getValue();
		uint endFrame = this->varEndFrame()->getValue();

		uint frame = this->inFrameNumber()->getValue();

		uint stride = this->varStride()->getValue();

		if (frame >= startFrame && frame <= endFrame)
		{
			if ((frame - startFrame) % stride == 0)
			{
				//OutputFile
				this->output();
			}
		}
	}

	std::string OutputModule::constructFileName()
	{
		uint num = this->inFrameNumber()->getValue();
		uint stride = this->varStride()->getValue();

		uint index = num;

		if (this->varReordering()->getValue()){
			index = num / stride;
		}

		std::string prefix = this->varPrefix()->getValue();

		auto path = this->varOutputPath()->getValue().path();

		std::stringstream ss; ss << index;

		//TODO: check whether the path already contains "\\" or "/" at the end
		std::string filename = path.string() + "\\" + prefix + ss.str();
		
		return filename;
	}
}