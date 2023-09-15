#include "Module/OutputModule.h"

namespace dyno
{
	OutputModule::OutputModule()
		: Module()
	{
		this->inFrameNumber()->tagOptional(true);
	}

	OutputModule::~OutputModule()
	{
	}

	void OutputModule::updateImpl()
	{
		this->updateFrameNumber();
		this->updateSkipFrame();
		
		if (mSkipFrame)
			return;

		//OutputFile
		this->output();

	}


	int OutputModule::getFrameNumber()
	{
		return mFileIndex;
	}


	int OutputModule::updateFrameNumber()
	{

		auto frame_step = this->varFrameStep()->getData();
		mCount++;
		//If the input is not empty,  use InFrameNumber
		if (!this->inFrameNumber()->isEmpty())
		{
			mFileIndex = inFrameNumber()->getValue();

			if (frame_step > 1)
			{
				if (this->varReCount()->getValue())				
					mFileIndex = mFileIndex / frame_step;
				//else
				//use inFrameNumber

			}
			//else
			//use inFrameNumber

			return mFileIndex;
		}
		//no input
		//use count;
		else
		{
			mFileIndex = mCount;
			return mFileIndex;
		}

	}

	void OutputModule::updateSkipFrame()
	{
		mSkipFrame = false;

		auto frameStep = this->varFrameStep()->getValue();

		if (!this->inFrameNumber()->isEmpty())
		{
			if (frameStep > 1 && this->inFrameNumber()->getValue() % frameStep != 0)
				mSkipFrame = true;
		}	
	}






}