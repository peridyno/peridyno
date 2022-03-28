#include "PSimulationThread.h"

#include "SceneGraph.h"
#include "SceneGraphFactory.h"

namespace dyno
{
	QWaitCondition m_wait_condition;

	PSimulationThread::PSimulationThread()
		: mFrameNum(1000)
	{

	}

	PSimulationThread* PSimulationThread::instance()
	{
		static PSimulationThread m_instance;
		return &m_instance;
	}

	void PSimulationThread::pause()
	{
		mPaused = true;
	}

	void PSimulationThread::resume()
	{
		mPaused = false;
	}

	void PSimulationThread::stop()
	{
		mRunning = false;
	}

	void PSimulationThread::run()
	{
		auto scn = SceneGraphFactory::instance()->active();
//		scn->initialize();

		int f = 0;
		while(mRunning && f < mFrameNum)
		{
			if (!mRendering)
			{
				if (mReset)
				{
					scn->reset();
					mReset = false;

					emit(oneFrameFinished());
				}

				if (!mPaused)
				{
					scn->takeOneFrame();

					f++;
					currentFrameNum = f;

					this->startRendering();
					
					emit(oneFrameFinished());
				}
				
			}
//	
		}
	}

	void PSimulationThread::reset()
	{
		mReset = true;
	}

	void PSimulationThread::startRendering()
	{
		mRendering = true;
	}

	void PSimulationThread::stopRendering()
	{
		mRendering = false;
	}

	void PSimulationThread::setTotalFrames(int num)
	{
		mFrameNum = num;
	}

	int PSimulationThread::getCurrentFrameNum() {
		return currentFrameNum;
	}
}
