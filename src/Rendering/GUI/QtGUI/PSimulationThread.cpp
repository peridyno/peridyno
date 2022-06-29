#include "PSimulationThread.h"

#include "SceneGraph.h"
#include "SceneGraphFactory.h"

namespace dyno
{
	QWaitCondition m_wait_condition;

	PSimulationThread::PSimulationThread()
		: mTotalFrame(1000)
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

		while (mRunning)
		{
			if (!mUpdatingGraphicsContext)
			{
				if (mReset)
				{
					if (mActiveNode == nullptr) {
						scn->reset();

						mFinished = false;
					}
					else {
						scn->reset(mActiveNode);
					}

					mReset = false;

					emit(oneFrameFinished());
				}

				if (!mPaused && scn->getFrameNumber() < mTotalFrame)
				{
					scn->takeOneFrame();

					this->startUpdatingGraphicsContext();

					emit(oneFrameFinished());
				}
				else if (!mFinished && scn->getFrameNumber() >= mTotalFrame)
				{
					mPaused = true;
					mFinished = true;

					emit(simulationFinished());
				}
			}
		}
	}

	void PSimulationThread::reset(int num)
	{
		this->pause();

		mReset = true;
		mFinished = true;

		mTotalFrame = num;

		mActiveNode = nullptr;

		auto scn = SceneGraphFactory::instance()->active();
		scn->setFrameNumber(0);
	}

	void PSimulationThread::resetNode(std::shared_ptr<Node> node)
	{
		mReset = true;

		mActiveNode = node;
	}

	void PSimulationThread::startUpdatingGraphicsContext()
	{
		mUpdatingGraphicsContext = true;
	}

	void PSimulationThread::stopUpdatingGraphicsContext()
	{
		mUpdatingGraphicsContext = false;
	}

	void PSimulationThread::setTotalFrames(int num)
	{
		mTotalFrame = num;
	}

	int PSimulationThread::getCurrentFrameNum() 
	{
		auto scn = SceneGraphFactory::instance()->active();
		return scn->getFrameNumber();
	}
}
