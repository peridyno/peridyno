#include "PSimulationThread.h"

#include <memory>
#include <mutex>
#include <utility>

#include "Node.h"
#include "SceneGraph.h"
#include "SceneGraphFactory.h"

#include "Action.h"

#include "NodeEditor/QtNodeWidget.h"

using namespace std::chrono_literals;
namespace dyno
{
	QWaitCondition m_wait_condition;

	PSimulationThread::PSimulationThread()
		: 
		mTotalFrame(1000),
		mReset(false),
		mPaused(true),
		mRunning(true),
		mFinished(false),
		mUpdatingGraphicsContext(false),
		mTimeOut(10000)
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
		notify();
	}

	void PSimulationThread::stop()
	{
		mRunning = false;
		notify();
	}

	void PSimulationThread::createNewScene()
	{
		if (mMutex.try_lock_for(mTimeOut))
		{
			SceneGraphFactory::instance()->createNewScene();
			this->reset(0);

			mMutex.unlock();

			emit sceneGraphChanged();
		}
		else
		{
			Log::sendMessage(Log::Info, "Time out");
		}
	}


	void PSimulationThread::createNewScene(std::shared_ptr<SceneGraph> scn)
	{
		if (mMutex.try_lock_for(mTimeOut))
		{
			SceneGraphFactory::instance()->pushScene(scn);
			scn->reset();
			this->reset(0);

			mMutex.unlock();

			emit sceneGraphChanged();
		}
		else
		{
			Log::sendMessage(Log::Info, "Time out");
		}
	}

	void PSimulationThread::closeCurrentScene()
	{
		if (mMutex.try_lock_for(mTimeOut))
		{
			SceneGraphFactory::instance()->popScene();
			this->reset(0);

			mMutex.unlock();

			emit sceneGraphChanged();
		}
		else
		{
			Log::sendMessage(Log::Info, "Time out");
		}
	}

	void PSimulationThread::closeAllScenes()
	{
		if (mMutex.try_lock_for(mTimeOut))
		{
			SceneGraphFactory::instance()->popAllScenes();
			this->reset(0);

			mMutex.unlock();

			emit sceneGraphChanged();
		}
		else
		{
			Log::sendMessage(Log::Info, "Time out");
		}
	}

	void PSimulationThread::notify() {
		mCond.notify_one();
	}

	void PSimulationThread::run()
	{
		for(;;)
		{
			std::unique_lock<decltype(mMutex)> lock(mMutex);
			auto scn = SceneGraphFactory::instance()->active();

			bool has_frame = scn->getFrameNumber() < mTotalFrame;

			while(mRunning && (mUpdatingGraphicsContext || (!mReset && mPaused))) {
				mCond.wait_for(lock, 500ms);
			}

			if(!mRunning) break;

			if (mReset)
			{
				if (mActiveNode == nullptr) {
					scn->reset(); 

					mFinished = false;
				}
				else {
					scn->reset(mActiveNode);

					mActiveNode = nullptr;
				}

				mReset = false;

				emit oneFrameFinished();
			}

			if(!mPaused) {
				if (scn->getFrameNumber() < mTotalFrame)
				{
					scn->takeOneFrame();

					this->startUpdatingGraphicsContext();

					emit oneFrameFinished();
				}

				if (scn->getFrameNumber() >= mTotalFrame)
				{
					mPaused = true;
					if(!std::exchange(mFinished,true)) {
						emit simulationFinished();
					}
				}
			}
		}
	}

	void PSimulationThread::reset(int num)
	{
		this->pause();

		mFinished = true;

		mTotalFrame = num;

		mActiveNode = nullptr;

		auto scn = SceneGraphFactory::instance()->active();
		scn->setFrameNumber(0);

		//Note: should set mReset at the end
		mReset = true;
		notify();
	}

	void PSimulationThread::proceed(int num)
	{
		this->pause();

		mFinished = false;

		mTotalFrame = num;

		notify();
	}

	void PSimulationThread::resetNode(std::shared_ptr<Node> node)
	{
		auto scn = SceneGraphFactory::instance()->active();
		scn->reset(node);

		notify();
	}

	void PSimulationThread::resetQtNode(Qt::QtNode& node)
	{
		auto model = node.nodeDataModel();
		auto widget = dynamic_cast<Qt::QtNodeWidget*>(model);

		if (widget != nullptr)
		{
			mActiveNode = widget->getNode();
			mReset = true;
		}
	}

	void PSimulationThread::syncNode(std::shared_ptr<Node> node)
	{
		auto scn = SceneGraphFactory::instance()->active();

		class SyncNodeAct : public Action
		{
		public:
			void process(Node* node) override {
				node->reset();
				node->updateGraphicsContext();
			}
		};

		scn->traverseForwardWithAutoSync<SyncNodeAct>(node);

		notify();
	}

	void PSimulationThread::startUpdatingGraphicsContext()
	{
		mUpdatingGraphicsContext = true;
	}

	void PSimulationThread::stopUpdatingGraphicsContext()
	{
		mUpdatingGraphicsContext = false;
		notify();
	}

	void PSimulationThread::setTotalFrames(int num)
	{
		mTotalFrame = num;
		notify();
	}

	int PSimulationThread::getCurrentFrameNum() 
	{
		auto scn = SceneGraphFactory::instance()->active();
		return scn->getFrameNumber();
	}
}
