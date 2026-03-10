#ifndef PSIMULATIONTHREAD_H
#define PSIMULATIONTHREAD_H

#include <QThread>
#include <QMutex>
#include <QTimer>
#include <QWaitCondition>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>

#include <nodes/QNode>

namespace dyno
{
	class Node;
	class SceneGraph;

	class PSimulationThread : public QThread
	{
		Q_OBJECT

	public:
		static PSimulationThread* instance();

		//When multithreading true, the simulation will be run in separate thread, other it runs on the same thread with GUI
		void launch(bool multithreading);
		void abort();

		void switchThreadingMode(bool multithreading);

		bool isMultithreading() { return mMultithreading; }

		void pause();
		void resume();
		void stop();

		void createNewScene();
		void createNewScene(std::shared_ptr<SceneGraph> scn);

		void closeCurrentScene();

		void closeAllScenes();

		std::shared_ptr<SceneGraph> getCurrentScene();

		/**
		 * @brief Reset the simulation
		 */
		void reset(int num);
		void reset();
		/**
		 * @brief Continue the simulation from the current frame
		 */
		void proceed(int num);

		void startUpdatingGraphicsContext();
		void stopUpdatingGraphicsContext();

		void setTotalFrames(int num);
		inline int getTotalFrames() { return mTotalFrame; }

		void takeOneStep();

		int getCurrentFrameNum();

		bool isPaused() {
			return mPaused;
		}

		bool isRunning() {
			return mRunning;
		}

	Q_SIGNALS:
		//Note: should not be emitted from the user

		void sceneGraphChanged();

		void oneFrameFinished(int frame);
		void simulationFinished();

	public slots:
		void resetNode(std::shared_ptr<Node> node);
		void resetQtNode(Qt::QtNode& node);

		void syncNode(std::shared_ptr<Node> node);

	private:
		void run() override;

		bool mainEventLoop();

	private:
		PSimulationThread();

		static std::atomic<PSimulationThread*> pInstance;
		static std::mutex mInstanceMutex;

		void notify();

		std::atomic<int> mTotalFrame;

		bool mReset;
		bool mFinished;
		bool mOneStep = false;

		bool mMultithreading = false;

		QTimer* mTimer = nullptr;

	 	std::atomic<bool> mPaused;
		std::atomic<bool> mRunning;
		
		std::atomic<bool> mUpdatingGraphicsContext;

		std::shared_ptr<Node> mActiveNode;

		std::chrono::milliseconds mTimeOut;
		std::timed_mutex mMutex;
		std::condition_variable_any mCond;
	};
}


#endif // PSIMULATIONTHREAD_H
