#ifndef PSIMULATIONTHREAD_H
#define PSIMULATIONTHREAD_H

#include <QThread>
#include <QMutex>
#include <QWaitCondition>

namespace dyno
{
	class Node;

	class PSimulationThread : public QThread
	{
		Q_OBJECT

	public:
		static PSimulationThread* instance();

		void pause();
		void resume();
		void stop();

		void run() override;

		void reset(int num);

		void startRendering();
		void stopRendering();

		void setTotalFrames(int num);

		int getCurrentFrameNum();
	Q_SIGNALS:
		//Note: should not be emitted from the user

		void oneFrameFinished();
		void simulationFinished();

	public slots:
		void resetNode(std::shared_ptr<Node> node);


		
	private:
		PSimulationThread();

		int mTotalFrame;

		bool mReset = false;
		bool mPaused = true;
		bool mRendering = false;
		bool mRunning = true;
		bool mFinished = false;

		std::shared_ptr<Node> mActiveNode = nullptr;

		QMutex mMutex;
	};
}


#endif // PSIMULATIONTHREAD_H
