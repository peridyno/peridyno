#ifndef PSIMULATIONTHREAD_H
#define PSIMULATIONTHREAD_H

#include <QThread>
#include <QMutex>
#include <QWaitCondition>

namespace dyno
{
	class PSimulationThread : public QThread
	{
		Q_OBJECT

	public:
		static PSimulationThread* instance();
		

		void pause();
		void resume();
		void stop();

		void run() override;

		void reset();

		void startRendering();
		void stopRendering();

		void setTotalFrames(int num);

		int getCurrentFrameNum();

	Q_SIGNALS:
		//Note: should not be emitted from the user

		void oneFrameFinished();

	private:
		PSimulationThread();

		int mFrameNum;
		int currentFrameNum;

		bool mReset = false;
		bool mPaused = true;
		bool mRendering = false;
		bool mRunning = true;

		QMutex mMutex;
	};
}


#endif // PSIMULATIONTHREAD_H
