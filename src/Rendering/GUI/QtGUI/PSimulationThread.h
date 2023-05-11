#ifndef PSIMULATIONTHREAD_H
#define PSIMULATIONTHREAD_H

#include <QThread>
#include <QMutex>
#include <QWaitCondition>

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

		void pause();
		void resume();
		void stop();

		void createNewScene();
		void createNewScene(std::shared_ptr<SceneGraph> scn);

		void closeCurrentScene();

		void closeAllScenes();

		void run() override;

		void reset(int num);

		void startUpdatingGraphicsContext();
		void stopUpdatingGraphicsContext();

		void setTotalFrames(int num);

		int getCurrentFrameNum();
	Q_SIGNALS:
		//Note: should not be emitted from the user

		void sceneGraphChanged();

		void oneFrameFinished();
		void simulationFinished();

	public slots:
		void resetNode(std::shared_ptr<Node> node);
		void resetQtNode(Qt::QtNode& node);

	private:
		PSimulationThread();

		int mTotalFrame;

		bool mReset = false;
		bool mPaused = true;
		
		bool mRunning = true;
		bool mFinished = false;

		bool mUpdatingGraphicsContext = false;

		std::shared_ptr<Node> mActiveNode = nullptr;

		int mTimeOut = 10000;

		QMutex mMutex;
	};
}


#endif // PSIMULATIONTHREAD_H
