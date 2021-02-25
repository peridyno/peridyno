#include "PSimulationThread.h"

#include "Framework/SceneGraph.h"

namespace dyno
{
	QWaitCondition m_wait_condition;

	PSimulationThread::PSimulationThread()
		: max_frames(1000)
	{

	}

	PSimulationThread* PSimulationThread::instance()
	{
		static PSimulationThread m_instance;
		return &m_instance;
	}

	void PSimulationThread::pause()
	{
		//this->m_mutex.lock();
		m_paused = true;
	}

	void PSimulationThread::resume()
	{
		//this->m_mutex.unlock();
		m_paused = false;
	}

	void PSimulationThread::stop()
	{
		this->exit();
	}

	void PSimulationThread::run()
	{
		SceneGraph::getInstance().initialize();

		int f = 0;
		while(true && f < max_frames)
		{
			if (!m_rendering && !m_paused)
			{
//				m_mutex.lock();
				SceneGraph::getInstance().takeOneFrame();

				f++;

				this->startRendering();
				emit(oneFrameFinished());

//				m_mutex.unlock();
			}
//			
		}

		this->stop();
	}

	void PSimulationThread::reset()
	{
		SceneGraph::getInstance().reset();
	}

	void PSimulationThread::startRendering()
	{
		m_rendering = true;
	}

	void PSimulationThread::stopRendering()
	{
		m_rendering = false;
	}

	void PSimulationThread::setTotalFrames(int num)
	{
		max_frames = num;
	}

}
