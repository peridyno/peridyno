#include "SceneGraph.h"
#include "Action/ActAnimate.h"
#include "Action/ActDraw.h"
#include "Action/ActInit.h"
#include "Action/ActReset.h"
#include "Action/ActQueryTimestep.h"
#include "Action/ActPostProcessing.h"
#include "Framework/SceneLoaderFactory.h"


namespace dyno
{
SceneGraph& SceneGraph::getInstance()
{
	static SceneGraph m_instance;
	return m_instance;
}

bool SceneGraph::isIntervalAdaptive()
{
	return m_advative_interval;
}

void SceneGraph::setAdaptiveInterval(bool adaptive)
{
	m_advative_interval = adaptive;
}

void SceneGraph::setGravity(Vector3f g)
{
	m_gravity = g;
}

Vector3f SceneGraph::getGravity()
{
	return m_gravity;
}

bool SceneGraph::initialize()
{
	if (m_initialized)
	{
		return true;
	}
	//TODO: check initialization
	if (m_root == nullptr)
	{
		return false;
	}

	m_root->traverseBottomUp<InitAct>();
	m_initialized = true;

	return m_initialized;
}

void SceneGraph::invalid()
{
	m_initialized = false;
}

void SceneGraph::draw()
{
	if (m_root == nullptr)
	{
		return;
	}

	m_root->traverseTopDown<DrawAct>();
}

void SceneGraph::advance(float dt)
{
//	AnimationController*  aController = m_root->getAnimationController();
	//	aController->
}

void SceneGraph::takeOneFrame()
{
	/*
	if (m_root == nullptr)
	{
		return;
	}
	m_root->traverseTopDown<AnimateAct>();*/
	std::cout << "****************Frame " << m_frameNumber << " Started" << std::endl;

	if (m_root == nullptr)
	{
		return;
	}

	

	float t = 0.0f;
	float dt = 0.0f;

	QueryTimeStep time;

	time.reset();
	m_root->traverseTopDown(&time);
	dt = time.getTimeStep();

	if (m_advative_interval)
	{
		m_root->traverseTopDown<AnimateAct>(dt);
		m_elapsedTime += dt;
	}
	else
	{
		float interval = 1.0f / m_frameRate;
		while (t + dt < interval)
		{
			m_root->traverseTopDown<AnimateAct>(dt);

			t += dt;
			time.reset();
			m_root->traverseTopDown(&time);
			dt = time.getTimeStep();
		}

		m_root->traverseTopDown<AnimateAct>(interval - t);

		m_elapsedTime += interval;
	}
	
	m_root->traverseTopDown<PostProcessing>();

	std::cout << "****************Frame " << m_frameNumber << " Ended" << std::endl << std::endl;

	m_frameNumber++;
}

void SceneGraph::run()
{

}

void SceneGraph::reset()
{
	if (m_root == nullptr)
	{
		return;
	}

	m_root->traverseBottomUp<ResetAct>();

	//m_root->traverseBottomUp();
}

bool SceneGraph::load(std::string name)
{
	SceneLoader* loader = SceneLoaderFactory::getInstance().getEntryByFileName(name);
	if (loader)
	{
		m_root = loader->load(name);
		return true;
	}

	return false;
}

Vector3f SceneGraph::getLowerBound()
{
	return m_lowerBound;
}

Vector3f SceneGraph::getUpperBound()
{
	return m_upperBound;
}

void SceneGraph::setLowerBound(Vector3f lowerBound)
{
	m_lowerBound = lowerBound;
}

void SceneGraph::setUpperBound(Vector3f upperBound)
{
	m_upperBound = upperBound;
}


}