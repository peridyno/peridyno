#include "SceneGraphFactory.h"

namespace dyno
{
	std::atomic<SceneGraphFactory*> SceneGraphFactory::pInstance;
	std::mutex SceneGraphFactory::mMutex;

	//Thread-safe singleton mode
	SceneGraphFactory* SceneGraphFactory::instance()
	{
		SceneGraphFactory* ins = pInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(mMutex);
			ins = pInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new SceneGraphFactory();
				pInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	std::shared_ptr<SceneGraph> SceneGraphFactory::active()
	{
		//If no SceneGraph is created, return an empty one.
		if (mSceneGraphs.empty())
			this->createNewScene();

		return mSceneGraphs.top();
	}

	std::shared_ptr<SceneGraph> SceneGraphFactory::createNewScene()
	{
		mSceneGraphs.push(std::make_shared<SceneGraph>());

		return mSceneGraphs.top();
	}

	void SceneGraphFactory::pushScene(std::shared_ptr<SceneGraph> scn)
	{
		mSceneGraphs.push(scn);
	}

	void SceneGraphFactory::popScene()
	{
		mSceneGraphs.pop();
	}

	void SceneGraphFactory::popAllScenes()
	{
		while (!mSceneGraphs.empty())
		{
			mSceneGraphs.pop();
		}
	}

}