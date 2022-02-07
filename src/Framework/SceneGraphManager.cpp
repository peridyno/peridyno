#include "SceneGraphManager.h"

namespace dyno
{

	SceneGraphManager::SceneGraphManager()
	{

	}

	SceneGraphManager& SceneGraphManager::instance()
	{
		static SceneGraphManager m_instance;
		return m_instance;
	}

	std::shared_ptr<SceneGraph> SceneGraphManager::active()
	{
		return mSceneGraphs.size() == 0 ? nullptr : mSceneGraphs[mActiveId];
	}

}