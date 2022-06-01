#include "NodeFactory.h"

namespace dyno
{
	std::atomic<NodeFactory*> NodeFactory::pInstance;
	std::mutex NodeFactory::mMutex;

	//Thread-safe singleton mode
	NodeFactory* NodeFactory::instance()
	{
		NodeFactory* ins = pInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(mMutex);
			ins = pInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new NodeFactory();
				pInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	void NodeGroup::addAction(std::string caption, std::string icon, std::function<std::shared_ptr<Node>()> act)
	{
		mActions.push_back(std::make_shared<NodeAction>(caption, icon, act));
	}

	std::shared_ptr<dyno::NodeGroup> NodeFactory::addGroup(std::string groupName, std::string caption, std::string icon)
	{
		if (mGroups.find(groupName) == mGroups.end())
		{
			mGroups[groupName] = std::make_shared<NodeGroup>(caption, icon);
		}

		return mGroups[groupName];
	}
}