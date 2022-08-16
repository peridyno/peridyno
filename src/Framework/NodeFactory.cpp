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

	void NodeGroup::addAction(std::shared_ptr<NodeAction> nAct)
	{
		mActions.push_back(nAct);
	}

	bool NodePage::hasGroup(std::string name)
	{
		return mGroups.find(name) != mGroups.end();
	}

	std::shared_ptr<dyno::NodeGroup> NodePage::addGroup(std::string name)
	{
		if (mGroups.find(name) == mGroups.end())
		{
			mGroups[name] = std::make_shared<NodeGroup>(name);
		}

		return mGroups[name];
	}

	bool NodeFactory::hasPage(std::string name)
	{
		return mPages.find(name) != mPages.end();
	}

	std::shared_ptr<NodePage> NodeFactory::addPage(std::string name, std::string icon)
	{
		if (mPages.find(name) == mPages.end())
		{
			mPages[name] = std::make_shared<NodePage>(name, icon);
		}

		return mPages[name];
	}
}