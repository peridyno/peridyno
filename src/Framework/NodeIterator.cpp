#include "NodeIterator.h"
#include "Node.h"

namespace dyno
{

	NodeIterator::NodeIterator()
	{
		node_current = nullptr;
	}


	NodeIterator::NodeIterator(std::list<Node*>& nList, std::map<ObjectId, std::shared_ptr<Node>>& nMap)
	{
		mNodeList.clear();

		for (auto it = nList.begin(); it != nList.end(); ++it)
		{
			if (nMap.find((*it)->objectId()) != nMap.end()) {
				mNodeList.push_back(nMap[(*it)->objectId()]);
			}
		}

		node_current = mNodeList.empty() ? nullptr : mNodeList.front();

		if (!mNodeList.empty())
			mNodeList.pop_front();
	}


	NodeIterator::~NodeIterator()
	{

	}

	NodeIterator& NodeIterator::operator++()
	{
		node_current = mNodeList.empty() ? nullptr : mNodeList.front();

		if (!mNodeList.empty())
			mNodeList.pop_front();

		return *this;
	}


	NodeIterator& NodeIterator::operator++(int)
	{
		return operator++();
	}

	std::shared_ptr<Node> NodeIterator::operator->() const
	{
		return node_current;
	}

	std::shared_ptr<Node> NodeIterator::get() const
	{
		return node_current;
	}

	bool NodeIterator::operator!=(const NodeIterator& iterator) const
	{
		return node_current != iterator.get();
	}

	bool NodeIterator::operator==(const NodeIterator& iterator) const
	{
		return node_current == iterator.get();
	}
}