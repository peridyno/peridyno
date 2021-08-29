#include "NodeIterator.h"
#include "Node.h"

namespace dyno
{

NodeIterator::NodeIterator()
{
	node_current = nullptr;
}


NodeIterator::NodeIterator(std::shared_ptr<Node> node)
{
	node_current = node;

	if (node_current != nullptr)
	{
		auto children = node_current->getAncestors();
		for each (auto c in children)
		{
			if (c->isControllable())
			{
				node_stack.push(c);
			}
		}
	}
}


NodeIterator::~NodeIterator()
{

}

NodeIterator& NodeIterator::operator++()
{
	if (node_stack.empty())
		node_current = nullptr;
	else
	{
		node_current = node_stack.top();
		node_stack.pop();

		auto children = node_current->getAncestors();
		for each (auto c in children)
		{
			if (c->isActive())
			{
				node_stack.push(c);
			}
		}
	}

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

bool NodeIterator::operator!=(const NodeIterator &iterator) const
{
	return node_current != iterator.get();
}

bool NodeIterator::operator==(const NodeIterator &iterator) const
{
	return node_current == iterator.get();
}


}