#ifndef LIST_ITERATOR_H
#define LIST_ITERATOR_H

#include "Platform.h"
#include "ListNode.h"

namespace dyno
{

	template <typename T>
	class List_Iterator
	{
	public:
		
			typedef ListNode<T> Node;
			typedef List_Iterator<T> Self;
			Node* _node;

			DYN_FUNC List_Iterator(Node* node)
				:_node(node) {}

			DYN_FUNC inline T& operator*()
			{
				return _node->_data;
			}

			DYN_FUNC inline T* operator->()
			{
				return &_node->_next;
			}

			DYN_FUNC inline Self& operator++()
			{
				_node = _node->_next;
				return *this;
			}

			DYN_FUNC inline Self operator++(int)
			{
				Self tmp(*this);

				++(*this);
				return tmp;
			}

			DYN_FUNC inline Self& operator--()
			{
				_node = _node->_prev;
				return *this;
			}

			DYN_FUNC inline Self operator--(int)
			{
				Self tmp(*this);

				--(*this);
				return tmp;
			}

			DYN_FUNC inline bool operator==(const Self& it)
			{
				return _node == it._node;
			}

			DYN_FUNC inline bool operator!=(const Self& it)
			{
				return _node != it._node;
			}

	};

}
#endif // LIST_ITERATOR_H
