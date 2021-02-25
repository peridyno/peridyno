#ifndef LISTNODE_H
#define LISTNODE_H

#include "Platform.h"

namespace dyno
{

	template <typename T>
	class ListNode
	{
	public:
		ListNode<T>* _next;
		ListNode<T>* _prev;
		T _data;

		DYN_FUNC ListNode(const T& data = T())
			:_data(data), _next(nullptr), _prev(nullptr) {}
	};

}
#endif // LISTNODE_H