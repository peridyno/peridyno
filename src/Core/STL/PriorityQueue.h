/**
 * Copyright 2023 Zixuan Lu
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PRIORITY_QUEUE_H
#define PRIORITY_QUEUE_H

#include "Heap.h"
#include "functional_base.h"
#include "RandomAccessContainer.h"

namespace dyno
{

	/// priority_queue
	///
	/// The behaviour of this class is just like the std::priority_queue
	///  - empty()		check is the priority_queue is empty
	///  - size()		return the size
	///  - top()		get the top of the priority_queue
	///  - push()		push the element and adjust the priority_queue with cmp
	///  - pop()		pop the first priority element and adjust the priority_queue
	///  - change()		moves the item at the given array index to a new location based on its current priority.
	///  - remove()		removes the item at the given array index
	/// The container should using randomAccess Container which have such routine: 
	/// empty(), size(), clear(), insert(), front(), push_back(), pop_back()

	template <typename T, typename Container = dyno::RandomAccessContainer<T>, typename Compare = dyno::less<typename Container::value_type> >
	class priority_queue
	{
	public:
		typedef priority_queue<T, Container, Compare>        this_type;
		typedef Container                                    container_type;
		typedef Compare                                      compare_type;

		typedef typename Container::value_type               value_type;
		typedef typename Container::reference                reference;
		typedef typename Container::const_reference          const_reference;
		typedef typename Container::size_type                size_type;
		typedef typename Container::difference_type          difference_type;

	public:
		container_type  container;
	private:
		compare_type    cmp;

	public:
		DYN_FUNC priority_queue();
		DYN_FUNC ~priority_queue();

		DYN_FUNC explicit priority_queue(const container_type& c);

		DYN_FUNC explicit priority_queue(const compare_type& compare);

		DYN_FUNC explicit priority_queue(const container_type& c, const compare_type& compare);

		DYN_FUNC inline bool empty() const;

		DYN_FUNC inline bool isEmpty() const;

		DYN_FUNC size_type size() const;

		DYN_FUNC const_reference top() const;

		DYN_FUNC void push(const value_type& value);

		DYN_FUNC void pop();

		DYN_FUNC void clear();

		DYN_FUNC void change(size_type n);

		DYN_FUNC void remove(size_type n);

	}; //priority_queue


}
#include "PriorityQueue.inl"
#endif