/*
BSD 3 - Clause License

Copyright(c) 2019, Electronic Arts
All rights reserved.
*/

//Modified implementation is based on C++ EASTL.
 
#ifndef HEAP_H
#define HEAP_H
#include "functional_base.h"
#include "Platform.h"
#include "type_traits.h"

namespace dyno
{

	/*The publicly usable functions we define are :
	//    push_heap     -- Adds an entry to a heap.                             
	//    pop_heap      -- Removes the top entry from a heap.                  
	//    make_heap     -- Converts an array to a heap.                         
	//    sort_heap     -- Sorts a heap in place (return reverse order of cmp, i.e. greater: descend, less: ascend).                               
	//    remove_heap   -- Removes an arbitrary entry from a heap.
	//    change_heap   -- Changes the priority of an entry in the heap.
	//    is_heap       -- Returns true if an array appears is in heap format.  
	//    is_heap_until -- Returns largest part of the range which is a heap.   
	// 
	// Please noted that, we do not implement heap for default cmp, so one have to add explicit cmp to heap, e.g. less<float> cmp.
	// See functional_base.h for those internal cmp. You are also welcomed to make own cmp inherited binary_function.
	*/

	/*============================================== promote heap ======================================================*/
	/// promote_heap_implementation
	template <typename RandomAccessIterator, typename Distance, typename T, typename Compare, typename ValueType>
	DYN_FUNC inline void promote_heap_impl(RandomAccessIterator first, Distance topPosition, Distance position, T value, Compare compare)
	{
		for (Distance parentPosition = (position - 1) >> 1; // This formula assumes that (position > 0). // We use '>> 1' instead of '/ 2' because we have seen VC++ generate better code with >>.
			(position > topPosition) && compare(*(first + parentPosition), value);
			parentPosition = (position - 1) >> 1)
		{
			*(first + position) = dyno::forward<ValueType>(*(first + parentPosition)); // Swap the node with its parent.
			position = parentPosition;
		}

		*(first + position) = dyno::forward<ValueType>(value);
	}

	/// promote_heap
	///
	/// Moves a value in the heap from a given position upward until it is sorted correctly. 
	/// This function requires that the value argument refer to a value that is currently not within the heap.
	/// Takes a Compare(a, b) function (or function object) which returns true if a < b.
	/// 
	template <typename RandomAccessIterator, typename Distance, typename T, typename Compare>
	DYN_FUNC inline void promote_heap(RandomAccessIterator first, Distance topPosition, Distance position, const T& value, Compare compare)
	{
		typedef typename iterator_traits<RandomAccessIterator>::value_type value_type;
		promote_heap_impl<RandomAccessIterator, Distance, const T&, Compare, const value_type>(first, topPosition, position, value, compare);
	}

	
	template <typename RandomAccessIterator, typename Distance, typename T, typename Compare>
	DYN_FUNC inline void promote_heap(RandomAccessIterator first, Distance topPosition, Distance position, T&& value, Compare compare)
	{
		typedef typename iterator_traits<RandomAccessIterator>::value_type value_type;
		promote_heap_impl<RandomAccessIterator, Distance, T&&, Compare, value_type>(first, topPosition, position, dyno::forward<T>(value), compare);
	}


	/*============================================== adjust heap ======================================================*/
	///adjust heap implementation
	template <typename RandomAccessIterator, typename Distance, typename T, typename Compare, typename ValueType>
	DYN_FUNC void adjust_heap_impl(RandomAccessIterator first, Distance topPosition, Distance heapSize, Distance position, T value, Compare compare)
	{
		// We do the conventional approach of moving the position down to the 
		// bottom then inserting the value at the back and moving it up.
		Distance childPosition = (2 * position) + 2;

		for (; childPosition < heapSize; childPosition = (2 * childPosition) + 2)
		{
			if (compare(*(first + childPosition), *(first + (childPosition - 1)))) // Choose the larger of the two children.
				--childPosition;
			*(first + position) = dyno::forward<ValueType>(*(first + childPosition)); // Swap positions with this child.
			position = childPosition;
		}

		if (childPosition == heapSize) // If we are at the bottom...
		{
			*(first + position) = dyno::forward<ValueType>(*(first + (childPosition - 1)));
			position = childPosition - 1;
		}

		dyno::promote_heap<RandomAccessIterator, Distance, T, Compare>(first, topPosition, position, dyno::forward<ValueType>(value), compare);
	}

	/// adjust_heap
	/// Given a position that has just been vacated, this function moves new values into that vacated position appropriately.
	/// The Compare function must work equivalently to the compare function used to make and maintain the heap.
	/// This function requires that the value argument refer to a value that is currently not within the heap.
	///
	template <typename RandomAccessIterator, typename Distance, typename T, typename Compare>
	DYN_FUNC void adjust_heap(RandomAccessIterator first, Distance topPosition, Distance heapSize, Distance position, const T& value, Compare compare)
	{
		typedef typename iterator_traits<RandomAccessIterator>::value_type value_type;
		dyno::adjust_heap_impl<RandomAccessIterator, Distance, const T&, Compare, const value_type>(first, topPosition, heapSize, position,dyno::forward<const T&>(value), compare);
	}

	template <typename RandomAccessIterator, typename Distance, typename T, typename Compare>
	DYN_FUNC void adjust_heap(RandomAccessIterator first, Distance topPosition, Distance heapSize, Distance position, T&& value, Compare compare)
	{
		typedef typename iterator_traits<RandomAccessIterator>::value_type value_type;
		dyno::adjust_heap_impl<RandomAccessIterator, Distance, T&&, Compare, value_type>(first, topPosition, heapSize, position, dyno::forward<T>(value), compare);
	}

	/*================================================ push heap ======================================================*/
	/// push_heap
	/// Adds an item to a heap(which is an array).The item necessarily comes from the back of the heap (array). 
	/// Thus, the insertion of a new item in a heap is a two step process: push_back and push_heap.
	/// 
	template <typename RandomAccessIterator, typename Compare>
	DYN_FUNC inline void push_heap(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
	{
		typedef typename dyno::iterator_traits<RandomAccessIterator>::difference_type difference_type;
		typedef typename dyno::iterator_traits<RandomAccessIterator>::value_type      value_type;

		const value_type tempBottom(*(last - 1));

		dyno::promote_heap<RandomAccessIterator, difference_type, value_type, Compare>
			(first, (difference_type)0, (difference_type)(last - first - 1), tempBottom, compare);
	}

	/*================================================= pop heap ======================================================*/
	/// pop_heap
	/// Removes the first item from the heap(which is an array), and adjusts the heap so that the highest priority item becomes the new first item.
	/// The logically removed element is actually in the back, so an extra pop_back is need.
	/// Thus, the pop of a prior-priority high item in a heap is a two step process: pop_heap and pop_back.
	/// 
	template <typename RandomAccessIterator, typename Compare>
	DYN_FUNC inline void pop_heap(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
	{
		typedef typename dyno::iterator_traits<RandomAccessIterator>::difference_type difference_type;
		typedef typename dyno::iterator_traits<RandomAccessIterator>::value_type      value_type;

		value_type tempBottom(dyno::forward<value_type>(*(last - 1)));
		*(last - 1) = dyno::forward<value_type>(*first);
		dyno::adjust_heap<RandomAccessIterator, difference_type, value_type, Compare>
			(first, (difference_type)0, (difference_type)(last - first - 1), 0, dyno::forward<value_type>(tempBottom), compare);
	}


	/*================================================ make heap ======================================================*/
	/// make_heap
	///
	/// Given an array, this function converts it into heap format. The complexity is O(n), where n is count of the range.
	/// The input range is not required to be in any order.
	///

	template <typename RandomAccessIterator, typename Compare>
	DYN_FUNC void make_heap(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
	{
		typedef typename dyno::iterator_traits<RandomAccessIterator>::difference_type difference_type;
		typedef typename dyno::iterator_traits<RandomAccessIterator>::value_type      value_type;

		const difference_type heapSize = last - first;

		if (heapSize >= 2) // If there is anything to do... (we need this check because otherwise the math fails below).
		{
			difference_type parentPosition = ((heapSize - 2) >> 1) + 1; 

			do {
				--parentPosition;
				value_type temp(dyno::forward<value_type>(*(first + parentPosition)));
				dyno::adjust_heap<RandomAccessIterator, difference_type, value_type, Compare>
					(first, parentPosition, heapSize, parentPosition,  dyno::forward<value_type>(temp), compare);
			} while (parentPosition != 0);
		}
	}

	/*================================================ sort heap ====================================================*/
	/// sort_heap
	///
	/// After the application if this algorithm, the range it was applied to is no longer a heap, though it will be a reverse heap. 
	/// The item with the lowest priority will be first, and the highest last.
	/// This is not a stable sort because the relative order of equivalent elements is not necessarily preserved.
	/// The range referenced must be valid; all pointers must be dereferenceable and within the sequence the last position is reachable from the first by incrementation.
	/// The complexity is at most O(n * log(n)), where n is count of the range.
	///
	template <typename RandomAccessIterator, typename Compare>
	DYN_FUNC inline void sort_heap(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
	{
		for (; (last - first) > 1; --last) // We simply use the heap to sort itself.
			dyno::pop_heap<RandomAccessIterator, Compare>(first, last, compare);
	}

	/*================================================ remove heap ====================================================*/
	/// remove_heap
	/// Removes an arbitrary entry from the heapand adjusts the heap appropriately.
	/// This function is unlike pop_heap in that pop_heap moves the top item to the back of the heap, whereas remove_heap moves an arbitrary item to the back of the heap.
	/// Thus, the removing of a new item in a heap is a two step process: remove_heap and pop_back.
	/// 
	
	template <typename RandomAccessIterator, typename Distance, typename Compare>
	DYN_FUNC inline void remove_heap(RandomAccessIterator first, Distance heapSize, Distance position, Compare compare)
	{
		typedef typename dyno::iterator_traits<RandomAccessIterator>::difference_type difference_type;
		typedef typename dyno::iterator_traits<RandomAccessIterator>::value_type      value_type;

		const value_type tempBottom(*(first + heapSize - 1));
		*(first + heapSize - 1) = *(first + position);
		dyno::adjust_heap<RandomAccessIterator, difference_type, value_type, Compare>
			(first, (difference_type)0, (difference_type)(heapSize - 1), (difference_type)position, tempBottom, compare);
	}

	/*================================================ change heap ====================================================*/
	/// change_heap
	/// Given a value in the heap that has changed in priority, this function adjusts the heap appropriately. 
	/// The heap size remains unchanged after this operation. 
	/// 
	template <typename RandomAccessIterator, typename Distance, typename Compare>
	DYN_FUNC inline void change_heap(RandomAccessIterator first, Distance heapSize, Distance position, Compare compare)
	{
		typedef typename dyno::iterator_traits<RandomAccessIterator>::difference_type difference_type;
		typedef typename dyno::iterator_traits<RandomAccessIterator>::value_type      value_type;

		dyno::remove_heap<RandomAccessIterator, Distance, Compare>(first, heapSize, position, compare);

		value_type tempBottom(*(first + heapSize - 1));

		dyno::promote_heap<RandomAccessIterator, difference_type, value_type, Compare>
			(first, (difference_type)0, (difference_type)(heapSize - 1), tempBottom, compare);
	}

	/*================================================ is_heap_until ==================================================*/
	/// is_heap_until
	template <typename RandomAccessIterator, typename Compare>
	DYN_FUNC inline RandomAccessIterator is_heap_until(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
	{
		int counter = 0;

		for (RandomAccessIterator child = first + 1; child < last; ++child, counter ^= 1)
		{
			if (compare(*first, *child))
				return child;
			first += counter; // counter switches between 0 and 1 every time through.
		}

		return last;
	}

	/*================================================== is_heap ======================================================*/
	///is_heap
	/// This is a useful algorithm for verifying that a random access container is in heap format. 
	template <typename RandomAccessIterator>
	DYN_FUNC inline bool is_heap(RandomAccessIterator first, RandomAccessIterator last)
	{
		return (dyno::is_heap_until(first, last) == last);
	}

	
}

#endif // HEAP_H
