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

#ifndef RANDOM_ACCESS_CONTAINER_H
#define RANDOM_ACCESS_CONTAINER_H

#include "STLBuffer.h"
#include "functional_base.h"
#include "type_traits.h"
#include "Platform.h"

/// <summary>
/// This is a implementation of a std::vector liked Random Access Container, however, dynamic allocating is not provided.
/// User have to reserve a buffer strickly offer buffer_size.
/// If needed, user may use copy construct or use a pair of iterator and a buffer_size (default = iterator slice size) to reserve by an exist container or rescourse.
/// However, the second rountine is quite dangerous, because neight the implementation nor complier will check whether the memory is correctly construct, 
/// i.e. with proper alignment and struct when none-trivial type (i.e. ADT) is used.
/// All memory visiting behavior and insertion manipulation is asserting protected to aviod protential illegal memory access.  
/// 
///  - assign()			make assignment
///  - begin()			get begin iterator
///  - end()			get end (non-used) iterator
///  - empty()			check empty
///  - size()			get number of previous elements
///  - capacity()		get maximium capacity
///  - resize()			resize number of exist elements
///  - reserve()		which is required and should be called only once, (unless to recommit)
///  - data()			get pointer of memory
///  - operator[]		random access rountine, logically should not use assertion for range-out check, but used in thie version
///  - at()				random access rountine (implement exactly the same as operator[] in this version)
///  - front()			get front element reference, which is placed in begin()
///  - back()			get back reference, which is placed in end()-1
///  - push_back()		push back element
///  - pop_back()		pop back tail's element
///  - find()			find the first element with value, using == or predicate function, value_type is required to be comparable
///  - insert()			set of insertion rountine
///  - erase()			set of erase rountine
///  - clear()			clear container, the associated memory will be deconstructed (i.e. ~value_type() is called explicitly), reserved buffer will not be influenced.
/// 
/// </summary>

namespace dyno {
	template <typename T>
	class RandomAccessContainer : public STLBuffer<T>
	{
		typedef STLBuffer<T>						base_type;
		typedef RandomAccessContainer<T>			this_type;

	public:
		typedef T									value_type;
		typedef T*									pointer;
		typedef const T*							const_pointer;
		typedef T&									reference;
		typedef const T&							const_reference;  
		typedef T*									iterator;        
		typedef const T*							const_iterator;   
		typedef long long							difference_type;
		typedef uint								size_type;

	public:
		DYN_FUNC RandomAccessContainer() = default;

		DYN_FUNC RandomAccessContainer(const this_type& c) :
			m_size(c.m_size),
			m_Begin(c.m_Begin),
			m_End(c.m_End),
			m_bufferEnd(c.m_bufferEnd) {
			this->reserve(c.m_startLoc, c.m_maxSize);
		};

		DYN_FUNC ~RandomAccessContainer() {
			m_startLoc = nullptr;
			m_Begin = nullptr;
			m_End = nullptr;
			m_bufferEnd = nullptr;
		};

		DYN_FUNC void assign(size_type n, const value_type& value);

		template <typename InputIterator>
		DYN_FUNC void assign(InputIterator first, InputIterator last);

		DYN_FUNC inline iterator       begin() noexcept { return m_Begin; }
		DYN_FUNC inline const_iterator begin() const noexcept { return m_Begin; }

		DYN_FUNC inline iterator       end() noexcept {return m_End; }
		DYN_FUNC inline const_iterator end() const noexcept { return m_End; }

		DYN_FUNC inline bool      empty() const noexcept {return m_startLoc == nullptr;} //different from std's
		DYN_FUNC inline size_type size() const noexcept { return m_size; }
		DYN_FUNC inline size_type capacity() const noexcept { return m_maxSize; }

		DYN_FUNC void resize(size_type n);

		DYN_FUNC inline void reserve(iterator beg, size_type buffer_size); //only and must call one time.
		DYN_FUNC inline void reserve(iterator beg, iterator end, size_type buffer_size = 0);
	
		DYN_FUNC inline pointer       data() noexcept { return m_startLoc; }
		DYN_FUNC inline const_pointer data() const noexcept {return m_startLoc; }

		DYN_FUNC inline reference       operator[](size_type n);
		DYN_FUNC inline const_reference operator[](size_type n) const;

		DYN_FUNC inline reference       at(size_type n);
		DYN_FUNC inline const_reference at(size_type n) const;

		DYN_FUNC inline reference       front();
		DYN_FUNC inline const_reference front() const;

		DYN_FUNC inline reference       back();
		DYN_FUNC inline const_reference back() const;

		DYN_FUNC void      push_back(const value_type& value);
		DYN_FUNC void	   push_back(value_type&& value);
		DYN_FUNC reference push_back();
		DYN_FUNC void      pop_back();

		DYN_FUNC inline iterator  find(iterator first, iterator last, const value_type& value); //O(n)

		template<typename Predicate = dyno::predicate<value_type>>
		DYN_FUNC inline iterator  find(iterator first, iterator last, const value_type& value, Predicate pre); //O(n)

		DYN_FUNC iterator insert(const_iterator position, const value_type& value);
		DYN_FUNC iterator insert(const_iterator position, size_type n, const value_type& value);
		DYN_FUNC iterator insert(const_iterator position, value_type&& value);

		template <typename InputIterator>
		DYN_FUNC iterator insert(const_iterator position, InputIterator first, InputIterator last);

		DYN_FUNC iterator erase_first(const T& value);
		DYN_FUNC iterator erase_first_unsorted(const T& value); // Same as erase, except it doesn't preserve order, but is faster because it simply copies the last item in the RandomAccessContainer over the erased position.
		
		DYN_FUNC iterator erase(const_iterator position);
		DYN_FUNC iterator erase(const_iterator first, const_iterator last);
		DYN_FUNC iterator erase_unsorted(const_iterator position);         // Same as erase, except it doesn't preserve order, but is faster because it simply copies the last item in the RandomAccessContainer over the erased position.

		DYN_FUNC void clear() noexcept; //clear will not change m_Begin and capacity, only change size and m_End

		private:
		size_type	m_size = 0;
		iterator	m_bufferEnd = nullptr;
		iterator	m_End = nullptr;
		iterator	m_Begin = nullptr;		//Logically, m_Begin is the iterator of the beginner of element, m_startLoc is the pointer to the container, though they will be equal in this implementation.

	}; // class RandomAccessContainer

}

#include "RandomAccessContainer.inl"
#endif // RANDOM_ACCESS_CONTAINER_H