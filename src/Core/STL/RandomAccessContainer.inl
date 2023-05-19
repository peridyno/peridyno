#include "type_traits.h"
namespace dyno {

	template <typename T>
	template <typename InputIterator>
	DYN_FUNC void RandomAccessContainer<T>::assign(InputIterator first, InputIterator last) {

		assert((size_type)(last - first) <= m_maxSize);
		iterator position(m_startLoc);

		while ((position != m_End) && (first != last))
		{
			*position = *first;
			++first;
			++position;
		}
		if (first == last)
			erase(position, m_End);
		else
			insert(m_End, first, last);
	}

	template<typename T>
	DYN_FUNC void RandomAccessContainer<T>::assign(size_type n, const value_type& value) {
		assert(n <= m_maxSize);
		iterator position(m_startLoc);
		size_type index = 0;
		while ((position != m_End) && (index != n))
		{
			*position = value_type(value);
			++index;
			++position;
		}
		if (index == n)
			erase(position, m_End);
		else
			insert(m_End, n - index, value);
	}


	template<typename T>
	DYN_FUNC inline void RandomAccessContainer<T>::reserve(iterator beg, size_type buffer_size) {
		m_startLoc = beg;
		m_maxSize = buffer_size;
		m_bufferEnd = this->bufferEnd();
		m_End = beg;
		m_Begin = beg;
		m_size = 0;
	}

	template<typename T>
	DYN_FUNC inline void RandomAccessContainer<T>::reserve(iterator beg, iterator end, size_type buffer_size) {
		m_startLoc = beg;
		m_maxSize = buffer_size ? buffer_size: (size_type)(end - beg);
		m_bufferEnd = this->bufferEnd();
		m_End = end;
		m_Begin = beg;
		m_size = (size_type)(end - beg);
	}

	template<typename T>
	DYN_FUNC void RandomAccessContainer<T>::resize(size_type n) {
		assert(n <= m_maxSize);
		if (m_size == n)
			return;

		if (n > (size_type)(m_End - m_Begin)) {
			//push back default value_type
			while(n > (size_type)(m_End - m_Begin))
				push_back();
		}
		else {
			//destory 
			for (iterator beg = m_Begin + n; beg != m_End; ++beg)
				beg->~value_type();
		}
		m_End = m_Begin + n;
		m_size = n;
	}

	template<typename T>
	DYN_FUNC inline RandomAccessContainer<T>::reference
		RandomAccessContainer<T>::operator[](size_type n) {
		assert(n < m_size);
		return *(m_Begin + n);
	}

	template<typename T>
	DYN_FUNC inline RandomAccessContainer<T>::const_reference
		RandomAccessContainer<T>::operator[](size_type n) const {
		assert(n < m_size);
		return *(m_Begin + n);
	}

	template<typename T>
	DYN_FUNC inline RandomAccessContainer<T>::reference
		RandomAccessContainer<T>::at(size_type n) {
		assert(n < m_size); //although this should be changed as a throw.
		return *(m_Begin + n);
	}

	template<typename T>
	DYN_FUNC inline RandomAccessContainer<T>::const_reference
		RandomAccessContainer<T>::at(size_type n) const {
		assert(n < m_size); //although this should be changed as a throw.
		return *(m_Begin + n);
	}

	template<typename T>
	DYN_FUNC inline RandomAccessContainer<T>::reference
		RandomAccessContainer<T>::front() {
		assert(m_startLoc != nullptr && m_size > 0);
		// The second assertion is because if one get a empty container and assign to its front (though the reference is vaild),
		// the size will be change from 0->1, but front routine will not able to change m_size.
		return *(m_Begin);
	}

	template<typename T>
	DYN_FUNC inline RandomAccessContainer<T>::const_reference
		RandomAccessContainer<T>::front() const {
		assert(m_startLoc != nullptr && m_size > 0);
		return *(m_Begin);
	}

	template<typename T>
	DYN_FUNC inline RandomAccessContainer<T>::reference
		RandomAccessContainer<T>::back() {
		assert(m_startLoc != nullptr && m_size > 0);
		return *(m_End - 1);
	}

	template<typename T>
	DYN_FUNC inline RandomAccessContainer<T>::const_reference
		RandomAccessContainer<T>::back() const {
		assert(m_startLoc != nullptr && m_size > 0);
		return *(m_End - 1);
	}

	template<typename T>
	DYN_FUNC void RandomAccessContainer<T>::push_back(const value_type& value) {
		assert(m_size < m_maxSize);
		*m_End = value_type(value);
		++m_End;
		++m_size;
	}


	template<typename T>
	DYN_FUNC void RandomAccessContainer<T>::push_back(value_type&& value) {
		assert(m_size < m_maxSize);
		*m_End = value_type(dyno::forward<value_type>(value));
		++m_End;
		++m_size;
	}

	template<typename T>
	DYN_FUNC RandomAccessContainer<T>::reference
		RandomAccessContainer<T>::push_back() {
		assert(m_size < m_maxSize);
		value_type tmp = value_type();
		push_back(tmp);
		return *(m_End - 1);
	}

	template<typename T>
	DYN_FUNC void RandomAccessContainer<T>::pop_back() {
		assert(m_size != 0);
		--m_End;
		m_End->~value_type(); //destruct
		--m_size;
	}

	//inplace move
	template <typename InputIterator, typename OutputIterator>
	DYN_FUNC static OutputIterator move_or_copy(InputIterator first, InputIterator last, OutputIterator result)
	{
		for (; first != last; ++result, ++first)
			*result = *first;
		return result;
	}

	//inplace back_move
	template <typename Iterator>
	DYN_FUNC static Iterator move_or_copy_backward(Iterator first, Iterator last, Iterator resultEnd)
	{
		while (first != last)
			*--resultEnd = *--last;
		return resultEnd;
	}

	template<typename T>
	DYN_FUNC RandomAccessContainer<T>::iterator
		RandomAccessContainer<T>::insert(const_iterator position, const value_type& value) {
		assert(m_size < m_maxSize);
		assert(position >= m_Begin && position <= m_End);
		push_back();
		const difference_type n = position - m_Begin;
		iterator destPosition = const_cast<value_type*>(position);
		dyno::move_or_copy_backward(destPosition, m_End - 1, m_End);
		*destPosition = value_type(value);
		return m_Begin + n;
	}


	template<typename T>
	DYN_FUNC RandomAccessContainer<T>::iterator
		RandomAccessContainer<T>::insert(const_iterator position, value_type&& value) {
		assert(m_size < m_maxSize);
		assert(position >= m_Begin && position <= m_End);
		push_back();
		const difference_type n = position - m_Begin;
		iterator destPosition = const_cast<value_type*>(position);
		dyno::move_or_copy_backward(destPosition, m_End - 1, m_End);
		*destPosition = value_type(dyno::forward<value_type>(value));
		return m_Begin + n;
	}


	template<typename T>
	DYN_FUNC RandomAccessContainer<T>::iterator
		RandomAccessContainer<T>::insert(const_iterator position, size_type n, const value_type& value) {
		assert((m_size + n) <= m_maxSize);
		assert(position >= m_Begin && position <= m_End);
		const difference_type p = position - m_Begin;
		if (n > 0) {
			for (int i = 0; i < n; ++i)
				push_back();
			iterator destPosition = const_cast<value_type*>(position);
			dyno::move_or_copy_backward(destPosition, m_End - n, m_End);

			for (; destPosition != (position + n); ++destPosition) {
				*destPosition = value_type(value);
			}
		}
		return m_Begin + p;
	}

	template<typename T>
	template <typename InputIterator>
	DYN_FUNC RandomAccessContainer<T>::iterator
		RandomAccessContainer<T>::insert(const_iterator position, InputIterator first, InputIterator last) {
		assert((m_size + (size_type)(last - first)) <= m_maxSize);
		assert(position >= m_Begin && position <= m_End);
		const difference_type p = position - m_Begin;
		size_type n = (size_type)(last - first);
		if (n > 0) {
			size_type i = 0;
			for (size_type i = 0; i < n; ++i)
				push_back();
		
			iterator destPosition = const_cast<value_type*>(position);
			dyno::move_or_copy_backward(destPosition, m_End - n, m_End);
			for (; destPosition != (position + n); ++destPosition, ++first) {
				iterator First = static_cast<value_type*>(first);
				*destPosition = *(First);
			}
		}

		return m_Begin + p;
	}

	template<typename T>
	DYN_FUNC inline RandomAccessContainer<T>::iterator
		RandomAccessContainer<T>::find(iterator first, iterator last, const value_type& value) {
		while ((first != last) && !(*first == value))
			++first;
		return first;
	}

	template<typename T>
	template<typename Predicate>
	DYN_FUNC inline RandomAccessContainer<T>::iterator
		RandomAccessContainer<T>::find(iterator first, iterator last, const value_type& value, Predicate pre) {
		while ((first != last) && !pre(*first, value))
			++first;
		return first;
	}

	template<typename T>
	DYN_FUNC RandomAccessContainer<T>::iterator
		RandomAccessContainer<T>::erase_first(const T& value) {
		iterator it = find(m_Begin, m_End, value);

		if (it != m_End)
			return erase(it);
		else
			return it;
	}

	template<typename T>
	DYN_FUNC RandomAccessContainer<T>::iterator
		RandomAccessContainer<T>::erase_first_unsorted(const T& value) {
		iterator it = find(m_Begin, m_End, value);

		if (it != m_End)
			return erase_unsorted(it);
		else
			return it;
	}

	template<typename T>
	DYN_FUNC RandomAccessContainer<T>::iterator
		RandomAccessContainer<T>::erase(const_iterator position) {
		assert(position >= m_Begin && position < m_End);
		iterator destPosition = const_cast<value_type*>(position);
		//inplace movement
		if ((position + 1) < m_End)
			dyno::move_or_copy(destPosition + 1, m_End, destPosition);

		(--m_End)->~value_type();
		--m_size;
		return destPosition;
	}

	template<typename T>
	DYN_FUNC RandomAccessContainer<T>::iterator
		RandomAccessContainer<T>::erase(const_iterator first, const_iterator last) {
		assert(first >= m_Begin && last <= m_End);
		iterator destFirstPosition = const_cast<value_type*>(first);
		iterator destLastPosition = const_cast<value_type*>(last);
		if (first != last && (destLastPosition) < m_End) {
			dyno::move_or_copy(destLastPosition, m_End, destFirstPosition);
		}
		iterator end = m_End;
		for (; m_End != end - (destLastPosition - destFirstPosition); ) {

			(--m_End)->~value_type();
			--m_size;
		}
		return destFirstPosition;
	}

	template<typename T>
	DYN_FUNC RandomAccessContainer<T>::iterator
		RandomAccessContainer<T>::erase_unsorted(const_iterator position) {
		assert(position >= m_Begin && position < m_End);
		iterator destPosition = const_cast<value_type*>(position);
		*destPosition = *(m_End - 1);
		(--m_End)->~value_type();
		--m_size;
		return destPosition;
	}

	template<typename T>
	DYN_FUNC void RandomAccessContainer<T>::clear() noexcept {
		if (!empty()) {
			for (; m_End != m_Begin;)
				(--m_End)->~value_type();

			m_size = 0;
			//assert(m_End == m_Begin);
		}
	}

	//global operators
	template <typename InputIterator1, typename InputIterator2>
	DYN_FUNC inline bool
		lexicographical_compare(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2)
	{
		for (; (first1 != last1) && (first2 != last2); ++first1, ++first2)
		{
			if (*first1 < *first2)
				return true;
			if (*first2 < *first1)
				return false;
		}
		return (first1 == last1) && (first2 != last2);
	}

	template <typename T>
	inline bool operator==(const RandomAccessContainer<T>& a, const RandomAccessContainer<T>& b)
	{
		return ((a.size() == b.size()) && (a.begin() == b.begin()));
	}


	template <typename T>
	inline bool operator!=(const RandomAccessContainer<T>& a, const RandomAccessContainer<T>& b)
	{
		return ((a.size() != b.size()) || !(a.begin() == b.begin()));
	}


	template <typename T>
	inline bool operator<(const RandomAccessContainer<T>& a, const RandomAccessContainer<T>& b)
	{
		return lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
	}


	template <typename T>
	inline bool operator>(const RandomAccessContainer<T>& a, const RandomAccessContainer<T>& b)
	{
		return b < a;
	}


	template <typename T>
	inline bool operator<=(const RandomAccessContainer<T>& a, const RandomAccessContainer<T>& b)
	{
		return !(b < a);
	}


	template <typename T>
	inline bool operator>=(const RandomAccessContainer<T>& a, const RandomAccessContainer<T>& b)
	{
		return !(a < b);
	}

}