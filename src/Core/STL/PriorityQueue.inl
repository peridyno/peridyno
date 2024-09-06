
namespace dyno {

	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline priority_queue<T, Container, Compare>::priority_queue()
		: container(),
		cmp()
	{
	}

	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline priority_queue<T, Container, Compare>::~priority_queue()
	{
	}


	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline priority_queue<T, Container, Compare>::priority_queue(const container_type& c)
		: container(c),
		cmp()
	{
	}

	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline priority_queue<T, Container, Compare>::priority_queue(const compare_type& compare)
		: container(),
		cmp(compare)
	{
	}

	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline priority_queue<T, Container, Compare>::priority_queue(const container_type& c, const compare_type& compare)
		: container(c),
		cmp(compare)
	{
	}



	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline bool priority_queue<T, Container, Compare>::empty() const
	{
		return container.empty();
	}


	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline bool priority_queue<T,Container, Compare>::isEmpty() const {
		return container.size() == 0;
	}

	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline typename priority_queue<T, Container, Compare>::size_type
		priority_queue<T, Container, Compare>::size() const
	{
		return container.size();
	}

	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline void priority_queue<T, Container, Compare>::clear()	
	{
		container.clear();
	}


	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline typename priority_queue<T, Container, Compare>::const_reference
		priority_queue<T, Container, Compare>::top() const
	{
		return container.front();
	}


	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline void priority_queue<T, Container, Compare>::push(const value_type& value)
	{
		container.push_back(value);
		dyno::push_heap(container.begin(), container.end(), cmp);
	}


	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline void priority_queue<T, Container, Compare>::pop()
	{
		dyno::pop_heap(container.begin(), container.end(), cmp);
		container.pop_back();
	}

	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline void priority_queue<T, Container, Compare>::change(size_type n) // This function is not in the STL std::priority_queue.
	{
		dyno::change_heap(container.begin(), container.size(), n, cmp);
	}


	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline void priority_queue<T, Container, Compare>::remove(size_type n) // This function is not in the STL std::priority_queue.
	{
		assert(n < container.size() && n >= 0);
		dyno::remove_heap(container.begin(), container.size(), n, cmp);
		container.pop_back();
	}

	// global operators
	template <typename T, typename Container, typename Compare>
	DYN_FUNC bool operator==(const priority_queue<T, Container, Compare>& a, const priority_queue<T, Container, Compare>& b)
	{
		return (a.container == b.container);
	}

	template <typename T, typename Container, typename Compare>
	DYN_FUNC bool operator<(const priority_queue<T, Container, Compare>& a, const priority_queue<T, Container, Compare>& b)
	{
		return (a.container < b.container);
	}

	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline bool operator!=(const priority_queue<T, Container, Compare>& a, const priority_queue<T, Container, Compare>& b)
	{
		return !(a.container == b.container);
	}

	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline bool operator>(const priority_queue<T, Container, Compare>& a, const priority_queue<T, Container, Compare>& b)
	{
		return (b.container < a.container);
	}

	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline bool operator<=(const priority_queue<T, Container, Compare>& a, const priority_queue<T, Container, Compare>& b)
	{
		return !(b.container < a.container);
	}

	template <typename T, typename Container, typename Compare>
	DYN_FUNC inline bool operator>=(const priority_queue<T, Container, Compare>& a, const priority_queue<T, Container, Compare>& b)
	{
		return !(a.container < b.container);
	}
}