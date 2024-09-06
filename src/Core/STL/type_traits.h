#ifndef TYPE_TRAITS_H
#define TYPE_TRAITS_H

#include "Platform.h"

namespace dyno
{
	
	template <typename T> struct remove_reference { typedef T type; };

	template <typename T> struct remove_reference<T&> { typedef T type; };

	template <typename T> struct remove_reference<T&&> { typedef T type; };


	template <typename T>
	DYN_FUNC constexpr T&& forward(typename dyno::remove_reference<T>::type& x) noexcept
	{
		return static_cast<T&&>(x);
	}


	template <typename T>
	DYN_FUNC constexpr T&& forward(typename dyno::remove_reference<T>::type&& x) noexcept
	{
		// should promise T isn't lvalue reference 
		return static_cast<T&&>(x);
	}

	template <typename T>
	struct iterator_traits {};

	typedef long long ptrDiff_t;

	template <typename T>
	struct iterator_traits<T*>
	{
		typedef T                                   value_type;
		typedef ptrDiff_t                           difference_type;
		typedef T*									pointer;
		typedef T&									reference;
	};

	template <typename T>
	struct iterator_traits<const T*>
	{
		typedef T                                   value_type;
		typedef ptrDiff_t                           difference_type;
		typedef const T*							pointer;
		typedef const T&							reference;
	};
}


#endif // TYPE_TRAITS_H
