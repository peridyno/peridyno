#ifndef FUNCTIONAL_BASE_H
#define FUNCTIONAL_BASE_H

#include "Platform.h"

namespace dyno
{
	template <typename Argument, typename Result>
	struct unary_function
	{
		typedef Argument argument_type;
		typedef Result   result_type;
		DYN_FUNC unary_function() = default;
	};


	template <typename Argument1, typename Argument2, typename Result>
	struct binary_function
	{
		typedef Argument1 first_argument_type;
		typedef Argument2 second_argument_type;
		typedef Result    result_type;
		DYN_FUNC binary_function() = default;
	};

	template <typename T = void>
	struct less : public binary_function<T, T, bool>
	{
		DYN_FUNC constexpr bool operator()(const T& a, const T& b) const
		{
			return a < b;
		}
	};

	template <typename T = void>
	struct greater : public binary_function<T, T, bool>
	{
		DYN_FUNC constexpr bool operator()(const T& a, const T& b) const
		{
			return a > b;
		}
	};
	
	template <typename T = void>
	struct predicate : public binary_function<T, T, bool> {
		DYN_FUNC constexpr bool operator()(const T& a, const T& b) const
		{
			return a == b;
		}
	};
}


#endif // FUNCTIONAL_BASE
