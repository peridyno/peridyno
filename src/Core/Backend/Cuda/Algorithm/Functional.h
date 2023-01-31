#pragma once
#include "Platform.h"
#include "Math/SimpleMath.h"


namespace dyno
{
	template<typename T>
	struct PlusFunc
	{
		/*! Function call operator. The return value is <tt>lhs + rhs</tt>.
		*/
		DYN_FUNC inline T operator()(const T lhs, const T rhs) const { return lhs + rhs; }
	}; // end plus

	template<typename T>
	struct MinusFunc
	{
		/*! Function call operator. The return value is <tt>lhs - rhs</tt>.
		*/
		DYN_FUNC inline T operator()(const T lhs, const T rhs) const { return lhs - rhs; }
	}; // end minus

	template<typename T>
	struct MultiplyFunc
	{
		/*! Function call operator. The return value is <tt>lhs * rhs</tt>.
		*/
		DYN_FUNC inline T operator()(const T lhs, const T rhs) const { return lhs * rhs; }
	}; // end multiplies

	template<typename T>
	struct DivideFunc
	{
		/*! Function call operator. The return value is <tt>lhs / rhs</tt>.
		*/
		DYN_FUNC inline T operator()(const T lhs, const T rhs) const { return lhs / rhs; }
	}; // end divides

	template<typename T>
	struct ModulusFunc
	{
		/*! Function call operator. The return value is <tt>lhs % rhs</tt>.
		*/
		DYN_FUNC inline T operator()(const T lhs, const T rhs) const { return lhs % rhs; }
	}; // end modulus

	template<typename T>
	struct NegateFunc
	{
		/*! Function call operator. The return value is <tt>-x</tt>.
		*/
		DYN_FUNC inline T operator()(const T x) const { return -x; }
	}; // end negate

	template<typename T>
	struct EqualFunc
	{
		/*! Function call operator. The return value is <tt>lhs == rhs</tt>.
		*/
		DYN_FUNC inline bool operator()(const T lhs, const T rhs) const { return lhs == rhs; }
	}; // end equal_to

	template<typename T>
	struct NotEqualFunc
	{
		/*! Function call operator. The return value is <tt>lhs != rhs</tt>.
		*/
		DYN_FUNC inline bool operator()(const T lhs, const T rhs) const { return lhs != rhs; }
	}; // end not_equal_to

	template<typename T>
	struct GreaterFunc
	{
		/*! Function call operator. The return value is <tt>lhs > rhs</tt>.
		*/
		DYN_FUNC inline bool operator()(const T lhs, const T rhs) const { return lhs > rhs; }
	}; // end greater

	template<typename T>
	struct LessFunc
	{
		/*! Function call operator. The return value is <tt>lhs < rhs</tt>.
		*/
		DYN_FUNC inline bool operator()(const T lhs, const T rhs) const { return lhs < rhs; }
	}; // end less

	template<typename T>
	struct GreaterEqualFunc
	{
		/*! Function call operator. The return value is <tt>lhs >= rhs</tt>.
		*/
		DYN_FUNC inline bool operator()(const T lhs, const T rhs) const { return lhs >= rhs; }
	}; // end greater_equal

	template<typename T>
	struct LessEqualFunc
	{
		/*! Function call operator. The return value is <tt>lhs <= rhs</tt>.
		*/
		DYN_FUNC inline bool operator()(const T lhs, const T rhs) const { return lhs <= rhs; }
	}; // end less_equal

	template<typename T>
	struct LogicalAndFunc
	{
		/*! Function call operator. The return value is <tt>lhs && rhs</tt>.
		*/
		DYN_FUNC inline bool operator()(const T lhs, const T rhs) const { return lhs && rhs; }
	}; // end logical_and

	template<typename T>
	struct LogicalOrFunc
	{
		/*! Function call operator. The return value is <tt>lhs || rhs</tt>.
		*/
		DYN_FUNC inline bool operator()(const T lhs, const T rhs) const { return lhs || rhs; }
	}; // end logical_or

	template<typename T>
	struct LogicalNotFunc
	{
		/*! Function call operator. The return value is <tt>!x</tt>.
		*/
		DYN_FUNC inline bool operator()(const T &x) const { return !x; }
	}; // end logical_not

	template<typename T>
	struct BitAndFunc
	{
		/*! Function call operator. The return value is <tt>lhs & rhs</tt>.
		*/
		DYN_FUNC inline T operator()(const T lhs, const T rhs) const { return lhs & rhs; }
	}; // end bit_and

	template<typename T>
	struct BitOrFunc
	{
		/*! Function call operator. The return value is <tt>lhs | rhs</tt>.
		*/
		DYN_FUNC inline T operator()(const T lhs, const T rhs) const { return lhs | rhs; }
	}; // end bit_or

	template<typename T>
	struct BitXorFunc
	{
		/*! Function call operator. The return value is <tt>lhs ^ rhs</tt>.
		*/
		DYN_FUNC inline T operator()(const T lhs, const T rhs) const { return lhs ^ rhs; }
	}; // end bit_xor

	template<typename T>
	struct MaximumFunc
	{
		/*! Function call operator. The return value is <tt>rhs < lhs ? lhs : rhs</tt>.
		*/
		DYN_FUNC inline T operator()(const T lhs, const T rhs) const { return maximum(lhs, rhs); }
	}; // end maximum

	template<typename T>
	struct MinimumFunc
	{
		/*! Function call operator. The return value is <tt>lhs < rhs ? lhs : rhs</tt>.
		*/
		DYN_FUNC inline T operator()(const T lhs, const T rhs) const { return minimum(lhs, rhs); }
	}; // end minimum
}