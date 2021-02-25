#ifndef PAIR_H
#define PAIR_H

#include "Platform.h"

namespace dyno
{
	template <typename Key, typename T>
	class Pair
	{
	public:
		DYN_FUNC Pair() {};
		DYN_FUNC Pair(Key key, T val) { first = key; second = val; }

		DYN_FUNC inline bool operator>= (const Pair& other) const {
			return first >= other.first;
		}

		DYN_FUNC inline bool operator> (const Pair& other) const {
			return first > other.first;
		}

		DYN_FUNC inline bool operator<= (const Pair& other) const {
			return first <= other.first;
		}

		DYN_FUNC inline bool operator< (const Pair& other) const {
			return first < other.first;
		}

		DYN_FUNC inline bool operator== (const Pair& other) const {
			return first == other.first;
		}

		DYN_FUNC inline bool operator!= (const Pair& other) const {
			return first != other.first;
		}

	public:
		Key first;
		T second;
	};
}

#endif // PAIR_H
