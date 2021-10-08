#include <cmath>
#include <limits>

#include "Vector.h"

namespace dyno
{
	template <typename T>
	DYN_FUNC Transform<T, 3>::Transform()
	{
	}

	template <typename T>
	DYN_FUNC Transform<T, 3>::Transform(const SquareMatrix<T, 3>& rot, const Vector<T, 3>& trans)
	{
		mRotation = rot;
		mTranslation = trans;
	}

	template <typename T>
	DYN_FUNC Transform<T, 3>::Transform(const Transform<T, 3>& t)
	{
		mRotation = t.mRotation;
		mTranslation = t.mTranslation;
	}

	template <typename T>
	DYN_FUNC Transform<T, 3>::~Transform()
	{

	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Transform<T, 3>::operator* (const Vector<T, 3> &vec) const
	{
		return mRotation * vec + mTranslation;
	}
}
