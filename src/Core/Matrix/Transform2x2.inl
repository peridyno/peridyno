#include <cmath>
#include <limits>

#include "Vector.h"

namespace dyno
{
	template <typename T>
	DYN_FUNC Transform<T, 2>::Transform()
	{
		mTranslation = Vector<T, 2>(0);
		mScale = Vector<T, 2>(1);
		mRotation = SquareMatrix<T, 2>(1, 0, 0, 1);
	}

	template <typename T>
	DYN_FUNC Transform<T, 2>::Transform(const Vector<T, 2>& t, const T& angle, const Vector<T, 2>& s)
	{
		mTranslation = t;
		mScale = s;
		mRotation = SquareMatrix<T, 2>(glm::cos(angle), -glm::sin(angle), glm::sin(angle), glm::cos(angle));
	}

	template <typename T>
	DYN_FUNC Transform<T, 2>::Transform(const Vector<T, 2>& t, const SquareMatrix<T, 2>& m, const Vector<T, 2>& s /*= Vector<T, 2>(1)*/)
	{
		mTranslation = t;
		mScale = s;
		mRotation = m;
	}

	template <typename T>
	DYN_FUNC Transform<T, 2>::Transform(const Transform<T, 2>& t)
	{
		mRotation = t.mRotation;
		mTranslation = t.mTranslation;
		mScale = t.mScale;
	}

	template <typename T>
	DYN_FUNC Transform<T, 2>::~Transform()
	{

	}

	template <typename T>
	DYN_FUNC const Vector<T, 2> Transform<T, 2>::operator* (const Vector<T, 2> &vec) const
	{
		Vector<T, 2> scaled = Vector<T, 2>(vec.x*mScale.x, vec.y*mScale.y);
		return mRotation * scaled + mTranslation;
	}
}
