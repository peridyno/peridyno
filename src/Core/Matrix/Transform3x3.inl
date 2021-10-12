#include <cmath>
#include <limits>

#include "Vector.h"

namespace dyno
{
	template <typename T>
	DYN_FUNC Transform<T, 3>::Transform()
	{
		mTranslation = Vector<T, 3>(0);
		mScale = Vector<T, 3>(1);
		mRotation = SquareMatrix<T, 3>::identityMatrix();
	}

	template <typename T>
	DYN_FUNC Transform<T, 3>::Transform(const Vector<T, 3>& t, const SquareMatrix<T, 3>& M, const Vector<T, 3>& s)
	{
		mTranslation = t;
		mScale = s;
		mRotation = M;
	}

	template <typename T>
	DYN_FUNC Transform<T, 3>::Transform(const Transform<T, 3>& t)
	{
		mRotation = t.mRotation;
		mTranslation = t.mTranslation;
		mScale = t.mScale;
	}

	template <typename T>
	DYN_FUNC Transform<T, 3>::~Transform()
	{

	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Transform<T, 3>::operator* (const Vector<T, 3> &vec) const
	{
		Vector<T, 3> scaled = Vector<T, 3>(vec.x*mScale.x, vec.y*mScale.y, vec.z*mScale.z);
		return mRotation * scaled + mTranslation;
	}
}
