#pragma once
#include <iostream>
#include "rigid_base.h"
#include "../Vector.h"
#include "../Matrix.h"
#include "Quat.h"

namespace dyno {
	template <typename T>
	class Rigid<T, 3>
	{
	public:
		typedef Vector<T, 3> TranslationDOF;
		typedef Vector<T, 3> RotationDOF;

		DYN_FUNC Rigid()
			: m_p(0)
			, m_quat(Quat<T>::identity())
		{};

		DYN_FUNC Rigid(Vector<T, 3> p, Quat<T> quat)
			: m_p(p)
			, m_quat(quat)
		{};

		DYN_FUNC ~Rigid() {};

		DYN_FUNC Vector<T, 3> getCenter() const { return m_p; }

		DYN_FUNC SquareMatrix<T, 3> getRotationMatrix() const
		{
			return m_quat.toMatrix3x3();
		}

		DYN_FUNC Quat<T> getOrientation() const { return m_quat; }

	private:
		Vector<T, 3> m_p;
		Quat<T> m_quat;
	};

	template class Rigid<float, 3>;
	template class Rigid<double, 3>;

	typedef Rigid<float, 3> Rigid3f;
	typedef Rigid<double, 3> Rigid3d;
}  //end of namespace dyno
