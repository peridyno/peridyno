#pragma once
#include <iostream>
#include "rigid_base.h"
#include "../Vector.h"
#include "../Matrix.h"

namespace dyno {
	template <typename T>
	class Rigid<T, 2>
	{
	public:
		typedef Vector<T, 2> TranslationDOF;
		typedef T RotationDOF;

		DYN_FUNC Rigid()
			: m_p(0)
			, m_angle(0)
		{};

		DYN_FUNC Rigid(Vector<T, 2> p, T angle)
			: m_p(p)
			, m_angle(angle)
		{};

		DYN_FUNC ~Rigid() {};

		DYN_FUNC T getOrientation() const { return m_angle; }
		DYN_FUNC Vector<T, 2> getCenter() const { return m_p; }
		DYN_FUNC SquareMatrix<T, 2> getRotationMatrix() const {
			return SquareMatrix<T, 2>(glm::cos(m_angle), -glm::sin(m_angle),
				glm::sin(m_angle), glm::cos(m_angle));
		}
	private:
		Vector<T, 2> m_p;
		T m_angle;
	};

	template class Rigid<float, 2>;
	template class Rigid<double, 2>;
	//convenient typedefs 
	typedef Rigid<float, 2> Rigid2f;
	typedef Rigid<double, 2> Rigid2d;

}  //end of namespace dyno
