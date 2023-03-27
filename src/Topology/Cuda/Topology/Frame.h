#pragma once
#include "Module/TopologyModule.h"
#include "Vector.h"

namespace dyno
{
	/*!
	*	\class	Frame
	*	\brief	A frame represents a point equipped with the orientation.
	*/

	template<typename TDataType>
	class Frame : public TopologyModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		Frame();
		~Frame() override {};

		void copyFrom(Frame<TDataType>& frame);

		void setCenter(Coord c);

		Coord getCenter(){ return m_coord; }

		void setOrientation(Matrix mat) { m_rotation = mat; }
		Matrix getOrientation() { return m_rotation; }

	protected:
		Coord m_coord;
		Matrix m_rotation;
	};
}

