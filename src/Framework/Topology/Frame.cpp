#pragma once
#include "Frame.h"

namespace dyno
{

	template<typename TDataType>
	Frame<TDataType>::Frame()
	{
		m_coord = Coord(0);
		m_rotation = Matrix::identityMatrix();
	}

	template<typename TDataType>
	void Frame<TDataType>::setCenter(Coord c)
	{
		m_coord = c;
	}


	template<typename TDataType>
	void Frame<TDataType>::copyFrom(Frame<TDataType>& frame)
	{
		m_coord = frame.m_coord;
		m_rotation = frame.m_rotation;
	}

	DEFINE_CLASS(Frame);
}

