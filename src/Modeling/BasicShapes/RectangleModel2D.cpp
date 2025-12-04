#include "BasicShapes/RectangleModel2D.h"

namespace dyno
{
	template<typename TDataType>
	RectangleModel2D<TDataType>::RectangleModel2D()
		: BasicShape2D<TDataType>()
	{
	}

	DEFINE_CLASS(RectangleModel2D);
}