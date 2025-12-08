#include "BasicShape2D.h"

namespace dyno
{
	template<typename TDataType>
	BasicShape2D<TDataType>::BasicShape2D()
		: ParametricModel<TDataType>()
	{
	}

	DEFINE_CLASS(BasicShape2D);
}