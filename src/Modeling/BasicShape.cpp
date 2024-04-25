#include "BasicShape.h"

namespace dyno
{
	template<typename TDataType>
	BasicShape<TDataType>::BasicShape()
		: ParametricModel<TDataType>()
	{
	}

	DEFINE_CLASS(BasicShape);
}