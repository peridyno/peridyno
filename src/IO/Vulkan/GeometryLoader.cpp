#include "GeometryLoader.h"

namespace dyno
{
	template<typename TDataType>
	GeometryLoader<TDataType>::GeometryLoader()
		: ParametricModel<TDataType>()
	{

	}

	DEFINE_CLASS(GeometryLoader);
}