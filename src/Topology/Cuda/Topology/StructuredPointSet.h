#pragma once
#include "PointSet.h"


namespace dyno
{
	template<typename TDataType>
	class StructuredPointSet : public PointSet<TDataType>
	{
	public:
		StructuredPointSet();
		~StructuredPointSet();
	};
}

