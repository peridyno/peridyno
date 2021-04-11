#pragma once
#include "PointSet.h"


namespace dyno
{
	template<typename DataType3f>
	class StructuredPointSet : public PointSet<DataType3f>
	{
	public:
		StructuredPointSet();
		~StructuredPointSet();
	};
}

