#pragma once
#include "PointSet.h"


namespace dyno
{
	template<typename DataType3f>
	class UnstructuredPointSet : public PointSet<DataType3f>
	{
	public:
		UnstructuredPointSet();
		~UnstructuredPointSet();
	};
}

