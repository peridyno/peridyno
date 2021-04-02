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

	private:

		/**
		* @brief Neighboring particles
		*
		*/
		DEF_EMPTY_IN_NEIGHBOR_LIST(Neighborhood, int, "Neighboring particles' ids");
	};
}

