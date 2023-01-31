#pragma once
#include "TriangleSet.h"

namespace dyno
{
	class TetrahedronSet : public TriangleSet
	{
	public:
		TetrahedronSet();
		~TetrahedronSet() override;

	protected:
		void updateTopology() override;
	
	public:
		DArray<TopologyModule::Tetrahedron> mTetrahedronIndex;
	};
}

