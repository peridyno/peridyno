#pragma once
#include "TriangleSet.h"

namespace dyno
{
	class TetrahedronSet : public TriangleSet
	{
	public:
		TetrahedronSet();
		~TetrahedronSet() override;

		void updateTopology() override;
	public:
		VkDeviceArray<TopologyModule::Tetrahedron> mTetrahedronIndex;
	};
}

