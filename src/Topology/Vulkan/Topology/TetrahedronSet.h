#pragma once
#include "TriangleSet.h"

namespace px
{
	class TetrahedronSet : public TriangleSet
	{
	public:
		TetrahedronSet();
		~TetrahedronSet() override;

		void updateTopology() override;
	public:
		VkDeviceArray<dyno::TopologyModule::Tetrahedron> mTetrahedronIndex;
	};
}

