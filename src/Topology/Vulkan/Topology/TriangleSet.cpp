#include "TriangleSet.h"
#include "VkTransfer.h"

namespace px
{

	TriangleSet::TriangleSet()
	{
		
	}

	TriangleSet::~TriangleSet()
	{

	}

	void TriangleSet::updateTopology()
	{
		std::vector<Triangle> triSet(mTriangleIndex.size());
		std::vector<uint32_t> triIndex;

		vkTransfer(triSet, mTriangleIndex);
		for (size_t i = 0; i < triSet.size(); i++)
		{
			uint32_t v0 = triSet[i][0];
			uint32_t v1 = triSet[i][1];
			uint32_t v2 = triSet[i][2];

			triIndex.push_back(v2);
			triIndex.push_back(v1);
			triIndex.push_back(v0);
		}

		mIndex.resize(triIndex.size());

		vkTransfer(mIndex, triIndex);

		triIndex.clear();
	}

}