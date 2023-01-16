#include "TriangleSet.h"
#include "VkTransfer.h"

namespace dyno
{

	TriangleSet::TriangleSet()
	{
		
	}

	TriangleSet::~TriangleSet()
	{

	}

	void TriangleSet::updateTopology()
	{
		this->updateTriangles();

		EdgeSet::updateTopology();
	}

	void TriangleSet::updateTriangles()
	{
		//TODO: this is temporary, should be removed later
		std::vector<Triangle> triSet(mTriangleIndex.size());
		std::vector<uint32_t> triIndex;

		vkTransfer(triSet, *mTriangleIndex.handle());
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

		//vkTransfer(mIndex, triIndex);
		mIndex.assign(triIndex);

		triIndex.clear();
	}

}