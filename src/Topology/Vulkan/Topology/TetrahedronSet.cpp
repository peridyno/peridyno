#include "TetrahedronSet.h"
#include "VkTransfer.h"

namespace px
{
	TetrahedronSet::TetrahedronSet()
	{
		
	}

	TetrahedronSet::~TetrahedronSet()
	{

	}

	void TetrahedronSet::updateTopology()
	{
		std::vector<dyno::TopologyModule::Tetrahedron> tets(mTetrahedronIndex.size());
		std::vector<uint32_t> tris;

		std::vector<dyno::TopologyModule::Triangle> triangles;

		//TODO: atomic operations are not supported yet, replace the following implementation with a parallel algorithm later.
		vkTransfer(tets, mTetrahedronIndex);
		for (size_t i = 0; i < tets.size(); i++)
		{
			uint32_t v0 = tets[i][0];
			uint32_t v1 = tets[i][1];
			uint32_t v2 = tets[i][2];
			uint32_t v3 = tets[i][3];

			tris.push_back(v0);
			tris.push_back(v1);
			tris.push_back(v2);

			tris.push_back(v0);
			tris.push_back(v3);
			tris.push_back(v1);

			tris.push_back(v0);
			tris.push_back(v2);
			tris.push_back(v3);

			tris.push_back(v1);
			tris.push_back(v3);
			tris.push_back(v2);

			triangles.push_back(dyno::TopologyModule::Triangle(v0, v1, v2));
			triangles.push_back(dyno::TopologyModule::Triangle(v0, v1, v3));
			triangles.push_back(dyno::TopologyModule::Triangle(v0, v2, v3));
			triangles.push_back(dyno::TopologyModule::Triangle(v1, v2, v3));
		}

		mIndex.resize(tris.size());
		mTriangleIndex.resize(triangles.size());

		vkTransfer(mIndex, tris);
		vkTransfer(mTriangleIndex, triangles);

		tets.clear();
		tris.clear();
		triangles.clear();
	}

}