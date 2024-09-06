#pragma once
#include "TriangleSet.h"

namespace dyno
{
	class TetrahedronSet : public TriangleSet
	{
	public:
		TetrahedronSet();
		~TetrahedronSet() override;

		void setTetrahedrons(std::vector<TopologyModule::Tetrahedron>& indices);

		void copyFrom(TetrahedronSet& es);

		DArray<TopologyModule::Tetrahedron>& getTetrahedrons() { return mTetrahedronIndex; }

	protected:
		void updateTopology() override;
	
	public:
		DArray<TopologyModule::Tetrahedron> mTetrahedronIndex;
	};
}

