#pragma once
#include "TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class TetrahedronSet : public TriangleSet<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename TopologyModule::Tetrahedron Tetrahedron;

		TetrahedronSet();
		~TetrahedronSet() override;

		void loadTetFile(std::string filename);

		void setTetrahedrons(std::vector<Tetrahedron>& tetrahedrons);
		void setTetrahedrons(DArray<Tetrahedron>& tetrahedrons);

		DArray<Tetrahedron>& tetrahedronIndices() { return mTethedrons; }

		const DArrayList<int>& vertex2Tetrahedron() { return mVer2Tet; }

		const DArray<::dyno::TopologyModule::Tri2Tet>& triangle2Tetrahedron() { return mTri2Tet; }
		const DArray<::dyno::TopologyModule::Tet2Tri>& tetrahedron2Triangle() { return mTet2Tri; }

		void copyFrom(TetrahedronSet<TDataType>& tetSet);

		bool isEmpty() override;

	public:
		/**
		 * @brief Request the neighboring ids of each point according to the mesh topology
		 *			Neighbor contains 1 - hop and 2 - hop vertex (in horizon H)
		 * 			Be sure update() is called as long as the topology is changed
		 *
		 * @param lists A neighbor list storing the ids
		 */
		void requestPointNeighbors(DArrayList<int>& lists);

		void requestSurfaceMeshIds(
			DArray<int>& surfaceIds,
			DArray<int>& towardOutside,
			DArray<::dyno::TopologyModule::Tri2Tri>& t2t);

		void extractSurfaceMesh(TriangleSet<TDataType>& ts);

		void calculateVolume(DArray<Real>& volume);

	protected:
		void updateTopology() override;

		void updateTriangles() override;

		virtual void updateTetrahedrons() {}

	private:
		void updateVertex2Tetrahedron();

		DArray<Tetrahedron> mTethedrons;

		// Automatically updated when update() is called
		DArrayList<int> mVer2Tet;

		// Mapping for tetrahedron mesh
		DArray<::dyno::TopologyModule::Tri2Tet> mTri2Tet;
		DArray<::dyno::TopologyModule::Tet2Tri> mTet2Tri;
	};
}

