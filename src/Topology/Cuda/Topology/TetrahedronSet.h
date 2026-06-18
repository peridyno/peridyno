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
		typedef typename Topology::Edge Edge;
		typedef typename Topology::Triangle Triangle;
		typedef typename Topology::Tetrahedron Tetrahedron;

		TetrahedronSet();
		~TetrahedronSet() override;

		void loadTetFile(std::string filename);

		void setTetrahedrons(std::vector<Tetrahedron>& tetrahedrons);
		void setTetrahedrons(DArray<Tetrahedron>& tetrahedrons);

 		DArray<Tetrahedron>& tetrahedronIndices() { return mTethedrons; }

		const DArrayList<int>& vertex2Tetrahedron() { return mVer2Tet; }

		const DArray<::dyno::Topology::Tri2Tet>& triangle2Tetrahedron() { return mTri2Tet; }
		const DArray<::dyno::Topology::Tet2Tri>& tetrahedron2Triangle() { return mTet2Tri; }

		void copyFrom(TetrahedronSet<TDataType>& tetSet);

		std::shared_ptr<TetrahedronSet<TDataType>> 
			merge(TetrahedronSet<TDataType>& tetSet);

		bool isEmpty() override;

	public:
		static std::shared_ptr<TetrahedronSet<TDataType>> 
			merge(std::vector<std::shared_ptr<TetrahedronSet<TDataType>>>& tsArray);

		/**
		 * @brief Request the neighboring ids of each point according to the mesh topology
		 *			Neighbor contains 1 - hop and 2 - hop vertex (in horizon H)
		 * 			Be sure update() is called as long as the topology is changed
		 *
		 * @param lists A neighbor list storing the ids
		 */
		void requestPointNeighbors(DArrayList<int>& lists);

		/**
		 * @brief Request the vertices located on the boundary, while keeping the vertices the same as in the TetrahedronSet
		 */
		void requestBoundaryVertexIndices(DArray<int>& indices);

		/**
		 * @brief Request edges located on the boundary, while keeping the vertices the same as in the TetrahedronSet
		 */
		void requestBoundaryEdgeIndices(DArray<Topology::Edge>& indices);

		/**
		 * @brief Request the triangles located on the boundary, while keeping the vertices the same as in the TetrahedronSet
		 */
		void requestBoundaryTriangleIndices(DArray<Topology::Triangle>& indices);

		/**
		 * @brief Extract a TriangleSet representing the boundary, only vertices on the boundary will be stored.
		 *			
		 * @param ts	   The boundary mesh represented as a new TriangleSet, vertices inside will be removed.
		 * @param indices  Record the position of surface vertices located at the corresponding vertex array of the tetrahedron set.
		 */
		void extractSurfaceMesh(TriangleSet<TDataType>& ts, DArray<int>& indices);

		/**
		 * @brief Update the coordinates of the surface mesh, make surface indices are return by calling extractSurfaceMesh()
		 */
		void updateSurfaceMesh(TriangleSet<TDataType>& ts, DArray<int>& indices);

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

		// Mapping for tetrahedron mesh, automatically updated when update() is called
		DArray<::dyno::Topology::Tri2Tet> mTri2Tet;
		DArray<::dyno::Topology::Tet2Tri> mTet2Tri;
	};
}

