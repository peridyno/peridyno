#pragma once
#include "TriangleSet.h"
#include "Framework/ModuleTopology.h"


namespace dyno
{
	template<typename TDataType>
	class TetrahedronSet : public TriangleSet<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Tetrahedron Tetrahedron;

		TetrahedronSet();
		~TetrahedronSet();

		void loadTetFile(std::string filename);

		void setTetrahedrons(std::vector<Tetrahedron>& tetrahedrons);
		void setTetrahedrons(DArray<Tetrahedron>& tetrahedrons);

		DArray<Tetrahedron>& getTetrahedrons() { return m_tethedrons; }
		DArray<Tri2Tet>& getTri2Tet() { return tri2Tet; }

		NeighborList<int>& getVer2Tet();

		void getVolume(DArray<Real>& volume);

		void updateTriangles();

		void copyFrom(TetrahedronSet<TDataType> tetSet);

	protected:
		bool initializeImpl() override;

	protected:
		DArray<Tetrahedron> m_tethedrons;

	private:
		DArray<Tri2Tet> tri2Tet;
		NeighborList<int> m_ver2Tet;
	};

#ifdef PRECISION_FLOAT
	template class TetrahedronSet<DataType3f>;
#else
	template class TetrahedronSet<DataType3d>;
#endif
}

