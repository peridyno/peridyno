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
		void setTetrahedrons(DeviceArray<Tetrahedron>& tetrahedrons);

		DeviceArray<Tetrahedron>& getTetrahedrons() { return m_tethedrons; }
		DeviceArray<Tri2Tet>& getTri2Tet() { return tri2Tet; }

		NeighborList<int>& getVer2Tet();

		void getVolume(DeviceArray<Real>& volume);

		void updateTriangles();

		void copyFrom(TetrahedronSet<TDataType> tetSet);

	protected:
		bool initializeImpl() override;

	protected:
		DeviceArray<Tetrahedron> m_tethedrons;

	private:
		DeviceArray<Tri2Tet> tri2Tet;
		NeighborList<int> m_ver2Tet;
	};

#ifdef PRECISION_FLOAT
	template class TetrahedronSet<DataType3f>;
#else
	template class TetrahedronSet<DataType3d>;
#endif
}

