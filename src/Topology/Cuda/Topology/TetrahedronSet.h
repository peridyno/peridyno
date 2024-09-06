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
		~TetrahedronSet();

		void loadTetFile(std::string filename);

		void setTetrahedrons(std::vector<Tetrahedron>& tetrahedrons);
		void setTetrahedrons(DArray<Tetrahedron>& tetrahedrons);

		DArray<Tetrahedron>& getTetrahedrons() { return mTethedrons; }
		DArray<::dyno::TopologyModule::Tri2Tet>& getTri2Tet() { return mTri2Tet; }

		DArrayList<int>& getVer2Tet();

		void getVolume(DArray<Real>& volume);

		void copyFrom(TetrahedronSet<TDataType>& tetSet);

		bool isEmpty() override;

	protected:
		void updateTriangles() override;

	private:
		DArray<Tetrahedron> mTethedrons;

		DArray<::dyno::TopologyModule::Tri2Tet> mTri2Tet;

		DArrayList<int> mVer2Tet;
	};
}

