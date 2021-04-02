#pragma once
#include "Framework/ModuleTopology.h"
#include "Vector.h"
#include "Array/Array2D.h"


namespace dyno
{
	template<typename TDataType>
	class HeightField : public TopologyModule
	{
		DECLARE_CLASS_1(PointSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		HeightField();
		~HeightField() override;

		void copyFrom(HeightField<TDataType>& pointSet);

		void scale(Real s);
		void scale(Coord s);
		void translate(Coord t);

		void setSpace(Real dx, Real dz);

		Real getDx() { return m_dx; }
		Real getDz() { return m_dz; }

		Coord getOrigin() { return origin; }
		

		DArray2D<Real>& getHeights() { return m_height; }

	protected:
// 		DArray<Coord> m_coords;
// 		DArray<Coord> m_normals;
// 		NeighborList<int> m_pointNeighbors;

		Coord origin;

		Real m_dx;
		Real m_dz;

		DArray2D<Real> m_height;
	};
}

