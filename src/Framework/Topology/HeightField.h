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
		

		GArray2D<Real>& getHeights() { return m_height; }

	protected:
// 		GArray<Coord> m_coords;
// 		GArray<Coord> m_normals;
// 		NeighborList<int> m_pointNeighbors;

		Coord origin;

		Real m_dx;
		Real m_dz;

		GArray2D<Real> m_height;
	};


#ifdef PRECISION_FLOAT
	template class HeightField<DataType3f>;
#else
	template class HeightField<DataType3d>;
#endif
}

