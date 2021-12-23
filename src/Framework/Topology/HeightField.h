#pragma once
#include "Module/TopologyModule.h"
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

		Real getDx() { return mDx; }
		Real getDz() { return mDz; }

		Coord getOrigin() { return mOrigin; }

		uint length();
		uint width();
		
		DArray2D<Real>& getHeights() { return m_height; }

	protected:
		Coord mOrigin;

		Real mDx;
		Real mDz;

		DArray2D<Real> m_height;
	};
}

