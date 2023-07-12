#pragma once
#include "Module/TopologyModule.h"

#include "Array/Array2D.h"

namespace dyno
{
	template<typename TDataType>
	class HeightField : public TopologyModule
	{
		DECLARE_TCLASS(PointSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		HeightField();
		~HeightField() override;

		void copyFrom(HeightField<TDataType>& hf);

		void scale(Real s);
		void scale(Coord s);
		void translate(Coord t);

//		void setSpace(Real dx, Real dz);

		void setExtents(uint nx, uint ny);

// 		Real getDx() { return mDx; }
// 		Real getDz() { return mDz; }

		Real getGridSpacing() { return mGridSpacing; }
		void setGridSpacing(Real h) { mGridSpacing = h; }

		Coord getOrigin() { return mOrigin; }
		void setOrigin(Coord p) { mOrigin = p; }

		uint width();
		uint height();

		DArray2D<Coord>& getDisplacement() { return mDisplacement; }

	protected:
		Coord mOrigin;

// 		Real mDx;
// 		Real mDz;

		Real mGridSpacing;

		DArray2D<Coord> mDisplacement;
	};
}

