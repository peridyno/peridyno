#pragma once
#include "UniformGrid.h"

namespace dyno
{
	template<typename TDataType>
	class PhaseField : public UniformGrid3D<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PhaseField();
		~PhaseField() override;

		void initialize(uint nx, uint ny, uint nz) override;

		DArray3D<Real>& volumeFraction() { return mVolumeFraction; }

	private:
		DArray3D<Real> mVolumeFraction;
	};
}


