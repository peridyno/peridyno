#pragma once

#include "Platform.h"
#include "Array/Array.h"
#include "Array/Array3D.h"

#include "PhaseFieldKernels.h"

#include "Module/ComputeModule.h"

#include "Topology/PhaseField.h"

namespace dyno{
	template<typename TDataType>
	class TwoPhaseFlow : public ComputeModule
	{
		DECLARE_TCLASS(EulerianFluid3D, TDataType);
		
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		TwoPhaseFlow() {};
		~TwoPhaseFlow() override {};

		DEF_VAR_IN(Real, TimeStep, "");

		DEF_ARRAY3D_IN(Real, Mass, DeviceType::GPU, "");

		DEF_ARRAY3D_IN(Coord, Velocity, DeviceType::GPU, "");

		DEF_INSTANCE_IN(PhaseField<TDataType>, PhaseField, "");

	protected:
		void compute() override;

	private:
		uint mNx = 0;
		uint mNy = 0;
		uint mNz = 0;

		DArray3D<Real> vel_u;
		DArray3D<Real> vel_v;
		DArray3D<Real> vel_w;

		DArray3D<Real> pre_vel_u;
		DArray3D<Real> pre_vel_v;
		DArray3D<Real> pre_vel_w;

		DArray3D<Coord> velBuf;
		DArray3D<Coord> velSrc;

		DArray3D<Real> omega;

		DArray3D<Coord> dir;

		DArray3D<Coef> mat;
		DArray3D<Real> RHS;
		DArray3D<Real> pressure;

		//mass buffer
		DArray3D<Real> mb0;
		DArray3D<Real> mb1;
	};
}