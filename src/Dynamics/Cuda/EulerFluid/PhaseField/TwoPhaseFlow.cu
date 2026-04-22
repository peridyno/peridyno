#include "TwoPhaseFlow.h"
#include "Algorithm/Reduction.h"

namespace dyno{

	IMPLEMENT_TCLASS(TwoPhaseFlow, TDataType)

	template<typename TDataType>
	void TwoPhaseFlow<TDataType>::compute()
	{
		auto pf = this->inPhaseField()->getDataPtr();
		auto& mass = this->inMass()->getData();
		auto& vel = this->inVelocity()->getData();

		auto dt = this->inTimeStep()->getValue();
		auto h = pf->gridSpacing();

		if (mNx != mass.nx() || mNy != mass.ny() || mNz != mass.nz())
		{
			mNx = mass.nx();
			mNy = mass.ny();
			mNz = mass.nz();

			omega.resize(mNx, mNy, mNz);

			vel_u.resize(mNx + 1, mNy, mNz);
			vel_v.resize(mNx, mNy + 1, mNz);
			vel_w.resize(mNx, mNy, mNz + 1);

			pre_vel_u.resize(mNx + 1, mNy, mNz);
			pre_vel_v.resize(mNx, mNy + 1, mNz);
			pre_vel_w.resize(mNx, mNy, mNz + 1);

			mat.resize(mNx, mNy, mNz);
			RHS.resize(mNx, mNy, mNz);
			pressure.resize(mNx, mNy, mNz);
			pressure.reset();

			dir.resize(mNx, mNy, mNz);

			mb0.resize(mNx, mNy, mNz);
			mb1.resize(mNx, mNy, mNz);

			velBuf.resize(mNx, mNy, mNz);
			velSrc.resize(mNx, mNy, mNz);

			vel_u.reset();
			vel_v.reset();
			vel_w.reset();

			CArray3D<Real> host_omega(mNx, mNy, mNz);
			for (int i = 0; i < mNx; i++)
				for (int j = 0; j < mNy; j++)
				{
					for (int k = 0; k < mNz; k++)
					{
						host_omega(i, j, k) = 1;
					}
				}

			omega.assign(host_omega);
			host_omega.clear();
		}

		Real gamma = 0.005;

		PhaseFieldKernels<TDataType>::ApplyGravity(vel, Coord(0, -0.5, 0.0), dt);

		PhaseFieldKernels<TDataType>::InterpolateVelocity(vel_u, vel_v, vel_w, vel);

		PhaseFieldKernels<TDataType>::SetU(vel_u);
		PhaseFieldKernels<TDataType>::SetV(vel_v);
		PhaseFieldKernels<TDataType>::SetW(vel_w);

		PhaseFieldKernels<TDataType>::PrepareForProjection(vel_u, vel_v, vel_w, mat, RHS, mass, h, dt);
		for (int i = 0; i < 20; i++)
		{
			PhaseFieldKernels<TDataType>::Projection(pressure, mb0, mat, RHS, 1);
		}
		PhaseFieldKernels<TDataType>::UpdateVelocity(vel_u, vel_v, vel_w, pressure, mass, h, dt);
		// 
		PhaseFieldKernels<TDataType>::InterpolateVelocity(vel, vel_u, vel_v, vel_w);

		//Advect
		mb0.assign(mass);
		PhaseFieldKernels<TDataType>::AdvectForward(mass, mb0, vel, dt);

		velBuf.assign(vel);
		velSrc.assign(vel);
		PhaseFieldKernels<TDataType>::AdvectBackward(vel, velBuf, velSrc, dt);

		//Sharpening
		mb0.assign(mass);
		PhaseFieldKernels<TDataType>::Sharpening(mass, dir, mb0, vel_u, vel_v, vel_w, omega, gamma, h, dt);

		mb0.assign(mass);
		float a = 1.0f * gamma / h / h * dt;// dt;
		PhaseFieldKernels<TDataType>::Jacobi(mass, mb0, mb1, vel, a, 1.0f + 6.0f * a, 10);
	}

	DEFINE_CLASS(TwoPhaseFlow);
}