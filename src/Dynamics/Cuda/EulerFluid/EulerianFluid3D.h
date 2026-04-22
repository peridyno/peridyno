#pragma once
#include "Node.h"

#include "Topology/PhaseField.h"

#include "PhaseField/PhaseFieldKernels.h"

namespace dyno
{
	template<typename TDataType>
	class EulerianFluid3D : public Node
	{
		DECLARE_TCLASS(EulerianFluid3D, TDataType);
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		EulerianFluid3D();
		~EulerianFluid3D() override;

		DEF_VAR(Vec3i, Dimension, Vec3i(64), "");

	public:
		DEF_ARRAY3D_STATE(Real, Mass, DeviceType::GPU, "");

		DEF_ARRAY3D_STATE(Coord, Velocity, DeviceType::GPU, "");
		
		DEF_INSTANCE_STATE(PhaseField<TDataType>, PhaseField, "");

	protected:
		void resetStates() override;
		void postUpdateStates() override;

	private:
		
	};
}