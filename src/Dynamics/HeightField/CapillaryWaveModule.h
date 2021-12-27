#pragma once
#include "Module/ConstraintModule.h"
#include "Peridynamics/NeighborData.h"
#include "types.h"
#include "Module/ComputeModule.h"
#include "DeclarePort.h"
namespace dyno {

	/**
	  * @brief This is an implementation of elasticity based on projective peridynamics.
	  *		   For more details, please refer to[He et al. 2017] "Projective Peridynamics for Modeling Versatile Elastoplastic Materials"
	  */
	template<typename TDataType>
	class CapillaryWaveModule : public ComputeModule
	{
		DECLARE_CLASS_1(CapillaryWaveModule, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename Vector<Real, 2> Coord2D;
		typedef typename Vector<Real, 4> Coord4D;

		CapillaryWaveModule();
		~CapillaryWaveModule() override {};

		void compute() override;

		DEF_ARRAY2D_STATE(Coord2D, Position, DeviceType::GPU, "Height field velocity");
		
	};
}