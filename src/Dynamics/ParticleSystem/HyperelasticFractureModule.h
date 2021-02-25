#pragma once
#include "HyperelastoplasticityModule.h"

namespace dyno {

	template<typename TDataType> class SummationDensity;

	template<typename TDataType>
	class HyperelasticFractureModule : public ConstraintModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef TPair<TDataType> NPair;

		HyperelasticFractureModule();
		~HyperelasticFractureModule() override {};

		bool constrain() override;

		void updateTopology();

	protected:

	public:
		DEF_VAR(CriticalStretch, Real, 1.02, "");

		DEF_EMPTY_IN_ARRAY(FractureTag, bool, DeviceType::GPU, "");

		DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "");

		DEF_EMPTY_IN_ARRAY(RestPosition, Coord, DeviceType::GPU, "");
	};

#ifdef PRECISION_FLOAT
	template class HyperelasticFractureModule<DataType3f>;
#else
	template class HyperelasticFractureModule<DataType3d>;
#endif
}