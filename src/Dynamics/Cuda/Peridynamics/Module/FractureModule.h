#pragma once
#include "ElastoplasticityModule.h"

namespace dyno {

	template<typename TDataType> class SummationDensity;

	template<typename TDataType>
	class FractureModule : public ElastoplasticityModule<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TBond<TDataType> Bond;

		FractureModule();
		~FractureModule() override {};

		void applyPlasticity() override;
	};
}