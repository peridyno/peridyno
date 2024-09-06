#pragma once
#include "ElastoplasticityModule.h"

namespace dyno {

	template<typename TDataType> class SummationDensity;

	template<typename TDataType>
	class GranularModule : public ElastoplasticityModule<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TBond<TDataType> Bond;

		GranularModule();
		~GranularModule() override {};

	protected:
		void computeMaterialStiffness() override;

	private:
		std::shared_ptr<SummationDensity<TDataType>> m_densitySum;
	};
}