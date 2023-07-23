#pragma once
#include "Module/ComputeModule.h"

namespace dyno {

	template<typename TDataType>
	class SurfaceTension : public ComputeModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SurfaceTension();
		~SurfaceTension() override {};
		

		void setIntensity(Real intensity) { m_intensity = intensity; }
		void setSmoothingLength(Real len) { m_soothingLength = len; }

	private:
		void compute() override;

		Real m_intensity;
		Real m_soothingLength;
	};
}