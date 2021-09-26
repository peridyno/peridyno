#pragma once
#include "Module/ForceModule.h"

namespace dyno {

	template<typename TDataType>
	class SurfaceTension : public ForceModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SurfaceTension();
		~SurfaceTension() override {};
		
		void updateImpl() override;

		bool applyForce() override;

		void setIntensity(Real intensity) { m_intensity = intensity; }
		void setSmoothingLength(Real len) { m_soothingLength = len; }

	private:
		Real m_intensity;
		Real m_soothingLength;
	};
}