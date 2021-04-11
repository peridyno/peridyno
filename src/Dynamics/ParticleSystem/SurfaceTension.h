#pragma once
#include "Framework/ModuleForce.h"

namespace dyno {

	template<typename TDataType>
	class SurfaceTension : public ForceModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SurfaceTension();
		~SurfaceTension() override {};
		
		bool execute() override;

		bool applyForce() override;

		void setIntensity(Real intensity) { m_intensity = intensity; }
		void setSmoothingLength(Real len) { m_soothingLength = len; }

	private:
		Real m_intensity;
		Real m_soothingLength;
	};
}