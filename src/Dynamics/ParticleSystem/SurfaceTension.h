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

		void setPositionID(FieldID id) { m_posID = id; }
		void setVelocityID(FieldID id) { m_velID = id; }
		void setNeighborhoodID(FieldID id) { m_neighborhoodID = id; }

		void setIntensity(Real intensity) { m_intensity = intensity; }
		void setSmoothingLength(Real len) { m_soothingLength = len; }

	protected:
		FieldID m_posID;
		FieldID m_velID;
		FieldID m_neighborhoodID;

	private:
		Real m_intensity;
		Real m_soothingLength;

		DeviceArrayField<Real>* m_energy;
	};

#ifdef PRECISION_FLOAT
	template class SurfaceTension<DataType3f>;
#else
	template class SurfaceTension<DataType3d>;
#endif
}