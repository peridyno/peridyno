#pragma once
#include "Framework/ModuleConstraint.h"

namespace dyno 
{
	template<typename TDataType>
	class ImplicitViscosity : public ConstraintModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ImplicitViscosity();
		~ImplicitViscosity() override;
		
		bool constrain() override;

		void setIterationNumber(int n);

		void setViscosity(Real mu);


	protected:
		bool initializeImpl() override;

	public:
		VarField<Real> m_viscosity;
		VarField<Real> m_smoothingLength;

		DeviceArrayField<Coord> m_velocity;
		DeviceArrayField<Coord> m_position;

		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "");

	private:
		int m_maxInteration;

		DArray<Coord> m_velOld;
		DArray<Coord> m_velBuf;

		
	};
}