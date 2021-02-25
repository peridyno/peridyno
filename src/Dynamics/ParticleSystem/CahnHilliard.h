/**
 * @file CahnHilliard.h
 * @author Chen Xiaosong
 * @brief
 * @version 0.1
 * @date 2019-06-18
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once
#include "Framework/Module.h"

namespace dyno
{
    template<typename TDataType, int PhaseCount = 2>
	class CahnHilliard : public Module
    {
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
        using PhaseVector = Vector<Real, PhaseCount>;

		CahnHilliard();
		~CahnHilliard() override;

		bool initializeImpl() override;

		bool integrate();

		VarField<Real> m_particleVolume;
		VarField<Real> m_smoothingLength;

		VarField<Real> m_degenerateMobilityM;
		VarField<Real> m_interfaceEpsilon;

        DeviceArrayField<Coord> m_position;

		NeighborField<int> m_neighborhood;

		DeviceArrayField<PhaseVector> m_chemicalPotential;
        DeviceArrayField<PhaseVector> m_concentration;
	};
#ifdef PRECISION_FLOAT
	template class CahnHilliard<DataType3f>;
#else
	template class CahnHilliard<DataType3d>;
#endif
}

