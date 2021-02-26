#include "CahnHilliard.h"
#include "Framework/Node.h"
#include "Kernel.h"

// Implement paper: Fast Multiple-fluid Simulation Using Helmholtz Free Energy

namespace dyno
{
    template<typename TDataType, int PhaseCount>
    CahnHilliard<TDataType, PhaseCount>::CahnHilliard() 
	{
        m_particleVolume.setValue(Real(1e-3));
        m_degenerateMobilityM.setValue(Real(1e-4));
        m_interfaceEpsilon.setValue(Real(1e-3));
    }


    template<typename TDataType, int PhaseCount>
    CahnHilliard<TDataType, PhaseCount>::~CahnHilliard()
	{
	}


    template<typename TDataType, int PhaseCount>
    bool CahnHilliard<TDataType, PhaseCount>::initializeImpl() 
	{
        m_chemicalPotential.setElementCount(m_position.getElementCount());
        return true;
    }

    template<typename TDataType> 
    struct HelmholtzEnergyFunction 
	{
        using Real = typename TDataType::Real;
        using Coord = typename TDataType::Coord;
        using PhaseVector = typename CahnHilliard<TDataType>::PhaseVector;
        // equation 21 in the paper
        Real alpha, s1, s2;

        __host__ __device__
        Real operator()(PhaseVector p) {
            Real d1 = p[0] - s1;
            Real d2 = p[1] - s2;
            return alpha * d1 * d1 * d2 * d2;
        }

        __host__ __device__
        PhaseVector derivative(PhaseVector p) {
            Real d1 = p[0] - s1;
            Real d2 = p[1] - s2;
            PhaseVector r(2 * d1 * d2 * d2, 2 * d2 * d1 * d1);
            return alpha * r;
        }
    };


    template<typename TDataType,
    typename Real=typename TDataType::Real,
    typename Coord=typename TDataType::Coord,
    typename PhaseVector=typename CahnHilliard<TDataType>::PhaseVector>
    __global__ void calcChemicalPotential(
		GArray<Coord> posArr,
        GArray<PhaseVector> cArr,
        GArray<PhaseVector> muArr,
		NeighborList<int> neighbors,
        Real smoothingLength,
        Real particleVolume,
        Real epsilon)
	{
        const Real eta2 = smoothingLength * smoothingLength * Real(0.01);
        
        int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

        Coord pos_i = posArr[pId];
        PhaseVector c_i = cArr[pId];

		SpikyKernel<Real> kern;

        // equation 26
		PhaseVector lap_c_i(0);
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
            int j = neighbors.getElement(pId, ne);
            Coord r_ij = pos_i - posArr[j];
            Real r = r_ij.norm();
			if (r > EPSILON)
			{
                PhaseVector c_ij = c_i - cArr[j];
				lap_c_i += c_ij * (kern.Gradient(r, smoothingLength) * r / (r * r + eta2));
			}
        }
        lap_c_i *= 2 * particleVolume;
        //equation 17
        HelmholtzEnergyFunction<TDataType> F{/*alpha*/1, /*s1*/0, /*s2*/0};
        PhaseVector mu = F.derivative(c_i);
        Real sum = 0; for(int i = 0; i < mu.dims(); i++) sum += mu[i];
        mu -= sum/mu.dims();
        mu -= epsilon * epsilon * lap_c_i;
        muArr[pId] = mu;
    }

    template<typename TDataType,
    typename Real=typename TDataType::Real,
    typename Coord=typename TDataType::Coord,
    typename PhaseVector=typename CahnHilliard<TDataType>::PhaseVector>
    __global__ void updateConcentration(
		GArray<Coord> posArr,
        GArray<PhaseVector> cArr,
        GArray<PhaseVector> muArr,
		NeighborList<int> neighbors,
        Real smoothingLength,
        Real particleVolume,
        Real M,
        Real dt) 
	{
        const Real eta2 = smoothingLength * smoothingLength * Real(0.01);
        
        int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

        Coord pos_i = posArr[pId];
        PhaseVector mu_i = muArr[pId];

        SpikyKernel<Real> kern;

        // equation 25
		PhaseVector rhs_i(0);
        int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
            int j = neighbors.getElement(pId, ne);
            Coord r_ij = pos_i - posArr[j];
            Real r = r_ij.norm();
            if (r > EPSILON)
            {
                PhaseVector mu_ij = mu_i - muArr[j];
				rhs_i += mu_ij * kern.Gradient(r, smoothingLength)* r / (r * r + eta2);
			}
        }
        rhs_i *= 2 * M * particleVolume;
        // equation 16
        PhaseVector c_i = cArr[pId] + rhs_i * dt;
        // concentration correction
        for(int k = 0; k < c_i.dims(); k++)
            if(c_i[k] < 0) c_i[k] = 0;
        Real sum = 0; for(int i = 0; i < c_i.dims(); i++) sum += c_i[i];
        c_i /= sum;
        cArr[pId] = c_i;
    }


    template<typename TDataType, int PhaseCount>
    bool CahnHilliard<TDataType, PhaseCount>::integrate() 
	{
        Real dt = getParent()->getDt();

        int num = m_position.getElementCount();

		uint pDims = cudaGridSize(num, BLOCK_SIZE);
        calcChemicalPotential<TDataType><<<pDims, BLOCK_SIZE>>>(
            m_position.getValue(),
            m_concentration.getValue(),
            m_chemicalPotential.getValue(),
            m_neighborhood.getValue(),
            m_smoothingLength.getValue(),
            m_particleVolume.getValue(),
            m_interfaceEpsilon.getValue()
        );
        updateConcentration<TDataType><<<pDims, BLOCK_SIZE>>>(
            m_position.getValue(),
            m_concentration.getValue(),
            m_chemicalPotential.getValue(),
            m_neighborhood.getValue(),
            m_smoothingLength.getValue(),
            m_particleVolume.getValue(),
            m_degenerateMobilityM.getValue(),
            dt
        );
        return true;
    }
}
