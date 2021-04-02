#include "SolidFluidInteraction.h"
#include "PositionBasedFluidModel.h"

#include "Topology/PointSet.h"
#include "ParticleSystem.h"
#include "Topology/NeighborQuery.h"
#include "Kernel.h"
#include "DensityPBD.h"
#include "ImplicitViscosity.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(SolidFluidInteraction, TDataType)

	template<typename TDataType>
	SolidFluidInteraction<TDataType>::SolidFluidInteraction(std::string name)
		:Node(name)
	{
		this->attachField(&radius, "radius", "radius");
		radius.setValue(0.0075);

		m_nbrQuery = this->template addComputeModule<NeighborQuery<TDataType>>("neighborhood");
		radius.connect(m_nbrQuery->inRadius());
		m_position.connect(m_nbrQuery->inPosition());

		m_pbdModule = this->template addConstraintModule<DensityPBD<TDataType>>("collision");
		radius.connect(m_pbdModule->varSmoothingLength());
		m_position.connect(m_pbdModule->inPosition());
		m_vels.connect(m_pbdModule->inVelocity());
		m_nbrQuery->outNeighborhood()->connect(m_pbdModule->inNeighborIndex());
		m_pbdModule->varIterationNumber()->setValue(5);
	}


	template<typename TDataType>
	void SolidFluidInteraction<TDataType>::setInteractionDistance(Real d)
	{
		radius.setValue(d);
		m_pbdModule->varSamplingDistance()->setValue(d / 2);
	}

	template<typename TDataType>
	SolidFluidInteraction<TDataType>::~SolidFluidInteraction()
	{
		
	}

	template<typename TDataType>
	bool SolidFluidInteraction<TDataType>::initialize()
	{
		return true;
	}

	template<typename TDataType>
	bool SolidFluidInteraction<TDataType>::addRigidBody(std::shared_ptr<RigidBody<TDataType>> child)
	{
		return false;
	}

	template<typename TDataType>
	bool SolidFluidInteraction<TDataType>::addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child)
	{
		this->addChild(child);
		m_particleSystems.push_back(child);

		return false;
	}


	template<typename TDataType>
	bool SolidFluidInteraction<TDataType>::resetStatus()
	{
		int total_num = 0;
		std::vector<int> ids;
		std::vector<Real> mass;
		for (int i = 0; i < m_particleSystems.size(); i++)
		{
			auto points = m_particleSystems[i]->currentPosition()->getValue();
			total_num += points.size();
			Real m = m_particleSystems[i]->getMass() / points.size();
			for (int j = 0; j < points.size(); j++)
			{
				ids.push_back(i);
				mass.push_back(m);
			}
		}

		m_objId.resize(total_num);
		m_vels.setElementCount(total_num);
		m_mass.setElementCount(total_num);
		m_position.setElementCount(total_num);

		posBuf.resize(total_num);
		weights.resize(total_num);
		init_pos.resize(total_num);

		m_objId.assign(ids);
		m_mass.getValue().assign(mass);
		ids.clear();
		mass.clear();

		int start = 0;
		DArray<Coord>& allpoints = m_position.getValue();
		for (int i = 0; i < m_particleSystems.size(); i++)
		{
			DArray<Coord>& points = m_particleSystems[i]->currentPosition()->getValue();
			DArray<Coord>& vels = m_particleSystems[i]->currentVelocity()->getValue();
			int num = points.size();
			cudaMemcpy(allpoints.begin() + start, points.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			cudaMemcpy(m_vels.getValue().begin() + start, vels.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			start += num;
		}

		return true;
	}

	template<typename Real, typename Coord>
	__global__ void K_Collide(
		DArray<int> objIds,
		DArray<Real> mass,
		DArray<Coord> points,
		DArray<Coord> newPoints,
		DArray<Real> weights,
		NeighborList<int> neighbors,
		Real radius
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;

		SpikyKernel<Real> kernel;

		Real r;
		Coord pos_i = points[pId];
		int id_i = objIds[pId];
		Real mass_i = mass[pId];
		int nbSize = neighbors.getNeighborSize(pId);
		int col_num = 0;
		Coord pos_num = Coord(0);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Coord pos_j = points[j];

			r = (pos_i - pos_j).norm();
			if (r < radius && objIds[j] != id_i)
			{
				col_num++;
				Real mass_j = mass[j];
				Coord center = (pos_i + pos_j) / 2;
				Coord n = pos_i - pos_j;
				n = n.norm() < EPSILON ? Coord(0, 0, 0) : n.normalize();

				Real a = mass_i / (mass_i + mass_j);

				Real d = radius - r;

				Coord target_i = pos_i + (1 - a)*d*n;// (center + 0.5*radius*n);
				Coord target_j = pos_j - a*d*n;// (center - 0.5*radius*n);
				//				pos_num += (center + 0.4*radius*n);

				Real weight = kernel.Weight(r, 2 * radius);

				atomicAdd(&newPoints[pId][0], weight*target_i[0]);
				atomicAdd(&newPoints[j][0], weight*target_j[0]);

				atomicAdd(&weights[pId], weight);
				atomicAdd(&weights[j], weight);

				if (Coord::dims() >= 2)
				{
					atomicAdd(&newPoints[pId][1], weight*target_i[1]);
					atomicAdd(&newPoints[j][1], weight*target_j[1]);
				}

				if (Coord::dims() >= 3)
				{
					atomicAdd(&newPoints[pId][2], weight*target_i[2]);
					atomicAdd(&newPoints[j][2], weight*target_j[2]);
				}
			}
		}

		//		if (col_num != 0)
		//			pos_num /= col_num;
		//		else
		//			pos_num = pos_i;
		//
		//		newPoints[pId] = pos_num;
	}

	template<typename Real, typename Coord>
	__global__ void K_ComputeTarget(
		DArray<Coord> oldPoints,
		DArray<Coord> newPoints,
		DArray<Real> weights)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= oldPoints.size()) return;

		if (weights[pId] > EPSILON)
		{
			newPoints[pId] /= weights[pId];
		}
		else
			newPoints[pId] = oldPoints[pId];
	}

	template<typename Real, typename Coord>
	__global__ void K_ComputeVelocity(
		DArray<Coord> initPoints,
		DArray<Coord> curPoints,
		DArray<Coord> velocites,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velocites.size()) return;

		velocites[pId] += 0.5*(curPoints[pId] - initPoints[pId]) / dt;
	}

	template<typename TDataType>
	void SolidFluidInteraction<TDataType>::advance(Real dt)
	{
		int start = 0;
		DArray<Coord>& allpoints = m_position.getValue();
		for (int i = 0; i < m_particleSystems.size(); i++)
		{
			DArray<Coord>& points = m_particleSystems[i]->currentPosition()->getValue();
			DArray<Coord>& vels = m_particleSystems[i]->currentVelocity()->getValue();
			int num = points.size();
			cudaMemcpy(allpoints.begin() + start, points.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			cudaMemcpy(m_vels.getValue().begin() + start, vels.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			start += num;
		}

		m_nbrQuery->compute();

		auto module = this->template getModule<DensityPBD<TDataType>>("collision");
		module->constrain();

// 		auto module2 = this->template getModule<ImplicitViscosity<TDataType>>("viscosity");
// 		module2->constrain();

/*		Function1Pt::copy(init_pos, allpoints);

		uint pDims = cudaGridSize(allpoints.size(), BLOCK_SIZE);
		for (size_t it = 0; it < 5; it++)
		{
			weights.reset();
			posBuf.reset();
			K_Collide << <pDims, BLOCK_SIZE >> > (
				m_objId, 
				m_mass.getValue(),
				allpoints,
				posBuf, 
				weights, 
				m_nbrQuery->getNeighborList(),
				radius.getValue());

			K_ComputeTarget << <pDims, BLOCK_SIZE >> > (
				allpoints,
				posBuf, 
				weights);

			Function1Pt::copy(allpoints, posBuf);
		}

		K_ComputeVelocity << <pDims, BLOCK_SIZE >> > (init_pos, allpoints, m_vels.getValue(), getParent()->getDt());*/

		start = 0;
		for (int i = 0; i < m_particleSystems.size(); i++)
		{
			DArray<Coord>& points = m_particleSystems[i]->currentPosition()->getValue();
			DArray<Coord>& vels = m_particleSystems[i]->currentVelocity()->getValue();
			int num = points.size();
			cudaMemcpy(points.begin(), allpoints.begin() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			cudaMemcpy(vels.begin(), m_vels.getValue().begin() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);

			start += num;
		}

	}

	DEFINE_CLASS(SolidFluidInteraction);
}