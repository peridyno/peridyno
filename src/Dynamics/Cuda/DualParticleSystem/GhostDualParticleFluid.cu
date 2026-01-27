#include "GhostDualParticleFluid.h"

// DataType
#include "Auxiliary/DataSource.h"

// Collision
#include "Collision/NeighborPointQuery.h"

// ParticleSystem
#include "ParticleSystem/Module/ImplicitViscosity.h"
#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/SemiImplicitDensitySolver.h"
#include "ParticleSystem/Module/VariationalApproximateProjection.h"

// DualParticleSystem
#include "Module/DualParticleIsphModule.h"
#include <DualParticleSystem/Module/ThinFeature.h>

namespace dyno
{
	template <typename TDataType>
	GhostDualParticleFluid<TDataType>::GhostDualParticleFluid()
		: GhostDualParticleFluid(2)
	{
	}

	template <typename TDataType>
	GhostDualParticleFluid<TDataType>::GhostDualParticleFluid(int key)
		: DualParticleFluid<TDataType>::DualParticleFluid(key)
	{
		this->animationPipeline()->clear();

		this->varVirtualParticleSamplingStrategy()->getDataPtr()->setCurrentKey(key);
		this->varReshuffleParticles()->setValue(false);
		this->varSmoothingLength()->setValue(2.4);

		auto m_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->stateSmoothingLength()->connect(m_nbrQuery->inRadius());
		this->statePositionMerged()->connect(m_nbrQuery->inPosition());
		this->animationPipeline()->pushModule(m_nbrQuery);

		if (key == DualParticleFluid<TDataType>::SpatiallyAdaptiveStrategy)
		{
			auto m_adaptive_virtual_position = std::make_shared<VirtualSpatiallyAdaptiveStrategy<TDataType>>();
			this->statePositionMerged()->connect(m_adaptive_virtual_position->inRPosition());
			m_adaptive_virtual_position->varSamplingDistance()->setValue(Real(0.005)); /**Virtual particle radius*/
			m_adaptive_virtual_position->varCandidatePointCount()->getDataPtr()->setCurrentKey(VirtualSpatiallyAdaptiveStrategy<TDataType>::neighbors_33);
			this->vpGen = m_adaptive_virtual_position;
		}
		else if (key == DualParticleFluid<TDataType>::ParticleShiftingStrategy)
		{
			auto m_virtual_particle_shifting = std::make_shared<VirtualParticleShiftingStrategy<TDataType>>();
			this->stateFrameNumber()->connect(m_virtual_particle_shifting->inFrameNumber());
			this->statePositionMerged()->connect(m_virtual_particle_shifting->inRPosition());
			this->stateTimeStep()->connect(m_virtual_particle_shifting->inTimeStep());
			this->animationPipeline()->pushModule(m_virtual_particle_shifting);
			this->vpGen = m_virtual_particle_shifting;
		}
		else if (key == DualParticleFluid<TDataType>::ColocationStrategy)
		{
			auto m_virtual_equal_to_Real = std::make_shared<VirtualColocationStrategy<TDataType>>();
			this->statePositionMerged()->connect(m_virtual_equal_to_Real->inRPosition());
			this->animationPipeline()->pushModule(m_virtual_equal_to_Real);
			this->vpGen = m_virtual_equal_to_Real;
		}
		else if (key == DualParticleFluid<TDataType>::FissionFusionStrategy)
		{
			auto feature = std::make_shared<ThinFeature<TDataType>>();
			this->statePositionMerged()->connect(feature->inPosition());
			m_nbrQuery->outNeighborIds()->connect(feature->inNeighborIds());
			this->stateSmoothingLength()->connect(feature->inSmoothingLength());
			this->stateSamplingDistance()->connect(feature->inSamplingDistance());
			feature->varThreshold()->setValue(0.05f);
			this->animationPipeline()->pushModule(feature);

			auto gridFission = std::make_shared<VirtualFissionFusionStrategy<TDataType>>();
			gridFission->varTransitionRegionThreshold()->setValue(0.01);
			feature->outThinSheet()->connect(gridFission->inThinSheet());
			feature->outThinFeature()->connect(gridFission->inThinFeature());
			this->statePositionMerged()->connect(gridFission->inRPosition());
			this->stateVelocityMerged()->connect(gridFission->inRVelocity());
			m_nbrQuery->outNeighborIds()->connect(gridFission->inNeighborIds());
			this->stateSmoothingLength()->connect(gridFission->inSmoothingLength());
			this->stateSamplingDistance()->connect(gridFission->inSamplingDistance());
			this->stateFrameNumber()->connect(gridFission->inFrameNumber());
			this->stateTimeStep()->connect(gridFission->inTimeStep());
			this->animationPipeline()->pushModule(gridFission);
			gridFission->varMinDist()->setValue(0.002);
			this->vpGen = gridFission;
		}

		this->animationPipeline()->pushModule(this->vpGen);
		this->vpGen->outVirtualParticles()->connect(this->stateVirtualPosition());

		auto m_rv_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->stateSmoothingLength()->connect(m_rv_nbrQuery->inRadius());
		this->statePositionMerged()->connect(m_rv_nbrQuery->inOther());
		this->vpGen->outVirtualParticles()->connect(m_rv_nbrQuery->inPosition());
		this->animationPipeline()->pushModule(m_rv_nbrQuery);

		auto m_vr_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->stateSmoothingLength()->connect(m_vr_nbrQuery->inRadius());
		this->statePositionMerged()->connect(m_vr_nbrQuery->inPosition());
		this->vpGen->outVirtualParticles()->connect(m_vr_nbrQuery->inOther());
		this->animationPipeline()->pushModule(m_vr_nbrQuery);

		auto m_vv_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->stateSmoothingLength()->connect(m_vv_nbrQuery->inRadius());
		this->vpGen->outVirtualParticles()->connect(m_vv_nbrQuery->inPosition());
		this->animationPipeline()->pushModule(m_vv_nbrQuery);

		auto m_dualIsph = std::make_shared<DualParticleIsphModule<TDataType>>();
		this->stateSmoothingLength()->connect(m_dualIsph->varSmoothingLength());
		this->stateTimeStep()->connect(m_dualIsph->inTimeStep());
		this->statePositionMerged()->connect(m_dualIsph->inRPosition());
		this->vpGen->outVirtualParticles()->connect(m_dualIsph->inVPosition());
		this->stateVelocityMerged()->connect(m_dualIsph->inVelocity());
		this->stateAttributeMerged()->connect(m_dualIsph->inParticleAttribute());
		this->stateNormalMerged()->connect(m_dualIsph->inBoundaryNorm());
		m_nbrQuery->outNeighborIds()->connect(m_dualIsph->inNeighborIds());
		m_rv_nbrQuery->outNeighborIds()->connect(m_dualIsph->inRVNeighborIds());
		m_vr_nbrQuery->outNeighborIds()->connect(m_dualIsph->inVRNeighborIds());
		m_vv_nbrQuery->outNeighborIds()->connect(m_dualIsph->inVVNeighborIds());
		this->stateTimeStep()->connect(m_dualIsph->inTimeStep());
		this->animationPipeline()->pushModule(m_dualIsph);

		auto m_integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->stateTimeStep()->connect(m_integrator->inTimeStep());
		this->statePositionMerged()->connect(m_integrator->inPosition());
		this->stateVelocityMerged()->connect(m_integrator->inVelocity());
		this->stateAttributeMerged()->connect(m_integrator->inAttribute());
		this->animationPipeline()->pushModule(m_integrator);

		auto m_visModule = std::make_shared<ImplicitViscosity<TDataType>>();
		m_visModule->varViscosity()->setValue(Real(0.5));
		this->stateTimeStep()->connect(m_visModule->inTimeStep());
		this->stateSamplingDistance()->connect(m_visModule->inSamplingDistance());
		this->stateSmoothingLength()->connect(m_visModule->inSmoothingLength());
		this->stateTimeStep()->connect(m_visModule->inTimeStep());
		this->statePositionMerged()->connect(m_visModule->inPosition());
		this->stateVelocityMerged()->connect(m_visModule->inVelocity());
		m_nbrQuery->outNeighborIds()->connect(m_visModule->inNeighborIds());
		this->animationPipeline()->pushModule(m_visModule);
	}

	template <typename TDataType>
	GhostDualParticleFluid<TDataType>::~GhostDualParticleFluid()
	{
	}

	__global__ void GDPF_SetupFluidAttributes(
		DArray<Attribute> allAttributes,
		int num)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= num)
			return;

		allAttributes[pId].setDynamic();
		allAttributes[pId].setFluid();
	}

	__global__ void GDPF_SetupBoundaryAttributes(
		DArray<Attribute> allAttributes,
		DArray<Attribute> boundaryAttributes,
		int offset)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= boundaryAttributes.size())
			return;

		allAttributes[offset + pId].setFixed();
		allAttributes[offset + pId].setRigid();
	}

	template <typename TDataType>
	void GhostDualParticleFluid<TDataType>::resetStates()
	{
		this->DualParticleFluid<TDataType>::resetStates();
	}

	template <typename TDataType>
	void GhostDualParticleFluid<TDataType>::preUpdateStates()
	{
		this->DualParticleFluid<TDataType>::preUpdateStates();
		this->constructMergedArrays();
	}

	template <typename TDataType>
	void GhostDualParticleFluid<TDataType>::postUpdateStates()
	{
		auto &pos = this->statePosition()->getData();
		auto &vel = this->stateVelocity()->getData();

		auto &posMerged = this->statePositionMerged()->constData();
		auto &velMerged = this->stateVelocityMerged()->constData();

		pos.assign(posMerged, pos.size());
		vel.assign(velMerged, vel.size());

		this->DualParticleFluid<TDataType>::postUpdateStates();
	}

	template <typename TDataType>
	void GhostDualParticleFluid<TDataType>::constructMergedArrays()
	{
		auto &pos = this->statePosition()->constData();
		auto &vel = this->stateVelocity()->constData();

		int totalNumber = 0;

		uint numOfGhostParticles = 0;
		auto boundaryParticles = this->getBoundaryParticles();
		if (boundaryParticles.size() > 0)
		{
			for (int i = 0; i < boundaryParticles.size(); i++)
			{
				numOfGhostParticles += boundaryParticles[i]->statePosition()->size();
			}
		}

		uint numOfFluidParticles = pos.size();

		totalNumber += (numOfFluidParticles + numOfGhostParticles);

		if (totalNumber <= 0)
			return;

		if (totalNumber != this->statePositionMerged()->size())
		{
			this->statePositionMerged()->resize(totalNumber);
			this->stateVelocityMerged()->resize(totalNumber);
			this->stateAttributeMerged()->resize(totalNumber);
			this->stateNormalMerged()->resize(totalNumber);
		}

		auto &posMerged = this->statePositionMerged()->getData();
		auto &velMerged = this->stateVelocityMerged()->getData();

		int offset = 0;
		posMerged.assign(pos, pos.size(), 0, 0);
		velMerged.assign(vel, vel.size(), 0, 0);

		offset += pos.size();

		auto &normMerged = this->stateNormalMerged()->getData();
		normMerged.reset();

		auto &attMerged = this->stateAttributeMerged()->getData();
		if (numOfFluidParticles != 0)
		{
			cuExecute(offset,
					  GDPF_SetupFluidAttributes,
					  attMerged,
					  offset);
		}

		if (boundaryParticles.size() > 0)
		{

			for (int i = 0; i < boundaryParticles.size(); i++)
			{
				auto &bPos = boundaryParticles[i]->statePosition()->constData();
				auto &bVel = boundaryParticles[i]->stateVelocity()->constData();
				auto &bNor = boundaryParticles[i]->stateNormal()->constData();
				posMerged.assign(bPos, bPos.size(), offset, 0);
				velMerged.assign(bVel, bVel.size(), offset, 0);
				normMerged.assign(bNor, bNor.size(), offset, 0);
				int b_num = bPos.size();
				offset += b_num;
			}
		}

		// if (boundaryParticles != nullptr)
		for (int i = 0; i < boundaryParticles.size(); i++)
		{
			auto &bAtt = boundaryParticles[i]->stateAttribute()->getData();
			cuExecute(bAtt.size(),
					  GDPF_SetupBoundaryAttributes,
					  attMerged,
					  bAtt,
					  offset);
		}
	}

	DEFINE_CLASS(GhostDualParticleFluid);
}