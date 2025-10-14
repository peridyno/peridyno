#include "DualParticleFluid.h"
//DataType
#include "Auxiliary/DataSource.h"

//Collision
#include "Collision/NeighborPointQuery.h"

//ParticleSystem
#include "ParticleSystem/Module/ImplicitViscosity.h"
#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/SemiImplicitDensitySolver.h"
#include "ParticleSystem/Module/VariationalApproximateProjection.h"

//DualParticleSystem
#include "Module/DualParticleIsphModule.h"
#include <DualParticleSystem/Module/ThinFeature.h>


namespace dyno
{
	__global__ void  DPS_AttributeReset(
		DArray<Attribute> att
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= att.size()) return;

		att[pId].setFluid();
		att[pId].setDynamic();
	}

	template<typename TDataType>
	DualParticleFluid<TDataType>::DualParticleFluid()
		: DualParticleFluid<TDataType>::DualParticleFluid(3)
	{

	}


	template<typename TDataType>
	DualParticleFluid<TDataType>::DualParticleFluid(int key)
		: ParticleFluid<TDataType>()
	{
		this->varVirtualParticleSamplingStrategy()->getDataPtr()->setCurrentKey(key);
		this->varReshuffleParticles()->setValue(false);
		this->varSmoothingLength()->setValue(2.4);

		this->animationPipeline()->clear();

		auto m_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->stateSmoothingLength()->connect(m_nbrQuery->inRadius());
		this->statePosition()->connect(m_nbrQuery->inPosition());
		this->animationPipeline()->pushModule(m_nbrQuery);

		if (key == EVirtualParticleSamplingStrategy::SpatiallyAdaptiveStrategy)
		{
			auto m_adaptive_virtual_position = std::make_shared<VirtualSpatiallyAdaptiveStrategy<TDataType>>();
			this->statePosition()->connect(m_adaptive_virtual_position->inRPosition());
			m_adaptive_virtual_position->varSamplingDistance()->setValue(Real(0.005));		/**Virtual particle radius*/
			m_adaptive_virtual_position->varCandidatePointCount()->getDataPtr()->setCurrentKey(VirtualSpatiallyAdaptiveStrategy<TDataType>::neighbors_33);
			vpGen = m_adaptive_virtual_position;
		}
		else if (key == EVirtualParticleSamplingStrategy::ParticleShiftingStrategy)
		{
			auto m_virtual_particle_shifting = std::make_shared<VirtualParticleShiftingStrategy<TDataType >>();
			this->stateFrameNumber()->connect(m_virtual_particle_shifting->inFrameNumber());
			this->stateFrameNumber()->connect(m_virtual_particle_shifting->inFrameNumber());
			this->statePosition()->connect(m_virtual_particle_shifting->inRPosition());
			this->stateTimeStep()->connect(m_virtual_particle_shifting->inTimeStep());
			this->animationPipeline()->pushModule(m_virtual_particle_shifting);
			vpGen = m_virtual_particle_shifting;
		}
		else if (key == EVirtualParticleSamplingStrategy::ColocationStrategy)
		{
			auto m_virtual_equal_to_Real = std::make_shared<VirtualColocationStrategy<TDataType>>();
			this->statePosition()->connect(m_virtual_equal_to_Real->inRPosition());
			this->animationPipeline()->pushModule(m_virtual_equal_to_Real);
			vpGen = m_virtual_equal_to_Real;
		}
		else if (key == EVirtualParticleSamplingStrategy::FissionFusionStrategy)
		{
			auto feature = std::make_shared<ThinFeature<TDataType>>();
			this->statePosition()->connect(feature->inPosition());
			m_nbrQuery->outNeighborIds()->connect(feature->inNeighborIds());
			this->stateSmoothingLength()->connect(feature->inSmoothingLength());
			this->stateSamplingDistance()->connect(feature->inSamplingDistance());
			feature->varThreshold()->setValue(0.1f);
			this->animationPipeline()->pushModule(feature);

			auto gridFission = std::make_shared<VirtualFissionFusionStrategy<TDataType>>();
			gridFission->varTransitionRegionThreshold()->setValue(0.01);
			feature->outThinSheet()->connect(gridFission->inThinSheet());
			feature->outThinFeature()->connect(gridFission->inThinFeature());
			this->statePosition()->connect(gridFission->inRPosition());
			this->stateVelocity()->connect(gridFission->inRVelocity());
			m_nbrQuery->outNeighborIds()->connect(gridFission->inNeighborIds());
			this->stateSmoothingLength()->connect(gridFission->inSmoothingLength());
			this->stateSamplingDistance()->connect(gridFission->inSamplingDistance());
			this->stateFrameNumber()->connect(gridFission->inFrameNumber());
			this->stateTimeStep()->connect(gridFission->inTimeStep());
			this->animationPipeline()->pushModule(gridFission);
			gridFission->varMinDist()->setValue(0.002);
			vpGen = gridFission;
		}

		this->animationPipeline()->pushModule(vpGen);
		vpGen->outVirtualParticles()->connect(this->stateVirtualPosition());

		auto m_rv_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->stateSmoothingLength()->connect(m_rv_nbrQuery->inRadius());
		this->statePosition()->connect(m_rv_nbrQuery->inOther());
		vpGen->outVirtualParticles()->connect(m_rv_nbrQuery->inPosition());
		this->animationPipeline()->pushModule(m_rv_nbrQuery);

		auto m_vr_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->stateSmoothingLength()->connect(m_vr_nbrQuery->inRadius());
		this->statePosition()->connect(m_vr_nbrQuery->inPosition());
		vpGen->outVirtualParticles()->connect(m_vr_nbrQuery->inOther());
		this->animationPipeline()->pushModule(m_vr_nbrQuery);

		auto m_vv_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->stateSmoothingLength()->connect(m_vv_nbrQuery->inRadius());
		vpGen->outVirtualParticles()->connect(m_vv_nbrQuery->inPosition());
		this->animationPipeline()->pushModule(m_vv_nbrQuery);

		auto m_dualIsph = std::make_shared<DualParticleIsphModule<TDataType>>();
		this->stateSmoothingLength()->connect(m_dualIsph->varSmoothingLength());
		this->stateSamplingDistance()->connect(m_dualIsph->varSamplingDistance());
		this->stateTimeStep()->connect(m_dualIsph->inTimeStep());
		this->statePosition()->connect(m_dualIsph->inRPosition());
		vpGen->outVirtualParticles()->connect(m_dualIsph->inVPosition());
		this->stateVelocity()->connect(m_dualIsph->inVelocity());
		m_dualIsph->varResidualThreshold()->setValue(0.001f);
		//this->stateParticleAttribute()->connect(m_dualIsph->inParticleAttribute());
		//this->stateBoundaryNorm()->connect(m_dualIsph->inBoundaryNorm());
		m_nbrQuery->outNeighborIds()->connect(m_dualIsph->inNeighborIds());
		m_rv_nbrQuery->outNeighborIds()->connect(m_dualIsph->inRVNeighborIds());
		m_vr_nbrQuery->outNeighborIds()->connect(m_dualIsph->inVRNeighborIds());
		m_vv_nbrQuery->outNeighborIds()->connect(m_dualIsph->inVVNeighborIds());
		m_dualIsph->varWarmStart()->setValue(true);
		this->animationPipeline()->pushModule(m_dualIsph);

		auto m_integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->stateTimeStep()->connect(m_integrator->inTimeStep());
		this->statePosition()->connect(m_integrator->inPosition());
		this->stateVelocity()->connect(m_integrator->inVelocity());
		this->animationPipeline()->pushModule(m_integrator);

		auto m_visModule = std::make_shared<ImplicitViscosity<TDataType>>();
		m_visModule->varViscosity()->setValue(Real(0.5));
		this->stateTimeStep()->connect(m_visModule->inTimeStep());
		this->stateSamplingDistance()->connect(m_visModule->inSamplingDistance());
		this->stateSmoothingLength()->connect(m_visModule->inSmoothingLength());
		this->stateTimeStep()->connect(m_visModule->inTimeStep());
		this->statePosition()->connect(m_visModule->inPosition());
		this->stateVelocity()->connect(m_visModule->inVelocity());
		m_nbrQuery->outNeighborIds()->connect(m_visModule->inNeighborIds());
		this->animationPipeline()->pushModule(m_visModule);
	}


	template<typename TDataType>
	DualParticleFluid<TDataType>::~DualParticleFluid()
	{
	
	}

	template<typename TDataType>
	void DualParticleFluid<TDataType>::resetStates()
	{
		this->ParticleFluid<TDataType>::resetStates();

		auto ptSet = this->statePointSet()->getDataPtr();
		if(ptSet != nullptr)
		{
			auto pts = ptSet->getPoints();
		}

		std::cout << "Real particle number " << this->statePosition()->size() << std::endl;

		if (this->stateVirtualPointSet()->isEmpty())
		{
			this->stateVirtualPointSet()->allocate();
		}

		if (!this->stateVirtualPosition()->isEmpty())
		{
			auto virtualPoints = this->stateVirtualPointSet()->getDataPtr();
			virtualPoints->setPoints(this->stateVirtualPosition()->getData());
		}
		else
		{
			auto virtualPoints = this->stateVirtualPointSet()->getDataPtr();
			virtualPoints->clear();
		}
	}

	template<typename TDataType>
	void DualParticleFluid<TDataType>::preUpdateStates()
	{
		this->varReshuffleParticles()->setValue(false);
		this->ParticleFluid<TDataType>::preUpdateStates();

	}


	template<typename TDataType>
	void DualParticleFluid<TDataType>::postUpdateStates()
	{
		this->ParticleSystem<TDataType>::postUpdateStates();

		if (!this->stateVirtualPosition()->isEmpty())
		{
			auto virtualPoints = this->stateVirtualPointSet()->getDataPtr();
			virtualPoints->setPoints(this->stateVirtualPosition()->getData());
		}
		else
		{
			auto virtualPoints = this->stateVirtualPointSet()->getDataPtr();
			virtualPoints->clear();
		}
	}
	
	DEFINE_CLASS(DualParticleFluid);
}


