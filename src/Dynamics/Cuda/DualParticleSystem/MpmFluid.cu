#include "MpmFluid.h"
//DataType
#include "Auxiliary/DataSource.h"

//Collision
#include "Collision/NeighborPointQuery.h"

//ParticleSystem
#include "ParticleSystem/Module/ImplicitViscosity.h"
#include "ParticleSystem/Module/ParticleIntegrator.h"

//DualParticleSystem
#include "Module/DualParticleIsphModule.h"

//MPM Fluid
#include "Module/FlipFluidExplicitSolver.h"

namespace dyno
{
	__global__ void  MPM_AttributeReset(
		DArray<Attribute> att
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= att.size()) return;

		att[pId].setFluid();
		att[pId].setDynamic();
	}


	template<typename TDataType>
	MpmFluid<TDataType>::MpmFluid()
		: ParticleFluid<TDataType>()
	{
		this->varReshuffleParticles()->setValue(false);

		this->animationPipeline()->clear();

		auto smoothingLength = std::make_shared<FloatingNumber<TDataType>>();
		this->animationPipeline()->pushModule(smoothingLength);
		smoothingLength->varValue()->setValue(Real(0.012));

		auto samplingDistance = std::make_shared<FloatingNumber<TDataType>>();
		this->animationPipeline()->pushModule(samplingDistance);
		samplingDistance->varValue()->setValue(Real(0.005));

		auto m_adaptive_virtual_position = std::make_shared<VirtualSpatiallyAdaptiveStrategy<TDataType>>();
		this->statePosition()->connect(m_adaptive_virtual_position->inRPosition());
		m_adaptive_virtual_position->varSamplingDistance()->setValue(Real(0.005));		/**Virtual particle radius*/
		m_adaptive_virtual_position->varCandidatePointCount()->getDataPtr()->setCurrentKey(VirtualSpatiallyAdaptiveStrategy<TDataType>::neighbors_125);
		vpGen = m_adaptive_virtual_position;
		
		this->animationPipeline()->pushModule(vpGen);
		vpGen->outVirtualParticles()->connect(this->stateGridPosition());

		auto m_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		smoothingLength->outFloating()->connect(m_nbrQuery->inRadius());
		this->statePosition()->connect(m_nbrQuery->inPosition());
		this->animationPipeline()->pushModule(m_nbrQuery);

		auto m_rv_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		smoothingLength->outFloating()->connect(m_rv_nbrQuery->inRadius());
		this->statePosition()->connect(m_rv_nbrQuery->inOther());
		vpGen->outVirtualParticles()->connect(m_rv_nbrQuery->inPosition());
		this->animationPipeline()->pushModule(m_rv_nbrQuery);

		auto m_vr_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		smoothingLength->outFloating()->connect(m_vr_nbrQuery->inRadius());
		this->statePosition()->connect(m_vr_nbrQuery->inPosition());
		vpGen->outVirtualParticles()->connect(m_vr_nbrQuery->inOther());
		this->animationPipeline()->pushModule(m_vr_nbrQuery);

		auto m_vv_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		smoothingLength->outFloating()->connect(m_vv_nbrQuery->inRadius());
		vpGen->outVirtualParticles()->connect(m_vv_nbrQuery->inPosition());
		this->animationPipeline()->pushModule(m_vv_nbrQuery);

		auto m_flipEx = std::make_shared <FlipFluidExplicitSolver<TDataType>>();
		samplingDistance->outFloating()->connect(m_flipEx->inSamplingDistance());
		this->stateTimeStep()->connect(m_flipEx->inTimeStep());
		this->statePosition()->connect(m_flipEx->inParticlePosition());
		this->stateVelocity()->connect(m_flipEx->inParticleVelocity());
		this->stateGridVelocity()->connect(m_flipEx->inGridVelocity());
		this->stateGridSpacing()->connect(m_flipEx->inGridSpacing());
		m_rv_nbrQuery->outNeighborIds()->connect(m_flipEx->inPGNeighborIds());
		this->stateFrameNumber()->connect(m_flipEx->inFrameNumber());
		vpGen->outVirtualParticles()->connect(m_flipEx->inAdaptGridPosition());
		this->animationPipeline()->pushModule(m_flipEx);
		

		auto m_integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->stateTimeStep()->connect(m_integrator->inTimeStep());
		this->statePosition()->connect(m_integrator->inPosition());
		this->stateVelocity()->connect(m_integrator->inVelocity());
		this->stateParticleAttribute()->connect(m_integrator->inAttribute());
		this->animationPipeline()->pushModule(m_integrator);

		//auto m_visModule = std::make_shared<ImplicitViscosity<TDataType>>();
		//m_visModule->varViscosity()->setValue(Real(0.3));
		//this->stateTimeStep()->connect(m_visModule->inTimeStep());
		//smoothingLength->outFloating()->connect(m_visModule->inSmoothingLength());
		//samplingDistance->outFloating()->connect(m_visModule->inSamplingDistance());
		//this->stateTimeStep()->connect(m_visModule->inTimeStep());
		//this->statePosition()->connect(m_visModule->inPosition());
		//this->stateVelocity()->connect(m_visModule->inVelocity());
		//m_nbrQuery->outNeighborIds()->connect(m_visModule->inNeighborIds());
		//this->animationPipeline()->pushModule(m_visModule);
	}


	template<typename TDataType>
	MpmFluid<TDataType>::~MpmFluid()
	{
	
	}

	template<typename TDataType>
	void MpmFluid<TDataType>::resetStates()
	{
		this->ParticleFluid<TDataType>::resetStates();

		auto ptSet = this->statePointSet()->getDataPtr();
		if(ptSet != nullptr)
		{
			auto pts = ptSet->getPoints();
			this->stateBoundaryNorm()->resize(pts.size());
			this->stateParticleAttribute()->resize(pts.size());

			cuExecute(pts.size(), MPM_AttributeReset,
				this->stateParticleAttribute()->getData());

			this->stateBoundaryNorm()->getDataPtr()->reset();
		}

		if (this->stateVirtualPointSet()->isEmpty())
		{
			this->stateVirtualPointSet()->allocate();
		}

		if (!this->stateGridPosition()->isEmpty())
		{
			auto virtualPoints = this->stateVirtualPointSet()->getDataPtr();
			virtualPoints->setPoints(this->stateGridPosition()->getData());
		}
		else
		{
			auto virtualPoints = this->stateVirtualPointSet()->getDataPtr();
			virtualPoints->clear();
		}

	}

	template<typename TDataType>
	void MpmFluid<TDataType>::preUpdateStates()
	{
		this->varReshuffleParticles()->setValue(false);
		this->ParticleFluid<TDataType>::preUpdateStates();

		this->stateBoundaryNorm()->resize(this->statePosition()->size());
		this->stateBoundaryNorm()->reset();
		this->stateParticleAttribute()->resize(this->statePosition()->size());

		cuExecute(this->statePosition()->size(), MPM_AttributeReset,
			this->stateParticleAttribute()->getData());

		if (this->stateGridVelocity()->size() != this->stateGridPosition()->size())
		{
			this->stateGridVelocity()->resize(this->stateGridPosition()->size());
			this->stateGridVelocity()->getData().reset();
		}

	}


	template<typename TDataType>
	void MpmFluid<TDataType>::postUpdateStates()
	{
		this->ParticleSystem<TDataType>::postUpdateStates();

		if (!this->stateGridPosition()->isEmpty())
		{
			auto virtualPoints = this->stateVirtualPointSet()->getDataPtr();
			virtualPoints->setPoints(this->stateGridPosition()->getData());
		}
		else
		{
			auto virtualPoints = this->stateVirtualPointSet()->getDataPtr();
			virtualPoints->clear();
		}
	}
	
	DEFINE_CLASS(MpmFluid);
}


