#include "GhostFluid.h"

#include "Module/ProjectionBasedFluidModel.h"

namespace dyno
{
	IMPLEMENT_TCLASS(GhostFluid, TDataType)

	template<typename TDataType>
	GhostFluid<TDataType>::GhostFluid()
		: ParticleSystem<TDataType>()
	{
		auto model = std::make_shared<ProjectionBasedFluidModel<DataType3f>>();
		model->varSmoothingLength()->setValue(0.01);
		
		this->stateTimeStep()->connect(model->inTimeStep());
		this->statePositionMerged()->connect(model->inPosition());
		this->stateVelocityMerged()->connect(model->inVelocity());
		this->stateNormalMerged()->connect(model->inNormal());
		this->stateAttributeMerged()->connect(model->inAttribute());
		this->animationPipeline()->pushModule(model);

		this->setDt(0.001f);
	}

	__global__ void SetupFluidAttributes(
		DArray<Attribute> allAttributes,
		int num)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= num) return;

		allAttributes[pId].setDynamic();
		allAttributes[pId].setFluid();
	}

	__global__ void SetupBoundaryAttributes(
		DArray<Attribute> allAttributes,
		DArray<Attribute> boundaryAttributes,
		int offset)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= boundaryAttributes.size()) return;

		allAttributes[offset + pId].setFixed();
		allAttributes[offset + pId].setRigid();
	}

	template<typename TDataType>
	void GhostFluid<TDataType>::resetStates()
	{
		auto boundaryParticles = this->getBoundaryParticles();
		auto fluidParticles = this->getFluidParticles();

		int totalNumber = 0;
		uint numOfGhostParticles = boundaryParticles != nullptr ? boundaryParticles->statePosition()->size() : 0;
		uint numOfFluidParticles = fluidParticles != nullptr ? fluidParticles->statePosition()->size() : 0;

		totalNumber += (numOfFluidParticles + numOfGhostParticles);

		if (totalNumber <= 0)
			return;

		//Initialize state fields for merged data
		if (totalNumber != this->statePositionMerged()->size()) {
			this->statePositionMerged()->resize(totalNumber);
			this->stateVelocityMerged()->resize(totalNumber);
			this->stateAttributeMerged()->resize(totalNumber);
			this->stateNormalMerged()->resize(totalNumber);
		}

		auto& posMerged = this->statePositionMerged()->getData();
		auto& velMerged = this->stateVelocityMerged()->getData();

		int offset = 0;
		if (fluidParticles != nullptr)
		{
			auto& fPos = fluidParticles->statePosition()->constData();
			auto& fVel = fluidParticles->stateVelocity()->constData();
			posMerged.assign(fPos, fPos.size(), 0, 0);
			velMerged.assign(fVel, fVel.size(), 0, 0);

			offset += fPos.size();
		}

		auto& normMerged = this->stateNormalMerged()->getData();
		normMerged.reset();

		if (boundaryParticles != nullptr)
		{
			auto& bPos = boundaryParticles->statePosition()->constData();
			auto& bVel = boundaryParticles->stateVelocity()->constData();
			auto& bNor = boundaryParticles->stateNormal()->constData();
			posMerged.assign(bPos, bPos.size(), offset, 0);
			velMerged.assign(bVel, bVel.size(), offset, 0);
			normMerged.assign(bNor, bNor.size(), offset, 0);
		}


		auto& attMerged = this->stateAttributeMerged()->getData();
		if (fluidParticles != nullptr)
		{
			cuExecute(offset,
				SetupFluidAttributes,
				attMerged,
				offset);
		}

		if (boundaryParticles != nullptr)
		{
			auto& bAtt = boundaryParticles->stateAttribute()->getData();
			cuExecute(bAtt.size(),
				SetupBoundaryAttributes,
				attMerged,
				bAtt,
				offset);
		}

		//Initialize state fields for the fluid
		if (numOfFluidParticles != this->statePosition()->size()) {
			this->statePosition()->resize(totalNumber);
			this->stateVelocity()->resize(totalNumber);
		}

		//Initialize the PointSet for fluid particles
		if (!this->statePosition()->isEmpty())
		{
			auto& fPos = fluidParticles->statePosition()->constData();
			auto& fVel = fluidParticles->stateVelocity()->constData();
			this->statePosition()->assign(fPos);
			this->stateVelocity()->assign(fVel);

			auto points = this->statePointSet()->getDataPtr();
			points->setPoints(this->statePosition()->getData());
		}
		else
		{
			auto points = this->statePointSet()->getDataPtr();
			points->clear();
		}
	}

	template<typename TDataType>
	void GhostFluid<TDataType>::preUpdateStates()
	{

	}

	template<typename TDataType>
	void GhostFluid<TDataType>::postUpdateStates()
	{
		auto& pos = this->statePosition()->getData();
		auto& vel = this->stateVelocity()->getData();

		auto& posMerged = this->statePositionMerged()->constData();
		auto& velMerged = this->stateVelocityMerged()->constData();

		pos.assign(posMerged, pos.size());
		vel.assign(velMerged, vel.size());

		//Initialize the PointSet for fluid particles
		if (!this->statePosition()->isEmpty())
		{
			auto points = this->statePointSet()->getDataPtr();
			points->setPoints(this->statePosition()->getData());
		}
		else
		{
			auto points = this->statePointSet()->getDataPtr();
			points->clear();
		}
	}

	DEFINE_CLASS(GhostFluid);
}