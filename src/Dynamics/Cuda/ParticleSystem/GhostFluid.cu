#include "GhostFluid.h"

#include "Module/ProjectionBasedFluidModel.h"

namespace dyno
{
	IMPLEMENT_TCLASS(GhostFluid, TDataType)

	template<typename TDataType>
	GhostFluid<TDataType>::GhostFluid()
		: ParticleFluid<TDataType>()
	{
		this->animationPipeline()->clear();

		auto model = std::make_shared<ProjectionBasedFluidModel<DataType3f>>();

		this->stateTimeStep()->connect(model->inTimeStep());
		this->stateSamplingDistance()->connect(model->inSamplingDistance());
		this->stateSmoothingLength()->connect(model->inSmoothingLength());
		this->statePositionMerged()->connect(model->inPosition());
		this->stateVelocityMerged()->connect(model->inVelocity());
		this->stateNormalMerged()->connect(model->inNormal());
		this->stateAttributeMerged()->connect(model->inAttribute());
		this->animationPipeline()->pushModule(model);

		this->setDt(0.001f);
	}

	template<typename TDataType>
	void GhostFluid<TDataType>::resetStates()
	{
		ParticleFluid<TDataType>::resetStates();

		constructMergedArrays();
	}

	template<typename TDataType>
	void GhostFluid<TDataType>::preUpdateStates()
	{
		ParticleFluid<TDataType>::preUpdateStates();

		//To ensure updates on fluid particles by other nodes can be mapped onto the merged arrays
		constructMergedArrays();
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

		ParticleFluid<TDataType>::postUpdateStates();
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
	void GhostFluid<TDataType>::constructMergedArrays()
	{
		auto& pos = this->statePosition()->constData();
		auto& vel = this->stateVelocity()->constData();

		auto boundaryParticles = this->getBoundaryParticles();

		int totalNumber = 0;
		uint numOfGhostParticles = boundaryParticles != nullptr ? boundaryParticles->statePosition()->size() : 0;
		uint numOfFluidParticles = pos.size();

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
		posMerged.assign(pos, pos.size(), 0, 0);
		velMerged.assign(vel, vel.size(), 0, 0);

		offset += pos.size();

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

		//Initialize the attribute field
		auto& attMerged = this->stateAttributeMerged()->getData();
		if (numOfFluidParticles != 0)
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
	}

	DEFINE_CLASS(GhostFluid);
}