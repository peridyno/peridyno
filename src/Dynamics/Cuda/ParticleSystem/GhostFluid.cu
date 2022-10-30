#include "GhostFluid.h"

#include "Module/ProjectionBasedFluidModel.h"

namespace dyno
{
	IMPLEMENT_TCLASS(GhostFluid, TDataType)

	template<typename TDataType>
	GhostFluid<TDataType>::GhostFluid()
		: Node()
	{
		auto model = std::make_shared<ProjectionBasedFluidModel<DataType3f>>();
		model->varSmoothingLength()->setValue(0.01);
		
		this->stateTimeStep()->connect(model->inTimeStep());
		this->statePosition()->connect(model->inPosition());
		this->stateVelocity()->connect(model->inVelocity());
		this->stateForce()->connect(model->inForce());
		this->stateNormal()->connect(model->inNormal());
		this->stateAttribute()->connect(model->inAttribute());
		this->animationPipeline()->pushModule(model);
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
	void GhostFluid<TDataType>::preUpdateStates()
	{
		auto boundaryParticles	= this->getBoundaryParticles();
		auto fluidParticles		= this->getFluidParticles();

		int pNum = 0;
		pNum += boundaryParticles != nullptr ? boundaryParticles->statePosition()->size() : 0;
		pNum += fluidParticles != nullptr ? fluidParticles->statePosition()->size() : 0;

		if (pNum <= 0)
			return;

		if (pNum != this->statePosition()->size())
		{
			this->statePosition()->resize(pNum);
			this->stateVelocity()->resize(pNum);
			this->stateForce()->resize(pNum);
			this->stateAttribute()->resize(pNum);
			this->stateNormal()->resize(pNum);
		}

		auto& pos = this->statePosition()->getData();
		auto& vel = this->stateVelocity()->getData();
		auto& force = this->stateForce()->getData();

		force.reset();

		int offset = 0;
		if (fluidParticles != nullptr)
		{
			auto& fPos = fluidParticles->statePosition()->getData();
			auto& fVel = fluidParticles->stateVelocity()->getData();
			pos.assign(fPos, fPos.size(), 0, 0);
			vel.assign(fVel, fVel.size(), 0, 0);

			offset += fPos.size();
		}
		
		auto& normals = this->stateNormal()->getData();
		normals.reset();

		if (boundaryParticles != nullptr)
		{
			auto& bPos = boundaryParticles->statePosition()->getData();
			auto& bVel = boundaryParticles->stateVelocity()->getData();
			auto& bNor = boundaryParticles->stateNormal()->getData();
			pos.assign(bPos, bPos.size(), offset, 0);
			vel.assign(bVel, bVel.size(), offset, 0);
			normals.assign(bNor, bNor.size(), offset, 0);
		}


		auto& atts = this->stateAttribute()->getData();
		if (fluidParticles != nullptr)
		{
			cuExecute(offset,
				SetupFluidAttributes,
				atts,
				offset);
		}
		
		if (boundaryParticles != nullptr)
		{
			auto& bAtt = boundaryParticles->stateAttribute()->getData();
			cuExecute(bAtt.size(),
				SetupBoundaryAttributes,
				atts,
				bAtt,
				offset);
		}
	}

	template<typename TDataType>
	void GhostFluid<TDataType>::postUpdateStates()
	{
		auto boundaryParticles = this->getBoundaryParticles();
		auto fluidParticles = this->getFluidParticles();

		auto& pos = this->statePosition()->getData();
		auto& vel = this->stateVelocity()->getData();

		int offset = 0;
		if (fluidParticles != nullptr)
		{
			auto& fPos = fluidParticles->statePosition()->getData();
			auto& fVel = fluidParticles->stateVelocity()->getData();
			fPos.assign(pos, fPos.size(), 0, 0);
			fVel.assign(vel, fVel.size(), 0, 0);

			offset += fPos.size();
		}

		if (boundaryParticles != nullptr)
		{
			auto& bPos = boundaryParticles->statePosition()->getData();
			auto& bVel = boundaryParticles->stateVelocity()->getData();
			bPos.assign(pos, bPos.size(), 0, offset);
			bVel.assign(vel, bVel.size(), 0, offset);
		}
	}

	DEFINE_CLASS(GhostFluid);
}