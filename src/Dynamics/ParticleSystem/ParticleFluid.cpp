#include "ParticleFluid.h"
#include "PositionBasedFluidModel.h"

#include "Topology/PointSet.h"
#include "SummationDensity.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ParticleFluid, TDataType)

	template<typename TDataType>
	ParticleFluid<TDataType>::ParticleFluid(std::string name)
		: ParticleSystem<TDataType>(name)
	{
// 		auto pbf = this->template setNumericalModel<PositionBasedFluidModel<TDataType>>("pbd");
// 		this->setNumericalModel(pbf);

		auto pbf = std::make_shared<PositionBasedFluidModel<TDataType>>();
		this->animationPipeline()->pushModule(pbf);

		this->varTimeStep()->connect(pbf->inTimeStep());
		this->statePosition()->connect(pbf->inPosition());
		this->stateVelocity()->connect(pbf->inVelocity());
		this->stateForce()->connect(pbf->inForce());
	}

	template<typename TDataType>
	ParticleFluid<TDataType>::~ParticleFluid()
	{
		Log::sendMessage(Log::Info, "ParticleFluid released \n");
	}

	template<typename TDataType>
	void ParticleFluid<TDataType>::preUpdateStates()
	{
		auto emitters = this->getParticleEmitters();

		int curNum = this->statePosition()->getElementCount();
		int totalNum = curNum;
		if (emitters.size() > 0)
		{
			for (int i = 0; i < emitters.size(); i++)
			{
				totalNum += emitters[i]->sizeOfParticles();
			}

			if (totalNum > curNum)
			{
				DArray<Coord> pBuf;
				DArray<Coord> vBuf;
				DArray<Coord> fBuf;

				if (curNum > 0)
				{
					pBuf.assign(this->statePosition()->getData());
					vBuf.assign(this->stateVelocity()->getData());
					fBuf.assign(this->stateForce()->getData());
				}

				this->statePosition()->setElementCount(totalNum);
				this->stateVelocity()->setElementCount(totalNum);
				this->stateForce()->setElementCount(totalNum);

				//printf("###### %d\n", this->currentPosition()->getElementCount());

				DArray<Coord>& new_pos = this->statePosition()->getData();
				DArray<Coord>& new_vel = this->stateVelocity()->getData();
				DArray<Coord>& new_force = this->stateForce()->getData();

				if (curNum > 0)
				{
					cudaMemcpy(new_pos.begin(), pBuf.begin(), curNum * sizeof(Coord), cudaMemcpyDeviceToDevice);
					cudaMemcpy(new_vel.begin(), vBuf.begin(), curNum * sizeof(Coord), cudaMemcpyDeviceToDevice);
					cudaMemcpy(new_force.begin(), fBuf.begin(), curNum * sizeof(Coord), cudaMemcpyDeviceToDevice);

					pBuf.clear();
					vBuf.clear();
					fBuf.clear();
				}

				int start = curNum;
				for (int i = 0; i < emitters.size(); i++)
				{
					int num = emitters[i]->sizeOfParticles();
					if (num > 0)
					{
						DArray<Coord>& points = emitters[i]->getPositions();
						DArray<Coord>& vels = emitters[i]->getVelocities();
						DArray<Coord> fors(num);
						fors.reset();

						cudaMemcpy(new_pos.begin() + start, points.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
						cudaMemcpy(new_vel.begin() + start, vels.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
						cudaMemcpy(new_force.begin() + start, fors.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
						fors.clear();

						start += num;
					}
				}
			}
		}
	}

	template<typename TDataType>
	void ParticleFluid<TDataType>::resetStates()
	{
		ParticleSystem<TDataType>::resetStates();
	}

	DEFINE_CLASS(ParticleFluid);
}