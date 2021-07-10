#include "ParticleFluid.h"
#include "PositionBasedFluidModel.h"

#include "Topology/PointSet.h"
#include "SummationDensity.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(ParticleFluid, TDataType)

	template<typename TDataType>
	ParticleFluid<TDataType>::ParticleFluid(std::string name)
		: ParticleSystem<TDataType>(name)
	{
// 		auto pbf = this->template setNumericalModel<PositionBasedFluidModel<TDataType>>("pbd");
// 		this->setNumericalModel(pbf);

		auto pbf = std::make_shared<PositionBasedFluidModel<TDataType>>();
		this->animationPipeline()->pushModule(pbf);

		this->varTimeStep()->connect(pbf->inTimeStep());
		this->currentPosition()->connect(pbf->inPosition());
		this->currentVelocity()->connect(pbf->inVelocity());
		this->currentForce()->connect(pbf->inForce());
	}

	template<typename TDataType>
	ParticleFluid<TDataType>::~ParticleFluid()
	{
		
	}

	template<typename TDataType>
	void ParticleFluid<TDataType>::advance(Real dt)
	{		
		auto emitters = this->getParticleEmitters();

		int curNum = this->currentPosition()->getElementCount();
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
					pBuf.assign(this->currentPosition()->getData());
					vBuf.assign(this->currentVelocity()->getData());
					fBuf.assign(this->currentForce()->getData());
				}

				this->currentPosition()->setElementCount(totalNum);
				this->currentVelocity()->setElementCount(totalNum);
				this->currentForce()->setElementCount(totalNum);

				//printf("###### %d\n", this->currentPosition()->getElementCount());

				DArray<Coord>& new_pos = this->currentPosition()->getData();
				DArray<Coord>& new_vel = this->currentVelocity()->getData();
				DArray<Coord>& new_force = this->currentForce()->getData();

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

		if (totalNum > 0)
		{
			this->animationPipeline()->update();
		}
	}


	template<typename TDataType>
	bool ParticleFluid<TDataType>::resetStatus()
	{
		return ParticleSystem<TDataType>::resetStatus();
	}

	DEFINE_CLASS(ParticleFluid);
}