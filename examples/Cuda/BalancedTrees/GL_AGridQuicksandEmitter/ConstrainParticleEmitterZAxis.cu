#include "ConstrainParticleEmitterZAxis.h"
#include <thrust/sort.h>

namespace dyno
{
	IMPLEMENT_TCLASS(ConstrainParticleEmitterZAxis, TDataType)

		template<typename TDataType>
	ConstrainParticleEmitterZAxis<TDataType>::ConstrainParticleEmitterZAxis()
		: Node()
	{
	}

	template<typename TDataType>
	ConstrainParticleEmitterZAxis<TDataType>::~ConstrainParticleEmitterZAxis()
	{
	}

	//template <typename Real, typename Coord>
	//__global__ void CPEZ_CountPoints(
	//	DArray<int> count,
	//	DArray<Coord> pos,
	//	Real zdx)
	//{
	//	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	//	if (tId >= pos.size()) return;

	//	if ((pos[tId][2] >= (-zdx / 2)) && (pos[tId][2] <= (zdx / 2)))
	//		count[tId] = 1;
	//}

	//template <typename Real, typename Coord>
	//__global__ void CPEZ_ComputePoints(
	//	DArray<Coord> npos,
	//	DArray<Coord> nvel,
	//	DArray<int> count,
	//	DArray<Coord> pos,
	//	DArray<Coord> vel,
	//	Real zdx)
	//{
	//	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	//	if (tId >= pos.size()) return;

	//	if ((pos[tId][2] >= (-zdx / 2)) && (pos[tId][2] <= (zdx / 2)))
	//	{
	//		npos[count[tId]] = pos[tId];
	//		nvel[count[tId]] = vel[tId];
	//	}
	//}

	template <typename Real, typename Coord>
	__global__ void CPEZ_ModifyPoints(
		DArray<Coord> pos,
		Real zdx_old,
		Real zdx)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= pos.size()) return;

		Real prop = (pos[tId][2] + (0.5*zdx_old)) / zdx_old;
		Real pz = prop * zdx - (0.5*zdx);

		pos[tId][2] = pz;
	}

	template<typename TDataType>
	void ConstrainParticleEmitterZAxis<TDataType>::updateStates()
	{
		auto emitters = this->getParticleEmitters();
		if (emitters.size() > 0)
		{
			for (int i = 0; i < emitters.size(); i++)
			{
				int oldnum = emitters[i]->sizeOfParticles();
				if (oldnum > 0)
				{
					DArray<Coord>& points = emitters[i]->getPositions();
					DArray<Coord>& vels = emitters[i]->getVelocities();

					//DArray<int> count(oldnum);
					//count.reset();
					//cuExecute(oldnum,
					//	CPEZ_CountPoints,
					//	count,
					//	points,
					//	this->varZdx()->getData());
					//Reduction<int> reduce;
					//int newnum = reduce.accumulate(count.begin(), count.size());
					//Scan<int> scan;
					//scan.exclusive(count.begin(), count.size());
					//printf("ConstrainParticleEmitter: %d %d \n", oldnum, newnum);

					//if (oldnum == newnum) break;

					//DArray<Coord> pbuf;
					//DArray<Coord> vbuf;
					//pbuf.assign(points);
					//vbuf.assign(vels);
					//points.resize(newnum);
					//vels.resize(newnum);
					//cuExecute(oldnum,
					//	CPEZ_ComputePoints,
					//	points,
					//	vels,
					//	count,
					//	pbuf,
					//	vbuf,
					//	this->varZdx()->getData());

					//count.clear();
					//pbuf.clear();
					//vbuf.clear();


					cuExecute(points.size(),
						CPEZ_ModifyPoints,
						points,
						this->varZHeight()->getData(),
						this->varZdx()->getData());
				}
			}
		}
	}


	DEFINE_CLASS(ConstrainParticleEmitterZAxis);
}