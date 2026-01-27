#include "StaticBoundaryEmitter.h"
#include "Algorithm/CudaRand.h"

namespace dyno
{
	IMPLEMENT_TCLASS(StaticBoundaryEmitter, TDataType)

	template <typename Real, typename Coord>
	__global__ void SBE_constrain(
		DArray<Coord> pos,
		DArray<Coord> vel,
		Coord location,
		Real xwidth,
		Real yhigh)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= pos.size()) return;

		Coord p = pos[tId];
		Coord v = vel[tId];

		if (p[1]>location[1])
		{
			Coord normal(0.0f, 1.0f, 0.0f);
			Real olddist = p[1] - location[1];
			normal = normal.normalize();
			RandNumber rGen(tId);
			Real dist = 0.0001f*rGen.Generate();
			// reflect position
			p -= (olddist + dist)*normal;
			// reflect velocity
			Real vlength = v.norm();
			Real vec_n = v.dot(normal);
			Coord vec_normal = vec_n * normal;
			Coord vec_tan = v - vec_normal;
			if (vec_n > 0) vec_normal = -vec_normal;
			vec_normal *= (0.0f);
			v = vec_normal + vec_tan * (1.0);
		}
		else if (p[0]<(location[0]-0.5*xwidth)&&p[1]>(location[1]-yhigh))
		{
			Coord normal(-1.0f, 0.0f, 0.0f);
			Real olddist = (location[0] - 0.5*xwidth) - p[0];
			normal = normal.normalize();
			RandNumber rGen(tId);
			Real dist = 0.0001f*rGen.Generate();
			// reflect position
			p -= (olddist + dist)*normal;
			// reflect velocity
			Real vlength = v.norm();
			Real vec_n = v.dot(normal);
			Coord vec_normal = vec_n * normal;
			Coord vec_tan = v - vec_normal;
			if (vec_n > 0) vec_normal = -vec_normal;
			vec_normal *= (0.0f);
			v = vec_normal + vec_tan * (1.0f);
		}
		else if (p[0]>(location[0] + 0.5*xwidth) && p[1]>(location[1] - yhigh))
		{
			Coord normal(1.0f, 0.0f, 0.0f);
			Real olddist = p[0] - (location[0] + 0.5*xwidth);
			normal = normal.normalize();
			RandNumber rGen(tId);
			Real dist = 0.0001f*rGen.Generate();
			// reflect position
			p -= (olddist + dist)*normal;
			// reflect velocity
			Real vlength = v.norm();
			Real vec_n = v.dot(normal);
			Coord vec_normal = vec_n * normal;
			Coord vec_tan = v - vec_normal;
			if (vec_n > 0) vec_normal = -vec_normal;
			vec_normal *= (0.0f);
			v = vec_normal + vec_tan * (1.0f);
		}

		pos[tId] = p;
		vel[tId] = v;
	}

	template<typename TDataType>
	void StaticBoundaryEmitter<TDataType>::compute()
	{
		auto posFd = this->inPosition()->getData();
		auto velFd = this->inVelocity()->getData();
		if (!posFd.isEmpty() && !velFd.isEmpty())
			cuExecute(posFd.size(),
				SBE_constrain,
				posFd,
				velFd,
				this->varLocation()->getData(),
				this->varXWidth()->getData(),
				this->varYHigh()->getData());
	}

	DEFINE_CLASS(StaticBoundaryEmitter);
}