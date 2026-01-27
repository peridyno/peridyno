#include "StaticBoundaryCircle.h"
#include "Algorithm/CudaRand.h"

namespace dyno
{
	IMPLEMENT_TCLASS(StaticBoundaryCircle, TDataType)

	template <typename Real, typename Coord>
	__global__ void SBC_constrain(
		DArray<Coord> pos,
		DArray<Coord> vel,
		Coord center,
		Real radius,
		Real dx,
		Real normalFriction,
		Real tangentialFriction,
		Real planeTangentialFriction)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= pos.size()) return;

		Coord p = pos[tId];
		Coord v = vel[tId];

		Real dist = (Coord(p[0], p[1], 0.0f) - center).norm();
		dist = radius - dist;

		if (dist<0)
		{
			Coord normal = p - center;
			Real olddist = normal.norm() - radius;
			normal = normal.normalize();
			RandNumber rGen(tId);
			dist = 0.0001f*rGen.Generate();
			// reflect position
			p -= (olddist + dist)*normal;
			// reflect velocity
			Real vlength = v.norm();
			Real vec_n = v.dot(normal);
			Coord vec_normal = vec_n * normal;
			Coord vec_tan = v - vec_normal;
			if (vec_n > 0) vec_normal = -vec_normal;
			vec_normal *= (1.0f - normalFriction);
			v = vec_normal + vec_tan * (1.0f - tangentialFriction);
		}
		else if (abs(p[2]) > (dx / 2))
		{
			Coord normal = p - Coord(p[0], p[1], 0.0f);
			Real olddist = normal.norm() - (dx / 2);
			normal = normal.normalize();
			RandNumber rGen(tId);
			dist = 0.0001f*rGen.Generate();
			// reflect position
			p -= (olddist + dist)*normal;
			// reflect velocity
			Real vlength = v.norm();
			Real vec_n = v.dot(normal);
			Coord vec_normal = vec_n * normal;
			Coord vec_tan = v - vec_normal;
			if (vec_n > 0) vec_normal = -vec_normal;
			vec_normal *= (1.0f - normalFriction);
			v = vec_normal + vec_tan * (1.0f - planeTangentialFriction);
			//v = vec_normal + vec_tan * (1.0f);
		}

		pos[tId] = p;
		vel[tId] = v;
	}

	template<typename TDataType>
	void StaticBoundaryCircle<TDataType>::compute()
	{
		auto posFd = this->inPosition()->getData();
		auto velFd = this->inVelocity()->getData();
		if (!posFd.isEmpty() && !velFd.isEmpty())
			cuExecute(posFd.size(),
				SBC_constrain,
				posFd,
				velFd,
				this->varCenter()->getData(),
				this->varRadius()->getData(),
				this->varDx()->getData(),
				this->varNormalFriction()->getData(),
				this->varTangentialFriction()->getData(),
				this->varPlaneTangentialFriction()->getData());
	}

	DEFINE_CLASS(StaticBoundaryCircle);
}