#include "AdaptiveBoundary.h"
#include "Algorithm/CudaRand.h"
//#include "Log.h"
//#include "Node.h"

//#include "Module/AdaptiveBoundaryConstraint.h"

//#include "Topology/DistanceField3D.h"
//#include "Topology/TriangleSet.h"

namespace dyno
{
	IMPLEMENT_TCLASS(AdaptiveBoundary, TDataType)

	template<typename TDataType>
	AdaptiveBoundary<TDataType>::AdaptiveBoundary()
		: Node()
	{
		this->varNormalFriction()->setValue(0.95);
		this->varTangentialFriction()->setValue(0.0);
	}

	template<typename TDataType>
	AdaptiveBoundary<TDataType>::~AdaptiveBoundary()
	{
	}

	template<typename Real, typename Coord>
	__global__ void K_ConstrainSDF(
		DArray<Coord> posArr,
		DArray<Coord> velArr,
		DArray<Real> posSDF,
		DArray<Coord> posNormal,
		Real normalFriction,
		Real tangentialFriction,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos = posArr[pId];
		Coord vec = velArr[pId];

		Real dist = posSDF[pId];
		Coord normal = posNormal[pId];
		// constrain particle
		if (dist <= 0) 
		{
			Real olddist = -dist;
			RandNumber rGen(pId);
			dist = 0.0001f*rGen.Generate();
			// reflect position
			pos -= (olddist + dist)*normal;
			// reflect velocity
			Real vlength = vec.norm();
			Real vec_n = vec.dot(normal);
			Coord vec_normal = vec_n * normal;
			Coord vec_tan = vec - vec_normal;
			if (vec_n > 0) vec_normal = -vec_normal;
			vec_normal *= (1.0f - normalFriction);
			vec = vec_normal + vec_tan * (1.0f - tangentialFriction);
			//vec *= pow(Real(M_E), -dt*tangentialFriction);
			//vec *= (1.0f - tangentialFriction);
		}

		posArr[pId] = pos;
		velArr[pId] = vec;
	}

	template<typename TDataType>
	void AdaptiveBoundary<TDataType>::updateStates()
	{
		Real dt = this->stateTimeStep()->getData();

		auto aSDF = this->getBoundarys();

		auto pSys = this->getParticleSystems();

		DArray<Coord> pos_normal;
		DArray<Real> pos_sdf;
		for (int i = 0; i < pSys.size(); i++)
		{
			DArray<Coord>& position = pSys[i]->statePosition()->getData();
			DArray<Coord>& velocity = pSys[i]->stateVelocity()->getData();
			for (int j = 0; j < aSDF.size(); j++)
			{
				aSDF[j]->stateSDFTopology()->constDataPtr()->getSignDistance(position, pos_sdf, pos_normal);
				cuExecute(position.size(),
					K_ConstrainSDF,
					position,
					velocity,
					pos_sdf,
					pos_normal,
					this->varNormalFriction()->getData(),
					this->varTangentialFriction()->getData(),
					dt);
			}
		}

		auto triSys = this->getTriangularSystems();
		for (int i = 0; i < triSys.size(); i++)
		{
			DArray<Coord>& position = triSys[i]->statePosition()->getData();
			DArray<Coord>& velocity = triSys[i]->stateVelocity()->getData();

			for (int j = 0; j < aSDF.size(); j++)
			{
				aSDF[j]->stateSDFTopology()->constDataPtr()->getSignDistance(position, pos_sdf, pos_normal);
				cuExecute(position.size(),
					K_ConstrainSDF,
					position,
					velocity,
					pos_sdf,
					pos_normal,
					this->varNormalFriction()->getData(),
					this->varTangentialFriction()->getData(),
					dt);
			}
		}

		pos_normal.clear();
		pos_sdf.clear();
	}

	DEFINE_CLASS(AdaptiveBoundary);
}