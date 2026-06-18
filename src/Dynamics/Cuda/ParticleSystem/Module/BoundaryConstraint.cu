#include "BoundaryConstraint.h"
#include "Log.h"
#include "Node.h"
#include "Algorithm/CudaRand.h"
#include "Topology/DistanceField3D.h"

namespace dyno
{
	//IMPLEMENT_TCLASS(BoundaryConstraint, TDataType)

	template<typename TDataType>
	BoundaryConstraint<TDataType>::BoundaryConstraint()
		: ConstraintModule()
	{
		Coord lo(0.0f);
		Coord hi(1.0f);
	}

	template<typename TDataType>
	BoundaryConstraint<TDataType>::~BoundaryConstraint()
	{
		m_position.clear();
		m_velocity.clear();
	}

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_ConstrainSDF(
		DArray<Coord> posArr,
		DArray<Coord> velArr,
		DistanceField3D<TDataType> df,
		Real normalFriction,
		Real tangentialFriction,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos = posArr[pId];
		Coord vec = velArr[pId];

		Real dist;
		Coord normal;
		df.getDistanceAndNormal(pos, dist, normal);
		// constrain particle
		if (dist <= 0) {
			Real olddist = -dist;
			RandNumber rGen(pId);
			dist = 0.0001f*rGen.Generate();
			// reflect position
			pos += (olddist + dist)*normal;
			// reflect velocity
			Real vlength = vec.norm();
			Real vec_n = vec.dot(normal);
			Coord vec_normal = vec_n*normal;
			Coord vec_tan = vec - vec_normal;
			if (vec_n < 0) vec_normal = -vec_normal;
			vec_normal *= (1.0f - normalFriction);
			vec = vec_normal + vec_tan * (1.0f - tangentialFriction);
		}

		posArr[pId] = pos;
		velArr[pId] = vec;
	}

	template<typename TDataType>
	void BoundaryConstraint<TDataType>::constrain()
	{
		uint pDim = cudaGridSize(m_position.size(), BLOCK_SIZE);
		K_ConstrainSDF << <pDim, BLOCK_SIZE >> > (
			m_position.getData(),
			m_velocity.getData(),
			*m_cSDF,
			this->varNormalFriction()->getData(),
			this->varTangentialFriction()->getData(),
			this->getParentNode()->getDt());
	}

	template<typename TDataType>
	bool BoundaryConstraint<TDataType>::constrain(DArray<Coord>& position, DArray<Coord>& velocity, Real dt)
	{
		uint pDim = cudaGridSize(position.size(), BLOCK_SIZE);
 		K_ConstrainSDF << <pDim, BLOCK_SIZE >> > (
			position,
			velocity,
			*m_cSDF,
			this->varNormalFriction()->getData(),
			this->varTangentialFriction()->getData(),
			dt);

		return true;
	}

	template<typename TDataType>
	void BoundaryConstraint<TDataType>::constrain(DArray<Coord>& position, DArray<Coord>& velocity, DistanceField3D<TDataType>& sdf, Real dt)
	{
		uint pDim = cudaGridSize(position.size(), BLOCK_SIZE);
		K_ConstrainSDF << <pDim, BLOCK_SIZE >> > (
			position,
			velocity,
			sdf,
			this->varNormalFriction()->getData(),
			this->varTangentialFriction()->getData(),
			dt);
	}

	DEFINE_CLASS(BoundaryConstraint);
}