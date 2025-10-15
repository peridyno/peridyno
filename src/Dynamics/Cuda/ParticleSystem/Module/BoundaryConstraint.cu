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
		m_cSDF = std::make_shared<DistanceField3D<DataType3f>>();
		m_cSDF->setSpace(lo - 0.025f, hi + 0.025f, Real(0.005));
		m_cSDF->loadBox(lo, hi, true);
	}

	template<typename TDataType>
	BoundaryConstraint<TDataType>::~BoundaryConstraint()
	{
		m_cSDF->release();
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
		df.getDistance(pos, dist, normal);
		// constrain particle
		if (dist <= 0) {
			Real olddist = -dist;
			RandNumber rGen(pId);
			dist = 0.0001f*rGen.Generate();
			// reflect position
			pos -= (olddist + dist)*normal;
			// reflect velocity
			Real vlength = vec.norm();
			Real vec_n = vec.dot(normal);
			Coord vec_normal = vec_n*normal;
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

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_ConstrainSDF_With_Coupling(
		DArray<Coord> posArr,
		DArray<Coord> velArr,
		DistanceField3D<TDataType> df,
		Real normalFriction,
		Real tangentialFriction,
		Real dt,
		Real particleMass,
		Coord rigidBodyCom,
		DArray<Coord> totalImpulse)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos = posArr[pId];
		Coord vel_initial = velArr[pId];

		Real dist;
		Coord normal;
		df.getDistance(pos, dist, normal);
		// constrain particle
		if (dist <= 0) {
			Real olddist = -dist;
			RandNumber rGen(pId);
			dist = 0.0001f * rGen.Generate();
			// reflect position
			pos -= (olddist + dist) * normal;
			// reflect velocity
			Coord vec_final = vel_initial;
			Real vlength = vec_final.norm();
			Real vec_n = vec_final.dot(normal);
			Coord vec_normal = vec_n * normal;
			Coord vec_tan = vec_final - vec_normal;
			if (vec_n > 0) vec_normal = -vec_normal;
			vec_normal *= (1.0f - normalFriction);
			vec_final = vec_normal + vec_tan * (1.0f - tangentialFriction);

			posArr[pId] = pos;
			velArr[pId] = vec_final;

			Coord impulse = (vel_initial - vec_final) * particleMass;

			Coord r = pos - rigidBodyCom;

			Coord torque = r.cross(impulse);

			atomicAdd(&totalImpulse[0].x, impulse.x);
			atomicAdd(&totalImpulse[0].y, impulse.y);
			atomicAdd(&totalImpulse[0].z, impulse.z);

			atomicAdd(&totalImpulse[1].x, torque.x);
			atomicAdd(&totalImpulse[1].y, torque.y);
			atomicAdd(&totalImpulse[1].z, torque.z);
		}

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
		/*
		K_ConstrainSDF << <pDim, BLOCK_SIZE >> > (
			position,
			velocity,
			sdf,
			this->varNormalFriction()->getData(),
			this->varTangentialFriction()->getData(),
			dt);
		*/
		Real particleMass = 0.001;
		CArray<Coord> impulse_host;
		impulse_host.resize(2);
		impulse_host.reset();
		DArray<Coord> impulse;
		impulse.assign(impulse_host);
		Coord com = (sdf.lowerBound() + sdf.upperBound()) / 2;
		
		K_ConstrainSDF_With_Coupling << <pDim, BLOCK_SIZE >> > (
			position,
			velocity,
			sdf,
			this->varNormalFriction()->getData(),
			this->varTangentialFriction()->getData(),
			dt,
			particleMass,
			com,
			impulse);

		impulse_host.assign(impulse);
		for (int i = 0; i < 2; i++) {
			std::cout << impulse_host[i].x << " " << impulse_host[i].y << " " << impulse_host[i].z << std::endl;
		}
	}

	template<typename TDataType>
	void BoundaryConstraint<TDataType>::load(std::string filename, bool inverted)
	{
		m_cSDF->loadSDF(filename, inverted);
	}


	template<typename TDataType>
	void BoundaryConstraint<TDataType>::setCube(Coord lo, Coord hi, Real distance, bool inverted)
	{
		int nx = floor((hi[0] - lo[0]) / distance);
		int ny = floor((hi[1] - lo[1]) / distance);
		int nz = floor((hi[2] - lo[2]) / distance);

		uint padding = 5;

		m_cSDF->setSpace(lo - padding *distance, hi + padding * distance, distance);
		m_cSDF->loadBox(lo, hi, inverted);
	}


	template<typename TDataType>
	void BoundaryConstraint<TDataType>::setSphere(Coord center, Real r, Real distance, bool inverted)
	{
		int nx = floor(2 * r / distance);

		uint padding = 5;

		m_cSDF->setSpace(center - r - padding * distance, center + r + padding * distance, distance);
		m_cSDF->loadSphere(center, r, inverted);
	}

	template<typename TDataType>
	void BoundaryConstraint<TDataType>::setCylinder(Coord center, Real r, Real height, Real distance, int axis, bool inverted)
	{
		int nx = floor(2 * r / distance);
		int ny = floor(0.5 * height / distance);

		uint padding = 5;

		m_cSDF->setSpace(center - height - r - padding * distance, center + height  + r + padding * distance, distance);
		m_cSDF->loadCylinder(center, r, height, axis, inverted);
	}


	DEFINE_CLASS(BoundaryConstraint);
}