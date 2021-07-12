#include "CollisionSDF.h"
#include "Node.h"
#include "Module/CollidableObject.h"
#include "Topology/DistanceField3D.h"
#include "Collision/CollidablePoints.h"
#include "Collision/CollidableSDF.h"
#include "Algorithm/CudaRand.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(CollisionSDF, TDataType)

	template<typename TDataType>
	CollisionSDF<TDataType>::CollisionSDF()
		: CollisionModel()
		, m_cSDF(nullptr)
		, m_normal_friction(0.95)
		, m_tangent_friction(0.1)
	{
	}

	template<typename TDataType>
	CollisionSDF<TDataType>::~CollisionSDF()
	{
		m_collidableObjects.clear();
	}

	template<typename TDataType>
	bool CollisionSDF<TDataType>::isSupport(std::shared_ptr<CollidableObject> obj)
	{
		if (obj->getType() == CollidableObject::POINTSET_TYPE)
		{
			return true;
		}

		return false;
	}

	template<typename TDataType>
	void CollisionSDF<TDataType>::addDrivenObject(std::shared_ptr<CollidableObject> obj)
	{
		if (obj->getType() == CollidableObject::POINTSET_TYPE)
		{
			m_collidableObjects.push_back(obj);
		}
	}

	template<typename TDataType>
	void CollisionSDF<TDataType>::addCollidableObject(std::shared_ptr<CollidableObject> obj)
	{
		if (obj->getType() == CollidableObject::POINTSET_TYPE)
			addDrivenObject(obj);

		if (obj->getType() == CollidableObject::SIGNED_DISTANCE_TYPE)
			setCollidableSDF(obj);
	}

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_ConstrainParticles(
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
			vec = vec_normal + vec_tan;
			vec *= pow(Real(M_E), -dt*tangentialFriction);
		}

		posArr[pId] = pos;
		velArr[pId] = vec;
	}


	template<typename TDataType>
	void CollisionSDF<TDataType>::doCollision()
	{
		if (!m_cSDF || m_collidableObjects.size() <= 0)
		{
			std::cout << "Collidable objects are not correctly set!" << std::endl;
			return;
		}

		auto sdf = m_cSDF->getSDF();

		for (int i = 0; i < m_collidableObjects.size(); i++)
		{
			if (m_collidableObjects[i]->getType() == CollidableObject::POINTSET_TYPE)
			{
				std::shared_ptr<CollidablePoints<TDataType>> cPoints = std::dynamic_pointer_cast<CollidablePoints<TDataType>>(m_collidableObjects[i]);
				DArray<Coord>& pos = cPoints->getPositions();
				DArray<Coord>& vel = cPoints->getVelocities();

				cPoints->updateCollidableObject();

				uint pDim = cudaGridSize(pos.size(), BLOCK_SIZE);
				K_ConstrainParticles << <pDim, BLOCK_SIZE >> > (
					pos,
					vel,
					*sdf,
					m_normal_friction,
					m_tangent_friction,
					getParent()->getDt());

				cPoints->updateMechanicalState();
 			}
		}


	}


	template<typename TDataType>
	bool CollisionSDF<TDataType>::initializeImpl()
	{
		for (int i = 0; i < m_collidableObjects.size(); i++)
		{
			m_collidableObjects[i]->initialize();
		}

		return true;
	}

	DEFINE_CLASS(CollisionSDF);
}