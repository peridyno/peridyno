#include "VolumeBoundary.h"
#include "Topology/TriangleSet.h"
#include "Topology/SignedDistanceField.h"

#include "ParticleSystem/Module/BoundaryConstraint.h"

namespace dyno
{
	template<typename TDataType>
	VolumeBoundary<TDataType>::VolumeBoundary()
		: Node()
	{
		this->varNormalFriction()->setValue(0.95);
		this->varTangentialFriction()->setValue(0.95);
	}

	template<typename TDataType>
	VolumeBoundary<TDataType>::~VolumeBoundary()
	{
	}

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_BoundaryHandling(
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
			//RandNumber rGen(pId);
			dist = 0.0001f;//*rGen.Generate();
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

	Real total_lifting = 0.0f;
	template<typename TDataType>
	void VolumeBoundary<TDataType>::updateVolume()
	{
		Real dt = this->stateTimeStep()->getValue();

// 		auto sdfModule = TypeInfo::cast<SignedDistanceField<DataType3f>>(this->stateTopology()->getDataPtr());
// 		if (sdfModule == nullptr)
// 		{
// 			Log::sendMessage(Log::Error, "Boundary: The topology module is not supported!");
// 			return;
// 		}

		for (size_t t = 0; t < m_obstacles.size(); t++)
		{
			auto pSys = this->getParticleSystems();

			for (int i = 0; i < pSys.size(); i++)
			{
				auto posFd = pSys[i]->statePosition();
				auto velFd = pSys[i]->stateVelocity();

				if(!posFd->isEmpty() && !velFd->isEmpty())
					m_obstacles[t]->constrain(posFd->getData(), velFd->getData(), dt);
			}

			auto triSys = this->getTriangularSystems();
			for (int i = 0; i < triSys.size(); i++)
			{
				auto posFd = triSys[i]->statePosition();
				auto velFd = triSys[i]->stateVelocity();

				if (!posFd->isEmpty() && !velFd->isEmpty())
					m_obstacles[t]->constrain(posFd->getData(), velFd->getData(), dt);
			}

			auto tetSys = this->getTetrahedralSystems();
			for (int i = 0; i < tetSys.size(); i++)
			{
				auto posFd = tetSys[i]->statePosition();
				auto velFd = tetSys[i]->stateVelocity();

				if (!posFd->isEmpty() && !velFd->isEmpty())
					m_obstacles[t]->constrain(posFd->getData(), velFd->getData(), dt);
			}
		}
	}

	template<typename TDataType>
	void VolumeBoundary<TDataType>::updateStates()
	{
		updateVolume();
	}

	template<typename TDataType>
	void VolumeBoundary<TDataType>::translate(Coord t)
	{
		auto topo = TypeInfo::cast<SignedDistanceField<DataType3f>>(this->stateTopology()->getDataPtr());
		if (topo == nullptr)
		{
			Log::sendMessage(Log::Error, "Boundary: The topology module is not supported!");
			return;
		}

		auto& sdf = topo->getSDF();

		sdf.translate(t);
	}

	template<typename TDataType>
	std::shared_ptr<Node> VolumeBoundary<TDataType>::loadCube(Coord lo, Coord hi, Real distance, bool bOutBoundary /*= false*/)
	{
		auto boundary = std::make_shared<BoundaryConstraint<TDataType>>();
		boundary->setCube(lo, hi, distance, bOutBoundary);

		this->varNormalFriction()->connect(boundary->varNormalFriction());
		this->varTangentialFriction()->connect(boundary->varTangentialFriction());

		m_obstacles.push_back(boundary);

		//Note: the size of standard cube is 2m*2m*2m
		Coord scale = (hi - lo) / 2;
		Coord center = (hi + lo) / 2;

		return nullptr;
	}

	template<typename TDataType>
	std::shared_ptr<Node> VolumeBoundary<TDataType>::loadSDF(std::string filename, bool bOutBoundary /*= false*/)
	{
		
		if (this->stateTopology()->isEmpty())
		{
			auto sdf_ptr = std::make_shared<SignedDistanceField<DataType3f>>();
			this->stateTopology()->setDataPtr(sdf_ptr);
		}
		
		auto topo = TypeInfo::cast<SignedDistanceField<DataType3f>>(this->stateTopology()->getDataPtr());


		if (topo == nullptr)
		{
			
			Log::sendMessage(Log::Error, "Boundary: The topology module is not supported!");
			return nullptr;
		}

		auto& sdf = topo->getSDF();

		sdf.loadSDF(filename, bOutBoundary);

		auto boundary = std::make_shared<BoundaryConstraint<TDataType>>();
		boundary->load(filename, bOutBoundary);
		this->varNormalFriction()->connect(boundary->varNormalFriction());
		this->varTangentialFriction()->connect(boundary->varTangentialFriction());
		m_obstacles.push_back(boundary);
		return nullptr;
	}


	template<typename TDataType>
	void VolumeBoundary<TDataType>::loadShpere(Coord center, Real r, Real distance /*= 0.005f*/, bool bOutBoundary /*= false*/, bool bVisible /*= false*/)
	{
		auto boundary = std::make_shared<BoundaryConstraint<TDataType>>();
		boundary->setSphere(center, r, distance, bOutBoundary);

		this->varNormalFriction()->connect(boundary->varNormalFriction());
		this->varTangentialFriction()->connect(boundary->varTangentialFriction());

		m_obstacles.push_back(boundary);
	}


	DEFINE_CLASS(VolumeBoundary);
}