#include "VolumeBoundary.h"
#include "Topology/TriangleSet.h"
#include "Topology/SignedDistanceField.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(VolumeBoundary, TDataType)

	template<typename TDataType>
	VolumeBoundary<TDataType>::VolumeBoundary()
		: Volume<TDataType>()
	{
		this->varNormalFriction()->setValue(0.95);
		this->varTangentialFriction()->setValue(0.0);
	}

	template<typename TDataType>
	VolumeBoundary<TDataType>::~VolumeBoundary()
	{
	}

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_BoundaryHandling(
		GArray<Coord> posArr,
		GArray<Coord> velArr,
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
	void VolumeBoundary<TDataType>::advance(Real dt)
	{
		auto sdfModule = TypeInfo::cast<SignedDistanceField<DataType3f>>(this->getTopologyModule());
		if (sdfModule == nullptr)
		{
			Log::sendMessage(Log::Error, "Boundary: The topology module is not supported!");
			return;
		}

		auto pSys = this->getParticleSystems();

		for (int i = 0; i < pSys.size(); i++)
		{
			int pNum = pSys[i]->currentPosition()->getElementCount();
		
			cuExecute(pNum,
				K_BoundaryHandling,
				pSys[i]->currentPosition()->getValue(),
				pSys[i]->currentVelocity()->getValue(),
				sdfModule->getSDF(),
				this->varNormalFriction()->getValue(),
				this->varTangentialFriction()->getValue(),
				dt);
		}

		this->translate(Coord(0.0f, 0.0f, 0.001f));
	}

	template<typename TDataType>
	void VolumeBoundary<TDataType>::translate(Coord t)
	{
		auto topo = TypeInfo::cast<SignedDistanceField<DataType3f>>(this->getTopologyModule());
		if (topo == nullptr)
		{
			Log::sendMessage(Log::Error, "Boundary: The topology module is not supported!");
			return;
		}

		auto& sdf = topo->getSDF();

		sdf.translate(t);

		auto triTopo = TypeInfo::cast<TriangleSet<DataType3f>>(m_surfaceNode->getTopologyModule());
		if (triTopo == nullptr)
		{
			Log::sendMessage(Log::Error, "Boundary Surface: The topology module is not supported!");
			return;
		}

		triTopo->translate(t);
	}

	template<typename TDataType>
	std::shared_ptr<Node> VolumeBoundary<TDataType>::loadCube(Coord lo, Coord hi, Real distance, bool bOutBoundary /*= false*/)
	{
		auto topo = TypeInfo::cast<SignedDistanceField<DataType3f>>(this->getTopologyModule());
		if (topo == nullptr)
		{
			Log::sendMessage(Log::Error, "Boundary: The topology module is not supported!");
			return nullptr;
		}

		int nx = floor((hi[0] - lo[0]) / distance);
		int ny = floor((hi[1] - lo[1]) / distance);
		int nz = floor((hi[2] - lo[2]) / distance);

		auto& sdf = topo->getSDF();

		sdf.setSpace(lo - 5 * distance, hi + 5 * distance, nx + 10, ny + 10, nz + 10);
		sdf.loadBox(lo, hi, bOutBoundary);

		//Attach another node to represent the surface
		//Note: the size of standard cube is 2m*2m*2m
		Coord scale = (hi - lo) / 2;
		Coord center = (hi + lo) / 2;

		m_surfaceNode = this->createChild<Node>("cube");
		m_surfaceNode->setControllable(false);

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		triSet->loadObjFile("../../data/standard/standard_cube.obj");
		triSet->scale(scale);
		triSet->translate(center);

		m_surfaceNode->setTopologyModule(triSet);

		return m_surfaceNode;
	}

	template<typename TDataType>
	std::shared_ptr<Node> VolumeBoundary<TDataType>::loadSDF(std::string filename, bool bOutBoundary /*= false*/)
	{
		auto topo = TypeInfo::cast<SignedDistanceField<DataType3f>>(this->getTopologyModule());
		if (topo == nullptr)
		{
			Log::sendMessage(Log::Error, "Boundary: The topology module is not supported!");
			return nullptr;
		}

		auto& sdf = topo->getSDF();

		sdf.loadSDF(filename, bOutBoundary);

		m_surfaceNode = this->createChild<Node>("cube");
		m_surfaceNode->setControllable(false);

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		m_surfaceNode->setTopologyModule(triSet);

		return m_surfaceNode;
	}
}