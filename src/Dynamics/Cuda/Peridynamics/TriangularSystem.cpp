#include "TriangularSystem.h"
#include "Topology/PointSet.h"
#include "Primitive/Primitive3D.h"
#include "Topology/TetrahedronSet.h"
#include "Module/FixedPoints.h"


#include "Smesh_IO/smesh.h"
#include "Gmsh_IO/gmsh.h"

namespace dyno
{
	IMPLEMENT_TCLASS(TriangularSystem, TDataType)
	
	template<typename TDataType>
	TriangularSystem<TDataType>::TriangularSystem()
		: Node()
	{
		//Create a node for surface mesh rendering
		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		this->stateTriangleSet()->setDataPtr(triSet);

		this->FixedIds.allocate();
		this->FixedPos.allocate();

		auto m_fixed = std::make_shared<FixedPoints<TDataType>>();
		this->statePosition()->connect(m_fixed->inPosition());
		this->stateVelocity()->connect(m_fixed->inVelocity());
		this->FixedIds.connect(&m_fixed->FixedIds);
		this->FixedPos.connect(&m_fixed->FixedPos);
		//this->animationPipeline()->pushModule(m_fixed);
		this->animationPipeline()->pushModule(m_fixed);
	}

	template<typename TDataType>
	TriangularSystem<TDataType>::~TriangularSystem()
	{
	}

	template<typename TDataType>
	void TriangularSystem<TDataType>::updateTopology()
	{
		auto triSet = this->stateTriangleSet()->getDataPtr();

		triSet->getPoints().assign(this->statePosition()->getData());
	}

// 	template<typename TDataType>
// 	std::shared_ptr<Node> TriangleSystem<TDataType>::getSurface()
// 	{
// 		return mSurfaceNode;
// 	}
	
	template<typename TDataType>
	void TriangularSystem<TDataType>::loadSurface(std::string filename)
	{
		this->stateTriangleSet()->getDataPtr()->loadObjFile(filename);
	}

	template<typename TDataType>
	void TriangularSystem<TDataType>::addFixedParticle(int id, Coord pos)
	{
		m_fixedIds.push_back(id);
		m_fixedPos.push_back(pos);
	}

	template<typename TDataType>
	bool TriangularSystem<TDataType>::translate(Coord t)
	{
		auto triSet = this->stateTriangleSet()->getDataPtr();
		triSet->translate(t);

		return true;
	}

	template<typename TDataType>
	bool dyno::TriangularSystem<TDataType>::scale(Real s)
	{
		auto triSet = this->stateTriangleSet()->getDataPtr();
		triSet->scale(s);

		return true;
	}

	template<typename TDataType>
	void TriangularSystem<TDataType>::resetStates()
	{
		auto triSet = this->stateTriangleSet()->getDataPtr();
		if (triSet == nullptr) return;

		printf("m_fixedPos size = %d\n", m_fixedIds.size());
		this->FixedIds.resize(m_fixedIds.size());
		this->FixedPos.resize(m_fixedIds.size());
		
		this->FixedIds.getData().assign(m_fixedIds);
		this->FixedPos.getData().assign(m_fixedPos);
		
		auto& pts = triSet->getPoints();

		if (pts.size() > 0)
		{
			this->statePosition()->resize(pts.size());
			this->stateVelocity()->resize(pts.size());
			this->stateForce()->resize(pts.size());

			this->statePosition()->getData().assign(pts);
			this->stateVelocity()->getDataPtr()->reset();
		}

		Node::resetStates();
	}

	DEFINE_CLASS(TriangularSystem);
}