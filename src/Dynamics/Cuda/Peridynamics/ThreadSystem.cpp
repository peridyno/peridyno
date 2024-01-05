#include "ThreadSystem.h"
#include "Topology/PointSet.h"
#include "Primitive/Primitive3D.h"
#include "Topology/TetrahedronSet.h"


#include "Smesh_IO/smesh.h"
#include "Gmsh_IO/gmsh.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ThreadSystem, TDataType)

	template<typename TDataType>
	ThreadSystem<TDataType>::ThreadSystem()
		: Node()
	{
		auto topo = std::make_shared<EdgeSet<TDataType>>();
		this->stateEdgeSet()->setDataPtr(topo);
	}

	template<typename TDataType>
	ThreadSystem<TDataType>::~ThreadSystem()
	{
	}

	template<typename TDataType>
	void ThreadSystem<TDataType>::updateTopology()
	{
		auto edgeSet = TypeInfo::cast<EdgeSet<TDataType>>(this->stateEdgeSet()->getDataPtr());
		if (edgeSet == nullptr) return;

		if (!this->statePosition()->isEmpty())
		{
			int num = this->statePosition()->size();
			auto& pts = edgeSet->getPoints();
			if (num != pts.size())
			{
				pts.resize(num);
			}

			//Function1Pt::copy(pts, this->statePosition()->getData());
			pts.assign(this->statePosition()->getData(), this->statePosition()->getData().size());
		}
	}

	template<typename TDataType>
	void ThreadSystem<TDataType>::addThread(Coord start, Coord end, int segSize)
	{
		int idx1;
		idx1 = particles.size();
		Real length = (end - start).norm() / segSize;
		Coord dir = (end - start) / (end - start).norm();
		for (int i = 0; i < segSize; i++)
		{
			particles.push_back(Coord(start + i * length * dir));
			edges.push_back(TopologyModule::Edge(idx1, idx1 + 1));
			idx1++;
		}
		particles.push_back(end);
	}


	template<typename TDataType>
	void ThreadSystem<TDataType>::resetStates()
	{
		auto edgeSet = TypeInfo::cast<EdgeSet<TDataType>>(this->stateEdgeSet()->getDataPtr());
		if (edgeSet == nullptr) return;

		
		edgeSet->setPoints(particles);
		
		auto& dEdge = edgeSet->getEdges();
		dEdge.resize(edges.size());
		dEdge.assign(edges);

		edgeSet->getVer2Edge();

		auto& pts = edgeSet->getPoints();

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

	DEFINE_CLASS(ThreadSystem);
}