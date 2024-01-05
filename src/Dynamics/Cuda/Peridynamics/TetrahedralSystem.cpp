#include "TetrahedralSystem.h"

#include "Primitive/Primitive3D.h"

#include "Topology/PointSet.h"
#include "Topology/TetrahedronSet.h"

#include "Smesh_IO/smesh.h"
#include "Gmsh_IO/gmsh.h"

namespace dyno
{
	IMPLEMENT_TCLASS(TetrahedralSystem, TDataType)

	template<typename TDataType>
	TetrahedralSystem<TDataType>::TetrahedralSystem()
		: Node()
	{
		auto topo = std::make_shared<TetrahedronSet<TDataType>>();
		this->stateTetrahedronSet()->setDataPtr(topo);
	}

	template<typename TDataType>
	TetrahedralSystem<TDataType>::~TetrahedralSystem()
	{
	}

	template<typename TDataType>
	void TetrahedralSystem<TDataType>::loadVertexFromGmshFile(std::string filename)
	{
		Gmsh meshLoader;
		meshLoader.loadFile(filename);

		auto tetSet = TypeInfo::cast<TetrahedronSet<TDataType>>(this->stateTetrahedronSet()->getDataPtr());

	
		tetSet->setPoints(meshLoader.m_points);
		tetSet->setTetrahedrons(meshLoader.m_tets);
		tetSet->update();
	}

	template<typename TDataType>
	void TetrahedralSystem<TDataType>::loadVertexFromFile(std::string filename)
	{
		Smesh meshLoader;
		meshLoader.loadNodeFile(filename + ".node");
		//meshLoader.loadTriangleFile(filename + ".face");
		meshLoader.loadTetFile(filename + ".ele");

		auto tetSet = TypeInfo::cast<TetrahedronSet<TDataType>>(this->stateTetrahedronSet()->getDataPtr());
		
		tetSet->setPoints(meshLoader.m_points);
		tetSet->setTetrahedrons(meshLoader.m_tets);
		tetSet->update();
	}

	template<typename TDataType>
	void TetrahedralSystem<TDataType>::updateTopology()
	{
		auto tetSet = TypeInfo::cast<TetrahedronSet<TDataType>>(this->stateTetrahedronSet()->getDataPtr());
		if (tetSet == nullptr) return;

		if (!this->statePosition()->isEmpty())
		{
			int num = this->statePosition()->size();
			auto& pts = tetSet->getPoints();
			if (num != pts.size())
			{
				pts.resize(num);
			}

			//Function1Pt::copy(pts, this->statePosition()->getData());
			pts.assign(this->statePosition()->getData(), this->statePosition()->getData().size());
		}
	}


	template<typename TDataType>
	void TetrahedralSystem<TDataType>::resetStates()
	{
		auto tetSet = this->stateTetrahedronSet()->getDataPtr();
		if (tetSet == nullptr) return;

		auto& pts = tetSet->getPoints();

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

	template<typename TDataType>
	bool TetrahedralSystem<TDataType>::translate(Coord t)
	{
		auto ptSet = this->stateTetrahedronSet()->getDataPtr();
		ptSet->translate(t);

		return true;
	}


	template<typename TDataType>
	bool TetrahedralSystem<TDataType>::scale(Real s)
	{
		auto ptSet = this->stateTetrahedronSet()->getDataPtr();
		ptSet->scale(s);

		return true;
	}


	template<typename TDataType>
	bool TetrahedralSystem<TDataType>::rotate(Quat<Real> q)
	{
		auto ptSet = this->stateTetrahedronSet()->getDataPtr();
		ptSet->rotate(q);

		return true;
	}

	DEFINE_CLASS(TetrahedralSystem);
}