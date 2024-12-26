#include "TriangularSystem.h"
#include "Topology/PointSet.h"
#include "Primitive/Primitive3D.h"
#include "Topology/TetrahedronSet.h"
#include "Module/FixedPoints.h"


#include "Smesh_IO/smesh.h"
#include "Gmsh_IO/gmsh.h"

namespace dyno
{
	template<typename TDataType>
	TriangularSystem<TDataType>::TriangularSystem()
		: Node()
	{
		//Create a node for surface mesh rendering
		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		this->stateTriangleSet()->setDataPtr(triSet);
	}

	template<typename TDataType>
	TriangularSystem<TDataType>::~TriangularSystem()
	{
	}

	template<typename TDataType>
	void TriangularSystem<TDataType>::loadSurface(std::string filename)
	{
		this->stateTriangleSet()->getDataPtr()->loadObjFile(filename);
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

		auto& pts = triSet->getPoints();

		if (pts.size() > 0)
		{
			this->statePosition()->resize(pts.size());
			this->stateVelocity()->resize(pts.size());

			this->statePosition()->getData().assign(pts);
			this->stateVelocity()->getDataPtr()->reset();
		}

		Node::resetStates();
	}

	template<typename TDataType>
	void TriangularSystem<TDataType>::postUpdateStates()
	{
		auto triSet = this->stateTriangleSet()->getDataPtr();

		triSet->getPoints().assign(this->statePosition()->getData());
	}

	DEFINE_CLASS(TriangularSystem);
}