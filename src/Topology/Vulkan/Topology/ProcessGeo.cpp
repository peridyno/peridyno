#include "ProcessGeo.h"

#include <set>


namespace dyno
{
	IMPLEMENT_TCLASS(ProcessGeo, TDataType)

	template<typename TDataType>
	ProcessGeo<TDataType>::ProcessGeo()
		: Module()
	{
		
	}
	template<typename TDataType>
	void ProcessGeo<TDataType>::setAllPoints()
	{
		auto geo = this->inGeoVisualSet()->getDataPtr();

		auto topo = this->outAllPointSet()->allocate();
		if (geo->allPoints.size()) {
			topo->setPoints(geo->allPoints);
		}

	}

	template<typename TDataType>
	void ProcessGeo<TDataType>::setPoints()
	{
		auto geo = this->inGeoVisualSet()->getDataPtr();
		auto topo = this->outPointSet()->allocate();
		if (geo->points.size()) {
			topo->setPoints(geo->points);
		}
		
	}

	template<typename TDataType>
	void ProcessGeo<TDataType>::setTriangles()
	{
		auto geo = this->inGeoVisualSet()->getDataPtr();

		auto topo = this->outTriangleSet()->allocate();
		if (geo->trinode.size()&& geo->triangles.size()) {
			topo->setPoints(geo->trinode);
			topo->setTriangles(geo->triangles);
			//topo->updateTriangle2Edge();
		}
	}

	template<typename TDataType>
	void ProcessGeo<TDataType>::setQuads()
	{
		auto geo = this->inGeoVisualSet()->getDataPtr();
		auto topo = this->outQuadSet()->allocate();
		if (geo->quadnode.size() && geo->quads.size()) {
			topo->setPoints(geo->quadnode);
			topo->setQuads(geo->quads);
			
		}
	}

	template<typename TDataType>
	void ProcessGeo<TDataType>::setTetrahedrons()
	{
		auto geo = this->inGeoVisualSet()->getDataPtr();
		auto topo = this->outTetrahedronSet()->allocate();
		if (geo->tetnode.size() && geo->tetrahedrons.size()) {
			topo->setPoints(geo->tetnode);
			topo->setTetrahedrons(geo->tetrahedrons);
		}
	}

	template<typename TDataType>
	void ProcessGeo<TDataType>::setHexahedrons()
	{
		auto geo = this->inGeoVisualSet()->getDataPtr();
		auto topo = this->outHexahedronSet()->allocate();
		if (geo->hexnode.size() && geo->hexahedrons.size()) {
			topo->setPoints(geo->hexnode);
			topo->setHexahedrons(geo->hexahedrons);
		}
	}

	template<typename TDataType>
	ProcessGeo<TDataType>::~ProcessGeo()
	{
	}

	template<typename TDataType>
	void ProcessGeo<TDataType>::updateImpl() 
	{
		this->setAllPoints();
		this->setPoints();
		this->setTriangles();
		this->setQuads();
		this->setTetrahedrons();
		this->setHexahedrons();

	}

	DEFINE_CLASS(ProcessGeo);
}