#include "GeoVisualSet.h"


namespace dyno
{
	IMPLEMENT_TCLASS(GeoVisualSet, TDataType)

	template<typename TDataType>
	GeoVisualSet<TDataType>::GeoVisualSet()
		: Module()
	{
	}

	template<typename TDataType>
	void GeoVisualSet<TDataType>::setPoints(std::vector<Coord>& points)
	{
		if (points.size() != 0) {
			this->points=points;
			this->allPoints.insert(this->allPoints.end(), points.begin(), points.end());

		}
	}

	template<typename TDataType>
	void GeoVisualSet<TDataType>::setTriangles(std::vector<Coord>& node ,std::vector<Triangle>& triangles)
	{
		if (triangles.size() != 0) {
			this->trinode = node;
			this->triangles = triangles;
			this->allPoints.insert(this->allPoints.end(), node.begin(), node.end());
		}
	}

	template<typename TDataType>
	void GeoVisualSet<TDataType>::setQuads(std::vector<Coord>& node, std::vector<Quad>& quads)
	{
		if (quads.size() != 0) {
			this->quadnode=node;
			this->quads = quads;
			this->allPoints.insert(this->allPoints.end(), node.begin(), node.end());
		}
	}

	template<typename TDataType>
	void GeoVisualSet<TDataType>::setTetrahedrons(std::vector<Coord>& node, std::vector<Tetrahedron>& tetrahedrons)
	{
		if (tetrahedrons.size() != 0) {
			this->tetnode = node;
			this->tetrahedrons = tetrahedrons;
			this->allPoints.insert(this->allPoints.end(), node.begin(), node.end());
		}
	}

	template<typename TDataType>
	void GeoVisualSet<TDataType>::setHexahedrons(std::vector<Coord>& node, std::vector<Hexahedron>& hexahedrons)
	{
		if (hexahedrons.size() != 0) {
			this->hexnode = node;
			this->hexahedrons = hexahedrons;
			this->allPoints.insert(this->allPoints.end(), node.begin(), node.end());
		}
	}


	template<typename TDataType>
	GeoVisualSet<TDataType>::~GeoVisualSet()
	{
	}

	DEFINE_CLASS(GeoVisualSet);
}