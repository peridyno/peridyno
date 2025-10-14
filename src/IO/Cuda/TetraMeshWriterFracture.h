/*
This Module is designed to output surface mesh file of TetrahedronSet;
the output file format: obj
*/

#pragma once
#include "Module/OutputModule.h"
#include "Module/TopologyModule.h"

#include "Topology/TetrahedronSet.h"

#include <string>
#include <memory>

namespace dyno
{

	template <typename TDataType> class TriangleSet;
	template <typename TDataType> class TetrahedronSet;


	struct OriginalFaceId
	{
		int vertexId1, vertexId2, vertexId3;
		int uvId1, uvId2, uvId3;
		int maxVertexID()
		{
			return maximum(maximum(vertexId1, vertexId2), vertexId3);
		}
		int minVertexID()
		{
			return minimum(minimum(vertexId1, vertexId2), vertexId3);
		}
		int sumVertexID()
		{
			return vertexId1 + vertexId2 + vertexId3;
		}
	};


	template<typename TDataType>
	class TetraMeshWriterFracture : public OutputModule
	{
		DECLARE_TCLASS(TetraMeshWriterFracture, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename TopologyModule::Tetrahedron Tetrahedron;
		typedef typename TopologyModule::Tri2Tet Tri2Tet;

		TetraMeshWriterFracture();
		~TetraMeshWriterFracture();

		void loadUVs(std::string path);

		void setTetrahedronSetPtr(std::shared_ptr<TetrahedronSet<TDataType>> ptr_tets) { this->ptr_TetrahedronSet = ptr_tets;  this->updatePtr(); }
		bool updatePtr();

		bool outputSurfaceMesh();

		DArray<int>* OringalID;

	protected:

	public:


	protected:

		std::string file_postfix = ".obj";

		DArray<Triangle>* ptr_triangles;
		DArray<Tri2Tet>* ptr_tri2tet;
		DArray<Coord>* ptr_vertices;
		DArray<Tetrahedron>* ptr_tets;
		std::shared_ptr<TetrahedronSet<TDataType>> ptr_TetrahedronSet;

		int onInitSurface(Triangle Tri);


		bool first = true;
		
		
		CArray<OriginalFaceId> FaceId;
		CArray<int> FaceStart;
		CArray<int> FaceMapping;
		CArray<int> onFace;

		std::vector<Coord> vnList;
	};
}