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

	template<typename TDataType>
	class TetraMeshWriter : public OutputModule
	{
		DECLARE_TCLASS(TetraMeshWriter, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename TopologyModule::Tetrahedron Tetrahedron;
		typedef typename TopologyModule::Tri2Tet Tri2Tet;

		TetraMeshWriter();
		~TetraMeshWriter();

		void setNamePrefix(std::string prefix);
		void setOutputPath(std::string path);

		void setTetrahedronSetPtr(std::shared_ptr<TetrahedronSet<TDataType>> ptr_tets) { this->ptr_TetrahedronSet = ptr_tets;  this->updatePtr(); }
		bool updatePtr();

		bool outputSurfaceMesh();

	protected:
		void updateImpl() override;

	public:


	protected:
		int m_output_index = 0;
		int max_output_files = 10000;
		int idle_frame_num = 9;		//output one file of [num]+1 frames
		int current_idle_frame = 0;
		std::string output_path = "D:/Model/tmp";
		std::string name_prefix = "defaut_";
		std::string file_postfix = ".obj";

		DArray<Triangle>* ptr_triangles;
		DArray<Tri2Tet>* ptr_tri2tet;
		DArray<Coord>* ptr_vertices;
		std::shared_ptr<TetrahedronSet<TDataType>> ptr_TetrahedronSet;
	};
}