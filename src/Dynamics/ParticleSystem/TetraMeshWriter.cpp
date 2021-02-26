#include "TetraMeshWriter.h"
#include "Framework/ModuleIO.h"
#include "Utility/Function1Pt.h"

#include <sstream>
#include <iostream>
#include <fstream>

namespace dyno
{
	IMPLEMENT_CLASS_1(TetraMeshWriter, TDataType)

	template<typename TDataType>
	TetraMeshWriter<TDataType>::TetraMeshWriter() : IOModule()
	{

	}

	template<typename TDataType>
	TetraMeshWriter<TDataType>::~TetraMeshWriter()
	{
	}

	template<typename TDataType>
	void TetraMeshWriter<TDataType>::setNamePrefix(std::string prefix)
	{
		this->name_prefix = prefix;
	}

	template<typename TDataType>
	void TetraMeshWriter<TDataType>::setOutputPath(std::string path)
	{
		this->output_path = path;
	}

	template<typename TDataType>
	bool TetraMeshWriter<TDataType>::updatePtr() 
	{
		if (this->ptr_TetrahedronSet == nullptr) {
			return false;
		}
		this->ptr_triangles = this->ptr_TetrahedronSet->getTriangles();
		this->ptr_tri2tet = &( this->ptr_TetrahedronSet->getTri2Tet() );
		this->ptr_vertices = &( this->ptr_TetrahedronSet->getPoints() );
	}

	template<typename TDataType>
	bool TetraMeshWriter<TDataType>::outputSurfaceMesh() 
	{
		if (this->ptr_tri2tet == nullptr || this->ptr_triangles == nullptr || this->ptr_vertices == nullptr) {
			printf("------Tetra Mesh Writer: array nullptr \n");
			return false;
		}

		std::stringstream ss; ss << m_output_index;
		std::string filename = output_path + "/" + this->name_prefix + ss.str() + this->file_postfix;
		std::ofstream output(filename.c_str(), std::ios::out);

		if (!output.is_open()) {
			printf("------Tetra Mesh Writer: open file failed \n");
			return false;
		}

		this->updatePtr();

		CArray<Coord> host_vertices;
		CArray<Triangle> host_triangles;
		CArray<Tri2Tet> host_tri2tet;

		host_vertices.resize( (*(this->ptr_vertices)).size() );
		host_triangles.resize( (*(this->ptr_triangles)).size() );
		host_tri2tet.resize( (*(this->ptr_tri2tet)).size() );

		Function1Pt::copy(host_vertices, *(this->ptr_vertices));
		Function1Pt::copy(host_triangles, *(this->ptr_triangles));
		Function1Pt::copy(host_tri2tet, *(this->ptr_tri2tet));

		for (int i = 0; i < host_vertices.size(); ++i) {
			output << "v " << host_vertices[i][0] << " " << host_vertices[i][1] << " " << host_vertices[i][2] << std::endl;
		}
		for (int i = 0; i < host_tri2tet.size(); ++i) {
			Tri2Tet tmp = host_tri2tet[i];
			bool isOnSurface = false;
			if (tmp[0] < 0 || tmp[1] < 0) { isOnSurface = true; }
			if (isOnSurface) {
				output << "f " << host_triangles[i][0] << " " << host_triangles[i][1] << " " << host_triangles[i][2] << std::endl;
			}
		}

		host_vertices.release();
		host_triangles.release();
		host_tri2tet.release();

		this->m_output_index ++;
		return true;
	}

	template<typename TDataType>
	bool TetraMeshWriter<TDataType>::execute() 
	{
		printf("===========Tetra Mesh Writer============\n");

		if (this->m_output_index >= this->max_output_files) { return true; }

		if (this->current_idle_frame <= 0) {
			this->current_idle_frame = this->idle_frame_num;
			return this->outputSurfaceMesh();
		}
		else {
			this->current_idle_frame--;
		}
	}
}