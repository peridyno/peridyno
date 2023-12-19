#include "TetraMeshWriter.h"
#include "Module/OutputModule.h"

#include <sstream>
#include <iostream>
#include <fstream>

namespace dyno
{
	IMPLEMENT_TCLASS(TetraMeshWriter, TDataType)

	template<typename TDataType>
	TetraMeshWriter<TDataType>::TetraMeshWriter() : OutputModule()
	{

	}

	template<typename TDataType>
	TetraMeshWriter<TDataType>::~TetraMeshWriter()
	{
	}


	template<typename TDataType>
	bool TetraMeshWriter<TDataType>::updatePtr() 
	{
		if (this->ptr_TetrahedronSet == nullptr) {
			return false;
		}
		this->ptr_triangles = &( this->ptr_TetrahedronSet->getTriangles() );
		this->ptr_tri2tet = &( this->ptr_TetrahedronSet->getTri2Tet() );
		this->ptr_vertices = &( this->ptr_TetrahedronSet->getPoints() );
		this->ptr_tets = &( this->ptr_TetrahedronSet->getTetrahedrons() );
	}

	template<typename TDataType>
	void TetraMeshWriter<TDataType>::output() 
	{
		if (this->ptr_tri2tet == nullptr || this->ptr_triangles == nullptr || this->ptr_vertices == nullptr) {
			printf("------Tetra Mesh Writer: array nullptr \n");
			return;
		}

		auto output_path = this->varOutputPath()->getValue();
		int frame_number = this->getFrameNumber();
		std::stringstream ss; ss << frame_number;
		std::string filename = output_path + ss.str() + this->file_postfix;
		std::ofstream output(filename.c_str(), std::ios::out);

		if (!output.is_open()) {
			printf("------Tetra Mesh Writer: open file failed \n");
			return;
		}

		printf("Output Surface!!!!!!!!\n");

		this->updatePtr();

		CArray<Coord> host_vertices;
		CArray<Triangle> host_triangles;
		CArray<Tri2Tet> host_tri2tet;
		CArray<Tetrahedron> host_tets;

		host_vertices.resize( (*(this->ptr_vertices)).size() );
		host_triangles.resize( (*(this->ptr_triangles)).size() );
		host_tri2tet.resize( (*(this->ptr_tri2tet)).size() );
		host_tets.resize((*(this->ptr_tets)).size());

		host_vertices.assign(*(this->ptr_vertices));
		host_triangles.assign(*(this->ptr_triangles));
		host_tri2tet.assign(*(this->ptr_tri2tet));
		host_tets.assign(*(this->ptr_tets));

		for (uint i = 0; i < host_vertices.size(); ++i) {
			output << "v " << host_vertices[i][0] << " " << host_vertices[i][1] << " " << host_vertices[i][2] << std::endl;
		}
		for (int i = 0; i < host_tri2tet.size(); ++i) {
			Tri2Tet tmp = host_tri2tet[i];
			bool isOnSurface = false;
			if (tmp[0] < 0 || tmp[1] < 0) { isOnSurface = true; }
			if (isOnSurface) {
				int idx_vertex;
				bool reverse = false;
				int idx_tet = tmp[0] < 0 ? tmp[1] : tmp[0]; 
				for (idx_vertex = 0; idx_vertex < 4; idx_vertex++)
				{

					if (host_tets[idx_tet][idx_vertex] != host_triangles[i][0]
						&& host_tets[idx_tet][idx_vertex] != host_triangles[i][1]
						&& host_tets[idx_tet][idx_vertex] != host_triangles[i][2]
						)
						break;
				}
				idx_vertex = host_tets[idx_tet][idx_vertex];
				if (
					((host_vertices[host_triangles[i][1]] - host_vertices[host_triangles[i][0]]).cross
					(host_vertices[host_triangles[i][2]] - host_vertices[host_triangles[i][1]]))
					.dot
					(host_vertices[idx_vertex] - host_vertices[host_triangles[i][0]]) > 0
					)
					reverse = true;
				if(!reverse)	
					output << "f " << host_triangles[i][0] + 1 << " " << host_triangles[i][1] + 1 << " " << host_triangles[i][2] + 1 << std::endl;
				else
					output << "f " << host_triangles[i][0] + 1<< " " << host_triangles[i][2] + 1<< " " << host_triangles[i][1] + 1<< std::endl;
			}
		}

		host_vertices.clear();
		host_triangles.clear();
		host_tri2tet.clear();

		return;
	}



	DEFINE_CLASS(TetraMeshWriter);
}