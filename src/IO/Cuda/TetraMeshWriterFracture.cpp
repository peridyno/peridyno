#include "TetraMeshWriterFracture.h"
#include "Module/OutputModule.h"

#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <algorithm>

namespace dyno
{
	IMPLEMENT_TCLASS(TetraMeshWriterFracture, TDataType)

	template<typename TDataType>
	TetraMeshWriterFracture<TDataType>::TetraMeshWriterFracture() : OutputModule()
	{

	}

	template<typename TDataType>
	TetraMeshWriterFracture<TDataType>::~TetraMeshWriterFracture()
	{
	}


	
	bool cmp(OriginalFaceId a, OriginalFaceId b)
	{
		return a.maxVertexID() < b.maxVertexID();
	}

	template<typename TDataType>
	void TetraMeshWriterFracture<TDataType>::loadUVs(std::string filename)
	{
		if (filename.size() < 5 || filename.substr(filename.size() - 4) != std::string(".obj")) {
			std::cerr << "Error: Expected OBJ file with filename of the form <name>.obj.\n";
			exit(-1);
		}

		std::ifstream infile(filename);
		if (!infile) {
			std::cerr << "Failed to open. Terminating.\n";
			exit(-1);
		}

		int ignored_lines = 0;
		std::string line;
		std::vector<Coord> vertList;
		std::vector<Triangle> faceList;
		std::vector<Triangle> faceVnList;
		while (!infile.eof()) {
			std::getline(infile, line);
			//std::cout << line << std::endl;
			//.obj files sometimes contain vertex normals indicated by "vn"
			if (line.substr(0, 1) == std::string("v") && line.substr(0, 2) != std::string("vn") && line.substr(0, 2) != std::string("vt")) {
				std::stringstream data(line);
				char c;
				Coord point;
				data >> c >> point[0] >> point[1] >> point[2];
				vertList.push_back(point);
			}
			else if (line.substr(0, 2) == std::string("vt"))
			{
				std::stringstream data(line);
				char c[20];
				Coord point;
				data >> c >> point[0] >> point[1];
				point[2] = 0.0f;
				vnList.push_back(point);
			}
			else if (line.substr(0, 1) == std::string("f")) {
				std::stringstream data(line);
				char c;
				char ss1[30];
				char ss2[30];
				char ss3[30];
				int v0, v1, v2;
				v0 = v1 = v2 = 0;
				int vt0, vt1, vt2;
				vt0 = vt1 = vt2 = 0;

				data >> c >> ss1 >> ss2 >> ss3;

				bool cnt = false;
				for (int i = 0; i < strlen(ss1); i++)
				{
					if (!cnt)
					{
						if (ss1[i] == '/') cnt = true;
						else
						{
							v0 *= 10;
							v0 += ss1[i] - '0';
						}
					}
					else
					{
						if (ss1[i] == '/') break;
						else
						{
							vt0 *= 10;
							vt0 += ss1[i] - '0';
						}
					}
				}

				cnt = false;
				for (int i = 0; i < strlen(ss2); i++)
				{
					if (!cnt)
					{
						if (ss2[i] == '/') cnt = true;
						else
						{
							v1 *= 10;
							v1 += ss2[i] - '0';
						}
					}
					else
					{
						if (ss2[i] == '/') break;
						else
						{
							vt1 *= 10;
							vt1 += ss2[i] - '0';
						}
					}
				}

				cnt = false;
				for (int i = 0; i < strlen(ss3); i++)
				{
					if (!cnt)
					{
						if (ss3[i] == '/') cnt = true;
						else
						{
							v2 *= 10;
							v2 += ss3[i] - '0';
						}
					}
					else
					{
						if (ss3[i] == '/') break;
						else
						{
							vt2 *= 10;
							vt2 += ss3[i] - '0';
						}
					}
				}
				/*std::cout << v0 << ' ' << v1 << ' ' << v2 << std::endl;
				std::cout << vt0 << ' ' << vt1 << ' ' << vt2 << std::endl;*/
				faceList.push_back(Triangle(v0 - 1, v1 - 1, v2 - 1));
				faceVnList.push_back(Triangle(vt0, vt1, vt2));
			}
			else {
				++ignored_lines;
			}
		}

		printf("end\n");

		infile.close();
		std::cout << faceList.size() << std::endl;
		FaceId.resize(faceList.size());
		for (int i = 0; i < faceList.size(); i++)
		{
			FaceId[i].vertexId1 = faceList[i][0];
			FaceId[i].vertexId2 = faceList[i][1];
			FaceId[i].vertexId3 = faceList[i][2];
			FaceId[i].uvId1 = faceVnList[i][0];
			FaceId[i].uvId2 = faceVnList[i][1];
			FaceId[i].uvId3 = faceVnList[i][2];
		}
		std::sort(FaceId.begin(), FaceId.begin() + faceList.size(), cmp);

		FaceStart.resize(vertList.size());

		//std::cout << vertList.size() << std::endl;
		for (int i = 0; i < FaceStart.size(); i++)
			FaceStart[i] = -1;
		for (int i = 0; i < FaceId.size(); i++)
		{
			if (i == 0 || FaceId[i].maxVertexID() != FaceId[i - 1].maxVertexID())
			{
				FaceStart[FaceId[i].maxVertexID()] = i;
			}
		}
		printf("outside\n");

	}

	template<typename TDataType>
	bool TetraMeshWriterFracture<TDataType>::updatePtr() 
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
	int TetraMeshWriterFracture<TDataType>::onInitSurface(Triangle Tri)
	{
		int maximum_index = glm::max(glm::max(Tri[0], Tri[1]), Tri[2]);
		int minimum_index = glm::min(glm::min(Tri[0], Tri[1]), Tri[2]);
		int sum_index = Tri[0] + Tri[1] + Tri[2];
		
		/*if(maximum_index > FaceStart.size())
			std::cout << "!!!!!" << maximum_index << ' ' << FaceStart.size() << std::endl;*/
		if (maximum_index < 0) return -1;
		
		if(FaceStart[maximum_index] == -1)
			return -1;
		//std::cout<< FaceStart[maximum_index] <<' '<< FaceId.size() << std::endl;
		for (int i = FaceStart[maximum_index]; i < FaceId.size() && FaceId[i].maxVertexID() == maximum_index; i++)
		{
			if(FaceId[i].minVertexID() == minimum_index && FaceId[i].sumVertexID() == sum_index)
				return i;
		}
		return -1;
	}


	template<typename TDataType>
	bool TetraMeshWriterFracture<TDataType>::outputSurfaceMesh() 
	{
		
		if (this->ptr_tri2tet == nullptr || this->ptr_triangles == nullptr || this->ptr_vertices == nullptr) {
			printf("------Tetra Mesh Writer: array nullptr \n");
			return false;
		}

		auto output_path = this->varOutputPath()->getValue();
		int frame_number = this->getFrameNumber();
		std::stringstream ss; ss <<frame_number;
		std::string filename = output_path + ss.str() + this->file_postfix;
		std::string filenameF = output_path + "Frac_" + ss.str() + this->file_postfix;
		std::ofstream output(filename.c_str(), std::ios::out);
		std::ofstream outputF(filenameF.c_str(), std::ios::out);

		if (!output.is_open()) {
			printf("------Tetra Mesh Writer: open file failed \n");
			return false;
		}

		this->updatePtr();

		CArray<Coord> host_vertices;
		CArray<Triangle> host_triangles;
		CArray<Tri2Tet> host_tri2tet;
		CArray<Tetrahedron> host_tets;
		CArray<int> host_VerId;

		host_vertices.resize( (*(this->ptr_vertices)).size() );
		host_triangles.resize( (*(this->ptr_triangles)).size() );
		host_tri2tet.resize( (*(this->ptr_tri2tet)).size() );
		host_tets.resize((*(this->ptr_tets)).size());

		host_VerId.resize((*(this->ptr_vertices)).size());
		host_VerId.assign((*OringalID));

		std::cout << host_VerId.size() << std::endl;
		
		host_vertices.assign(*(this->ptr_vertices));
		host_triangles.assign(*(this->ptr_triangles));
		host_tri2tet.assign(*(this->ptr_tri2tet));
		host_tets.assign(*(this->ptr_tets));


		if (first)
		{
			first = false;
			onFace.resize(host_VerId.size());
			FaceMapping.resize(host_VerId.size());
			for (int i = 0; i < onFace.size(); i++)
				onFace[i] = 1;
			
			for (int i = 0; i < host_tri2tet.size(); ++i) {
				Tri2Tet tmp = host_tri2tet[i];
				bool isOnSurface = false;
				if (tmp[0] < 0 || tmp[1] < 0) { isOnSurface = true; }
				if (isOnSurface)
					{ 
						onFace[host_triangles[i][0]] = true;
					    onFace[host_triangles[i][1]] = true;
						onFace[host_triangles[i][2]] = true;
					}
			}
			int mapping_last = 0;
			for (int i = 0; i < onFace.size(); i++)
			{
				if (onFace[i])
				{ 
					FaceMapping[i] = mapping_last;
					mapping_last++;
				}
				else
					FaceMapping[i] = -1;
			}
			std::cout <<"mapping_last_size = "<< mapping_last << std::endl;
			
		}

		for (uint i = 0; i < host_vertices.size(); ++i) {
			output << "v " << host_vertices[i][0] << " " << host_vertices[i][1] << " " << host_vertices[i][2] << std::endl;
			outputF << "v " << host_vertices[i][0] << " " << host_vertices[i][1] << " " << host_vertices[i][2] << std::endl;
		}

		for (uint i = 0; i < vnList.size(); ++i) {
			output << "vt " << vnList[i][0] << " " << vnList[i][1] << std::endl;
		}

		for (int i = 0; i < host_tri2tet.size(); ++i) {
			Tri2Tet tmp = host_tri2tet[i];
			bool isOnSurface = false;
			if (tmp[0] < 0 || tmp[1] < 0) { isOnSurface = true; }
			if (isOnSurface) {
				Triangle newTri;

				/*std::cout << 'a' 
					<< host_VerId[host_triangles[i][0]] << ' '
					<< host_VerId[host_triangles[i][1]] <<' '
					<< host_VerId[host_triangles[i][2]] << std::endl;*/

				newTri[0] = FaceMapping[host_VerId[host_triangles[i][0]]];
				newTri[1] = FaceMapping[host_VerId[host_triangles[i][1]]];
				newTri[2] = FaceMapping[host_VerId[host_triangles[i][2]]];
				//std::cout << 'a' << std::endl;
				int idInitSurface = onInitSurface(newTri);
				//std::cout << idInitSurface << std::endl;
				if (idInitSurface >= 0)
				{
					int newid1 = (newTri[0] == FaceId[idInitSurface].vertexId1) ? host_triangles[i][0] : ((newTri[1] == FaceId[idInitSurface].vertexId1) ? host_triangles[i][1] : host_triangles[i][2]);
					int newid2 = (newTri[0] == FaceId[idInitSurface].vertexId2) ? host_triangles[i][0] : ((newTri[1] == FaceId[idInitSurface].vertexId2) ? host_triangles[i][1] : host_triangles[i][2]);
					int newid3 = (newTri[0] == FaceId[idInitSurface].vertexId3) ? host_triangles[i][0] : ((newTri[1] == FaceId[idInitSurface].vertexId3) ? host_triangles[i][1] : host_triangles[i][2]);
					output << "f "
						<< newid1 + 1 << "/" << FaceId[idInitSurface].uvId1 << " "
						<< newid2 + 1 << "/" << FaceId[idInitSurface].uvId2 << " "
						<< newid3 + 1 << "/" << FaceId[idInitSurface].uvId3 << std::endl;

				}
				else
				{
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
					if (!reverse)
						outputF << "f " << host_triangles[i][0] + 1 << " " << host_triangles[i][1] + 1 << " " << host_triangles[i][2] + 1 << std::endl;
					else
						outputF << "f " << host_triangles[i][0] + 1 << " " << host_triangles[i][2] + 1 << " " << host_triangles[i][1] + 1 << std::endl;

				}
			}
			
		}

		host_vertices.clear();
		host_triangles.clear();
		host_tri2tet.clear();
		host_VerId.clear();

		return true;
	}


	DEFINE_CLASS(TetraMeshWriterFracture);
}