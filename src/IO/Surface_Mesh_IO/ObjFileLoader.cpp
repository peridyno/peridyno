#include "ObjFileLoader.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace dyno{
    
	ObjFileLoader::ObjFileLoader(std::string filename)
	{
		load(filename);
	}

	bool ObjFileLoader::load(const std::string &filename)
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
		std::vector<std::string> facelets;
		while (!infile.eof()) {
			std::getline(infile, line);

			//.obj files sometimes contain vertex normals indicated by "vn"
			if (line.substr(0, 1) == std::string("v") && line.substr(0, 2) != std::string("vn") && line.substr(0, 2) != std::string("vt")) {
				std::stringstream data(line);
				char c;
				Vec3f point;
				data >> c >> point[0] >> point[1] >> point[2];
				vertList.push_back(point);
			}
			else if (line.substr(0, 1) == std::string("f")) {
				facelets.push_back(line);
			}
			else if (line.substr(0, 2) == std::string("vn")) {
				std::cerr << "Obj-loader is not able to parse vertex normals, please strip them from the input file. \n";
				exit(-2);
			}
			else if (line.substr(0, 2) == std::string("vt")) {
				std::stringstream data(line);
				char c;
				TexCoord tex;
				data >> c >> tex[0] >> tex[1];
				texCoords.push_back(tex);
			}
			else {
				++ignored_lines;
			}
		}

		for(auto line : facelets)
		{
			std::stringstream data(line);

			std::string item;
			std::getline(data, item, ' ');

			Face faceIndex;
			TexIndex texIndex;

			for(int i = 0; i < 3; i++)
			{
				std::getline(data, item, ' ');
				std::stringstream vertexData(item);

				std::string triId;
				std::getline(vertexData, triId, '/');
				faceIndex[i] = atoi(triId.c_str()) - 1;

				if (texCoords.size() > 0)
				{
					std::string texId;
					std::getline(vertexData, texId, '/');
					texIndex[i] = atoi(texId.c_str()) - 1;
				}
			}

			faceList.push_back(faceIndex);
			if (texCoords.size() > 0)
			{
				texList.push_back(texIndex);
			}
		}

		infile.close();
		facelets.clear();
	}

	bool ObjFileLoader::save(const std::string &filename)
	{
		return true;
	}

	std::vector<Vec3f>& ObjFileLoader::getVertexList()
	{
		return vertList;
	}

	std::vector<Face>& ObjFileLoader::getFaceList()
	{
		return faceList;
	}

} //end of namespace dyno
