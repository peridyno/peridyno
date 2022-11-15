#include "Cloth.h"
#include "VulkanTools.h"
#include "VkSystem.h"
#include "VkContext.h"
#include "VkTransfer.h"
#include "VkProgram.h"
#include "Topology/TriangleSet.h"

#include <vector>
#include <set>
#include <random>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

namespace px
{
#define WORKGROUP_SIZE 64

	Cloth::Cloth(std::string name)
		: dyno::Node()
	{
		auto triSet = std::make_shared<TriangleSet>();
		this->stateTopology()->setDataPtr(triSet);
	}

	Cloth::~Cloth()
	{
	}

	void Cloth::updateStates()
	{
	}

	void Cloth::resetStates()
	{
	}

	void Cloth::loadObjFile(std::string filename)
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
		std::vector<dyno::Vec3f> vertList;
		std::vector<dyno::TopologyModule::Triangle> faceList;
		while (!infile.eof()) {
			std::getline(infile, line);

			//.obj files sometimes contain vertex normals indicated by "vn"
			if (line.substr(0, 1) == std::string("v") && line.substr(0, 2) != std::string("vn")) {
				std::stringstream data(line);
				char c;
				dyno::Vec3f point;
				data >> c >> point[0] >> point[1] >> point[2];
				vertList.push_back(point);
			}
			else if (line.substr(0, 1) == std::string("f")) {
				std::stringstream data(line);
				char c;
				int v0, v1, v2;
				data >> c >> v0 >> v1 >> v2;
				faceList.push_back(dyno::TopologyModule::Triangle(v0 - 1, v1 - 1, v2 - 1));
			}
			else {
				++ignored_lines;
			}
		}
		infile.close();

		auto topo = std::dynamic_pointer_cast<TriangleSet>(this->stateTopology()->getDataPtr());

		topo->mPoints.resize(vertList.size());
		topo->mTriangleIndex.resize(faceList.size());

		vkTransfer(topo->mPoints, vertList);
		vkTransfer(topo->mTriangleIndex, faceList);

		topo->updateTopology();

		vertList.clear();
		faceList.clear();
	}
}