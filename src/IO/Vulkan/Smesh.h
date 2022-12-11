#pragma once
#include <vector>
#include <set>
#include "Vector.h"
#include "Module/TopologyModule.h"

namespace dyno
{
	class Smesh {

	public:
		void loadFile(std::string filename);

		void loadNodeFile(std::string filename);
		void loadEdgeFile(std::string filename);
		void loadTriangleFile(std::string filename);
		void loadTetFile(std::string filename);



		std::vector <dyno::Vec3f> m_points;
		std::vector<dyno::TopologyModule::Edge> m_edges;
		std::vector<dyno::TopologyModule::Triangle> m_triangles;
		std::vector<dyno::TopologyModule::Quad> m_quads;
		std::vector<dyno::TopologyModule::Tetrahedron> m_tets;
		std::vector<dyno::TopologyModule::Hexahedron> m_hexs;
	};

}