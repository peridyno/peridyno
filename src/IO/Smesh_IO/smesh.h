#pragma once
#include <vector>
#include <set>
#include "Vector.h"
#include <Module/TopologyModule.h>

namespace dyno{


	class Smesh {

	public:
		void loadFile(std::string filename);

		void loadNodeFile(std::string filename);
		void loadEdgeFile(std::string filename);
		void loadTriangleFile(std::string filename);
		void loadTetFile(std::string filename);



		std::vector<Vec3f> m_points;
		std::vector<TopologyModule::Edge> m_edges;
		std::vector<TopologyModule::Triangle> m_triangles;
		std::vector<TopologyModule::Quad> m_quads;
		std::vector<TopologyModule::Tetrahedron> m_tets;
		std::vector<TopologyModule::Hexahedron> m_hexs;
	};

}