#pragma once
#include <vector>
#include <set>
#include "Vector.h"
#include <Topology.h>

namespace dyno{


	class Smesh {

	public:
		void loadFile(std::string filename);

		void loadNodeFile(std::string filename);
		void loadEdgeFile(std::string filename);
		void loadTriangleFile(std::string filename);
		void loadTetFile(std::string filename);



		std::vector<Vec3f> m_points;
		std::vector<Topology::Edge> m_edges;
		std::vector<Topology::Triangle> m_triangles;
		std::vector<Topology::Quad> m_quads;
		std::vector<Topology::Tetrahedron> m_tets;
		std::vector<Topology::Hexahedron> m_hexs;
	};

}