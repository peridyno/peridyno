#pragma once
#include <vector>
#include <set>
#include "Vector.h"
#include <Topology.h>

namespace dyno{


	class Gmsh {

	public:
		void loadFile(std::string filename);


		std::vector<Vec3f> m_points;
		std::vector<Topology::Tetrahedron> m_tets;
	};

}