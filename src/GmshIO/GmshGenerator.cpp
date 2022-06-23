#include "GmshGenerator.h"

#include <vector>
#include <gmsh.h>

namespace dyno
{
	template<typename TDataType>
	GmshGenerator<TDataType>::GmshGenerator()
	: Node()
	{
	}

	template<typename TDataType>
	GmshGenerator<TDataType>::~GmshGenerator()
	{
	}

	template<typename TDataType>
	void GmshGenerator<TDataType>::resetStates()
	{
		gmsh::initialize();

		gmsh::model::add("x2");


		double lc = 1;//2e-1;
		int pIndex = 1;
		gmsh::model::geo::addPoint(-1, -1, -1, lc, pIndex++);
		gmsh::model::geo::addPoint(1, -1, -1, lc, pIndex++);
		gmsh::model::geo::addPoint(1, 1, -1, lc, pIndex++);
		gmsh::model::geo::addPoint(-1, 1, -1, lc, pIndex++);
		gmsh::model::geo::addPoint(-1, -1, 1, lc, pIndex++);
		gmsh::model::geo::addPoint(1, -1, 1, lc, pIndex++);
		gmsh::model::geo::addPoint(1, 1, 1, lc, pIndex++);
		gmsh::model::geo::addPoint(-1, 1, 1, lc, pIndex++);

		// The API to create lines with the built-in kernel follows the same
		// conventions: the first 2 arguments are point tags, the last (optional one)
		// is the line tag.
		int lIndex = 1;
		int p, q;
		p = 1;
		for (int i = 0; i < 3; i++)
		{
			gmsh::model::geo::addLine(p, p + 1, lIndex++);
			p++;
		}
		gmsh::model::geo::addLine(p, 1, lIndex++);

		p = 5;
		for (int i = 0; i < 3; i++)
		{
			gmsh::model::geo::addLine(p, p + 1, lIndex++);
			p++;
		}
		gmsh::model::geo::addLine(p, 5, lIndex++);

		p = 1;
		q = 5;
		for (int i = 0; i < 4; i++)
		{
			gmsh::model::geo::addLine(p, q, lIndex++);
			p++;
			q++;
		}


		// The philosophy to construct curve loops and surfaces is similar: the first
		// argument is now a vector of integers.
		gmsh::model::geo::addCurveLoop({ 1, 2, 3, 4 }, 1);
		gmsh::model::geo::addCurveLoop({ 5, 6, 7, 8 }, 2);
		gmsh::model::geo::addCurveLoop({ 1, 10, -5, -9 }, 3);
		gmsh::model::geo::addCurveLoop({ 2, 11, -6, -10 }, 4);
		gmsh::model::geo::addCurveLoop({ 3, 12, -7, -11 }, 5);
		gmsh::model::geo::addCurveLoop({ -4,12, 8,   -9 }, 6);



		gmsh::model::geo::addPlaneSurface({ 1 }, 1);
		gmsh::model::geo::addPlaneSurface({ 2 }, 2);
		gmsh::model::geo::addPlaneSurface({ 3 }, 3);
		gmsh::model::geo::addPlaneSurface({ 4 }, 4);
		gmsh::model::geo::addPlaneSurface({ 5 }, 5);
		gmsh::model::geo::addPlaneSurface({ 6 }, 6);

		gmsh::model::geo::addSurfaceLoop({ 1, 2, 3, 4, 5, 6 }, 1);
		gmsh::model::geo::addVolume({ 1 }, 1);
		// Physical groups are defined by providing the dimension of the group (0 for
		// physical points, 1 for physical curves, 2 for physical surfaces and 3 for
		// phsyical volumes) followed by a vector of entity tags. The last (optional)
		// argument is the tag of the new group to create.

		gmsh::model::setPhysicalName(3, 1, "My volume");

		// Before it can be meshed, the internal CAD representation must be
		// synchronized with the Gmsh model, which will create the relevant Gmsh data
		// structures. This is achieved by the gmsh::model::geo::synchronize() API
		// call for the built-in CAD kernel. Synchronizations can be called at any
		// time, but they involve a non trivial amount of processing; so while you
		// could synchronize the internal CAD data after every CAD command, it is
		// usually better to minimize the number of synchronization points.
		gmsh::model::geo::synchronize();

		// We can then generate a 2D mesh...
		gmsh::model::mesh::generate();
		//gmsh::model::mesh::refine();

		std::vector<double> nodes, y;
		std::vector<std::size_t> nodeTags;
		gmsh::model::mesh::getNodes(nodeTags, nodes, y, -2, -1, false, true);
		std::cout << "The number of nodes: " << nodes.size() / 3 << std::endl;

		std::vector<int> elementTypes;
		std::vector<std::vector<std::size_t> > elementTags, nodeTags2;
		gmsh::model::mesh::getElements(elementTypes, elementTags, nodeTags2, -2, -1);
		std::cout << "The number of elements: " << nodeTags2[1].size() / 3 << std::endl;

		gmsh::finalize();


		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> indices;

		for (size_t i = 0; i < nodes.size() / 3; i++)
		{
			vertices.push_back(Coord(nodes[3 * i], nodes[3 * i + 1], nodes[3 * i + 2]));
		}

		for (size_t i = 0; i < nodeTags2[1].size() / 3; i++)
		{
			TopologyModule::Triangle t(nodeTags2[1][3 * i] - 1, nodeTags2[1][3 * i + 1] - 1, nodeTags2[1][3 * i + 2] - 1);
			indices.push_back(t);
		}

		//setup the output data
		if (this->outGmsh()->isEmpty())
		{
			this->outGmsh()->allocate();
		}

		auto mesh = this->outGmsh()->getDataPtr();

		mesh->setPoints(vertices);
		mesh->setTriangles(indices);

		mesh->update();

		vertices.clear();
		indices.clear();

		nodes.clear();
		y.clear();
		nodeTags.clear();
		elementTypes.clear();
		elementTags.clear();
		nodeTags2.clear();

		
	}

	DEFINE_CLASS(GmshGenerator)
}