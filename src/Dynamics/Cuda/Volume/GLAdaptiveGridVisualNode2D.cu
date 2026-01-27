#include "GLAdaptiveGridVisualNode2D.h"
#include "Algorithm/Reduction.h"

#include "Module/CalculateNorm.h"
#include "Topology/PointSet.h"
//#include "Topology/GridSet.h"
#include "GLPointVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "ColorMapping.h"
#include "../GUI/ImWidgets/ImColorbar.h"
#include <thrust/sort.h>

namespace dyno
{
	IMPLEMENT_TCLASS(GLAdaptiveGridVisualNode2D, TDataType)

	template<typename TDataType>
	GLAdaptiveGridVisualNode2D<TDataType>::GLAdaptiveGridVisualNode2D()
		: Node()
	{
		auto wRender = std::make_shared<GLWireframeVisualModule>();
		this->stateGrids()->connect(wRender->inEdgeSet());
		wRender->setColor(Color(0, 0, 0));

		this->graphicsPipeline()->pushModule(wRender);
	}

	template<typename TDataType>
	GLAdaptiveGridVisualNode2D<TDataType>::~GLAdaptiveGridVisualNode2D()
	{
		printf("GLAdaptiveXYPlaneVisualNode released \n");
	}

	template<typename Coord2D, typename Coord3D>
	__global__ void AG2DVN_CountNodes2DNeighbor(
		DArray<Coord3D> edge_points,
		DArray<TopologyModule::Edge> edge,
		DArray<Coord2D> nodes,
		DArrayList<int> neighbor,
		DArray<uint> index,
		int ptype)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		if (ptype == 0)
			edge_points[tId] = Coord3D(nodes[tId][0], nodes[tId][1], 0.0);
		else if (ptype == 1)
			edge_points[tId] = Coord3D(nodes[tId][0], 0.0, nodes[tId][1]);
		else if (ptype == 2)
			edge_points[tId] = Coord3D(0.0, nodes[tId][0], nodes[tId][1]);

		for (int i = 0; i < neighbor[4 * tId].size(); i++)
			edge[index[4 * tId] + i] = TopologyModule::Edge(tId, neighbor[4 * tId][i]);
		for (int i = 0; i < neighbor[4 * tId + 1].size(); i++)
			edge[index[4 * tId + 1] + i] = TopologyModule::Edge(tId, neighbor[4 * tId + 1][i]);
		for (int i = 0; i < neighbor[4 * tId + 2].size(); i++)
			edge[index[4 * tId + 2] + i] = TopologyModule::Edge(tId, neighbor[4 * tId + 2][i]);
		for (int i = 0; i < neighbor[4 * tId + 3].size(); i++)
			edge[index[4 * tId + 3] + i] = TopologyModule::Edge(tId, neighbor[4 * tId + 3][i]);
	}

	template <typename Real, typename Coord2D, typename Coord3D>
	__global__ void AG2DVN_CountNodes2D(
		DArray<Coord3D> edge_points,
		DArray<TopologyModule::Edge> edge,
		DArray<Coord2D> nodes,
		DArray<Real> scale,
		int ptype)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		Real up_dx = scale[tId];

		if (ptype == 0)
		{
			Coord3D pos(nodes[tId][0], nodes[tId][1], 0.0f);
			edge_points[4 * tId] = pos + Coord3D(-0.5 * up_dx, -0.5 * up_dx, 0.0);
			edge_points[4 * tId + 1] = pos + Coord3D(-0.5 * up_dx, 0.5 * up_dx, 0.0);
			edge_points[4 * tId + 2] = pos + Coord3D(0.5 * up_dx, 0.5 * up_dx, 0.0);
			edge_points[4 * tId + 3] = pos + Coord3D(0.5 * up_dx, -0.5 * up_dx, 0.0);
		}
		else if (ptype == 1)
		{
			Coord3D pos(nodes[tId][0], 0.0f, nodes[tId][1]);
			edge_points[4 * tId] = pos + Coord3D(-0.5 * up_dx, 0.0, -0.5 * up_dx);
			edge_points[4 * tId + 1] = pos + Coord3D(-0.5 * up_dx, 0.0, 0.5 * up_dx);
			edge_points[4 * tId + 2] = pos + Coord3D(0.5 * up_dx, 0.0, 0.5 * up_dx);
			edge_points[4 * tId + 3] = pos + Coord3D(0.5 * up_dx, 0.0, -0.5 * up_dx);
		}
		else if (ptype == 2)
		{
			Coord3D pos(0.0f, nodes[tId][0], nodes[tId][1]);
			edge_points[4 * tId] = pos + Coord3D(0.0, -0.5 * up_dx, -0.5 * up_dx);
			edge_points[4 * tId + 1] = pos + Coord3D(0.0, -0.5 * up_dx, 0.5 * up_dx);
			edge_points[4 * tId + 2] = pos + Coord3D(0.0, 0.5 * up_dx, 0.5 * up_dx);
			edge_points[4 * tId + 3] = pos + Coord3D(0.0, 0.5 * up_dx, -0.5 * up_dx);
		}

		edge[4 * tId] = TopologyModule::Edge(4 * tId, 4 * tId + 1);
		edge[4 * tId + 1] = TopologyModule::Edge(4 * tId + 1, 4 * tId + 2);
		edge[4 * tId + 2] = TopologyModule::Edge(4 * tId + 2, 4 * tId + 3);
		edge[4 * tId + 3] = TopologyModule::Edge(4 * tId + 3, 4 * tId);
	}

	template<typename TDataType>
	void GLAdaptiveGridVisualNode2D<TDataType>::resetStates()
	{
		this->stateGrids()->allocate();
		auto volumeSet = this->inAdaptiveVolume()->getDataPtr();

		DArray<Coord3D> edge_points;
		DArray<TopologyModule::Edge> edges;

		if (this->varEType()->currentKey() == EdgeType::Quadtree_Neighbor)
		{
			DArray<Coord2D> node;
			DArrayList<int> neighbor;
			volumeSet->extractLeafs(node, neighbor);
			if (node.size() == 0) return;

			edge_points.resize(node.size());
			edges.resize(neighbor.elementSize());
			cuExecute(node.size(),
				AG2DVN_CountNodes2DNeighbor,
				edge_points,
				edges,
				node,
				neighbor,
				neighbor.index(),
				int(this->varPPlane()->currentKey()));
			node.clear();
			neighbor.clear();
		}
		else
		{
			DArray<Coord2D> node;
			DArray<Real> scale;
			volumeSet->extractLeafs(node, scale);
			if (node.size() == 0) return;

			edge_points.resize(4 * node.size());
			edges.resize(4 * node.size());
			cuExecute(node.size(),
				AG2DVN_CountNodes2D,
				edge_points,
				edges,
				node,
				scale,
				int(this->varPPlane()->currentKey()));
			node.clear();
			scale.clear();
		}

		this->stateGrids()->getDataPtr()->setPoints(edge_points);
		this->stateGrids()->getDataPtr()->setEdges(edges);

		edge_points.clear();
		edges.clear();
	}

	template<typename TDataType>
	void GLAdaptiveGridVisualNode2D<TDataType>::updateStates()
	{
		this->resetStates();
	}

	template<typename TDataType>
	bool GLAdaptiveGridVisualNode2D<TDataType>::validateInputs()
	{
		if (this->inAdaptiveVolume()->isEmpty()) {
			return false;
		}

		return Node::validateInputs();
	}

	DEFINE_CLASS(GLAdaptiveGridVisualNode2D);
}
