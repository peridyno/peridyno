#include "GLAdaptiveXYPlaneVisualNode.h"
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
	IMPLEMENT_TCLASS(GLAdaptiveXYPlaneVisualNode, TDataType)

	template<typename TDataType>
	GLAdaptiveXYPlaneVisualNode<TDataType>::GLAdaptiveXYPlaneVisualNode()
		: Node()
	{
		//auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
		//this->inZPos()->connect(calculateNorm->inVec());

		//auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		//calculateNorm->outNorm()->connect(colorMapper->inScalar());
		//colorMapper->varMin()->setValue(-1.5f);
		//colorMapper->varMax()->setValue(5.0f);

		//auto ptRender = std::make_shared<GLPointVisualModule>();
		//ptRender->setColor(Color(1, 0, 0));
		//ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
		//this->varPSize()->connect(ptRender->varPointSize());
		//this->stateLeafNodes()->connect(ptRender->inPointSet());
		//colorMapper->outColor()->connect(ptRender->inColor());

		auto wRender = std::make_shared<GLWireframeVisualModule>();
		this->stateGrids()->connect(wRender->inEdgeSet());
		wRender->setColor(Color(0, 0, 0));

		this->graphicsPipeline()->pushModule(wRender);
		//this->graphicsPipeline()->pushModule(calculateNorm);
		//this->graphicsPipeline()->pushModule(colorMapper);
		//this->graphicsPipeline()->pushModule(ptRender);
	}

	template<typename TDataType>
	GLAdaptiveXYPlaneVisualNode<TDataType>::~GLAdaptiveXYPlaneVisualNode()
	{
		printf("GLAdaptiveXYPlaneVisualNode released \n");
	}

	template <typename Coord>
	__global__ void ESS_CountNodes2DNeighbor(
		DArray<Coord> points,
		DArray<uint> count,
		DArray<AdaptiveGridNode> octree,
		DArrayList<int> neighbor,
		Real zpos)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= octree.size()) return;

		points[tId] = octree[tId].m_position;
		points[tId][2] = zpos;

		count[4 * tId] = neighbor[4 * tId].size();
		count[4 * tId + 1] = neighbor[4 * tId + 1].size();
		count[4 * tId + 2] = neighbor[4 * tId + 2].size();
		count[4 * tId + 3] = neighbor[4 * tId + 3].size();
	}

	__global__ void ESS_CountEdges2D(
		DArray<TopologyModule::Edge> edge,
		DArray<uint> count,
		DArrayList<int> neighbor,
		int node_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= node_num) return;

		for (int i = 0; i < neighbor[4 * tId].size(); i++)
			edge[count[4 * tId] + i] = TopologyModule::Edge(tId, neighbor[4 * tId][i]);
		for (int i = 0; i < neighbor[4 * tId + 1].size(); i++)
			edge[count[4 * tId + 1] + i] = TopologyModule::Edge(tId, neighbor[4 * tId + 1][i]);
		for (int i = 0; i < neighbor[4 * tId + 2].size(); i++)
			edge[count[4 * tId + 2] + i] = TopologyModule::Edge(tId, neighbor[4 * tId + 2][i]);
		for (int i = 0; i < neighbor[4 * tId + 3].size(); i++)
			edge[count[4 * tId + 3] + i] = TopologyModule::Edge(tId, neighbor[4 * tId + 3][i]);
	}

	template <typename Real, typename Coord>
	__global__ void ESS_CountNodes2D(
		DArray<Coord> edge_points,
		DArray<TopologyModule::Edge> edge,
		DArray<AdaptiveGridNode> octree,
		Real dx,
		Level max_level,
		Real zpos)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= octree.size()) return;

		Coord pos = octree[tId].m_position;
		pos[2] = zpos;

		Real up_dx = dx * (1 << (max_level - octree[tId].m_level));

		edge_points[4 * tId] = pos + Coord(-0.5*up_dx, -0.5*up_dx, 0.0);
		edge_points[4 * tId + 1] = pos + Coord(-0.5*up_dx, 0.5*up_dx, 0.0);
		edge_points[4 * tId + 2] = pos + Coord(0.5*up_dx, 0.5*up_dx, 0.0);
		edge_points[4 * tId + 3] = pos + Coord(0.5*up_dx, -0.5*up_dx, 0.0);

		edge[4 * tId] = TopologyModule::Edge(4 * tId, 4 * tId + 1);
		edge[4 * tId + 1] = TopologyModule::Edge(4 * tId + 1, 4 * tId + 2);
		edge[4 * tId + 2] = TopologyModule::Edge(4 * tId + 2, 4 * tId + 3);
		edge[4 * tId + 3] = TopologyModule::Edge(4 * tId + 3, 4 * tId);
	}

	template<typename TDataType>
	void GLAdaptiveXYPlaneVisualNode<TDataType>::resetStates()
	{
		this->stateGrids()->allocate();

		DArray<AdaptiveGridNode> node;
		DArrayList<int> neighbor;
		auto volumeSet = this->inAdaptiveVolume()->getDataPtr();
		Real zpos = this->varZPos()->getData();
		volumeSet->extractLeafs2D(node, neighbor, zpos);
		if (node.size() == 0)
		{
			node.clear();
			neighbor.clear();
			return;
		}
		Level level = volumeSet->adaptiveGridLevelMax();
		Real dx = volumeSet->adaptiveGridDx();
		//printf("GLAdaptiveXYPlaneVisualNode node num %d \n", node.size());

		DArray<Coord> edge_points;
		DArray<TopologyModule::Edge> edges;
		if (this->varType()->getData() == EdgeData::Octree_Neighbor)
		{
			edge_points.resize(node.size());
			DArray<uint> count(4 * node.size());
			cuExecute(node.size(),
				ESS_CountNodes2DNeighbor,
				edge_points,
				count,
				node,
				neighbor,
				zpos);
			Reduction<uint> reduce;
			Scan<uint> scan;
			int edge_num = reduce.accumulate(count.begin(), count.size());
			scan.exclusive(count.begin(), count.size());
			edges.resize(edge_num);
			edges.reset();
			cuExecute(node.size(),
				ESS_CountEdges2D,
				edges,
				count,
				neighbor,
				node.size());
			count.clear();
		}
		else
		{
			edge_points.resize(4 * node.size());
			edges.resize(4 * node.size());
			cuExecute(node.size(),
				ESS_CountNodes2D,
				edge_points,
				edges,
				node,
				dx,
				level,
				zpos);
		}

		this->stateGrids()->getDataPtr()->setPoints(edge_points);
		this->stateGrids()->getDataPtr()->setEdges(edges);

		edge_points.clear();
		edges.clear();
		node.clear();
		neighbor.clear();

		//DArray<Coord> pos;
		//DArray<Coord> scale;
		//volumeSet->getLeafs(pos, scale);
		//this->stateLeafPosition()->allocate();
		//this->stateLeafPosition()->assign(pos);
		//this->stateLeafScale()->allocate();
		//this->stateLeafScale()->assign(scale);
		//DArray<Real> value;
		//value.resize(pos.size());
		//value.reset();
		//this->stateLeafsValue()->allocate();
		//this->stateLeafsValue()->assign(value);
		//pos.clear();
		//scale.clear();
		//value.clear();
	}

	template<typename TDataType>
	void GLAdaptiveXYPlaneVisualNode<TDataType>::updateStates()
	{
		this->resetStates();
	}

	template<typename TDataType>
	bool GLAdaptiveXYPlaneVisualNode<TDataType>::validateInputs()
	{
		if (this->inAdaptiveVolume()->isEmpty()) {
			return false;
		}

		return Node::validateInputs();
	}

	DEFINE_CLASS(GLAdaptiveXYPlaneVisualNode);
}
