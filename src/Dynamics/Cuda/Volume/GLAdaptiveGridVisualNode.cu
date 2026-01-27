#include "GLAdaptiveGridVisualNode.h"
#include "Algorithm/Reduction.h"

#include "Topology/PointSet.h"
#include "GLPointVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "ColorMapping.h"
#include "../GUI/ImWidgets/ImColorbar.h"
#include <thrust/sort.h>

namespace dyno
{
	IMPLEMENT_TCLASS(GLAdaptiveGridVisualNode, TDataType)

	template<typename TDataType>
	GLAdaptiveGridVisualNode<TDataType>::GLAdaptiveGridVisualNode()
		: Node()
	{
		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		this->stateLeafsValue()->connect(colorMapper->inScalar());
		colorMapper->varMin()->setValue(-5.0f);
		colorMapper->varMax()->setValue(5.0f);

		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->setColor(Color(1, 0, 0));
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
		this->varPointSize()->connect(ptRender->varPointSize());
		this->stateLeafs()->connect(ptRender->inPointSet());
		colorMapper->outColor()->connect(ptRender->inColor());

		auto wRender = std::make_shared<GLWireframeVisualModule>();
		this->stateLeafsEdge()->connect(wRender->inEdgeSet());
		wRender->setColor(Color(0, 0, 0));

		this->graphicsPipeline()->pushModule(colorMapper);
		this->graphicsPipeline()->pushModule(ptRender);
		this->graphicsPipeline()->pushModule(wRender);
	}

	template<typename TDataType>
	GLAdaptiveGridVisualNode<TDataType>::~GLAdaptiveGridVisualNode()
	{
	}
	 
	template<typename TDataType>
	void GLAdaptiveGridVisualNode<TDataType>::resetStates()
	{
		this->updateStates();
	}

	template <typename Coord>
	__global__ void AGVN_ComputeLeafPos(
		DArray<Coord> leafs_pos,
		DArray<AdaptiveGridNode> leafs)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= leafs.size()) return;

		leafs_pos[tId] = leafs[tId].m_position;
	}

	template <typename Coord>
	__global__ void AGVN_ComputeVertex(
		DArray<Coord> points,
		DArray<TopologyModule::Edge> edge,
		DArray<Coord> vertex,
		DArray<int> vertex_neighbor)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vertex.size()) return;

		points[tId] = vertex[tId];

		int nindex;
		nindex = vertex_neighbor[6 * tId] == EMPTY ? tId : vertex_neighbor[6 * tId];
		edge[6 * tId] = TopologyModule::Edge(tId, nindex);
		nindex = vertex_neighbor[6 * tId + 1] == EMPTY ? tId : vertex_neighbor[6 * tId + 1];
		edge[6 * tId + 1] = TopologyModule::Edge(tId, nindex);
		nindex = vertex_neighbor[6 * tId + 2] == EMPTY ? tId : vertex_neighbor[6 * tId + 2];
		edge[6 * tId + 2] = TopologyModule::Edge(tId, nindex);
		nindex = vertex_neighbor[6 * tId + 3] == EMPTY ? tId : vertex_neighbor[6 * tId + 3];
		edge[6 * tId + 3] = TopologyModule::Edge(tId, nindex);
		nindex = vertex_neighbor[6 * tId + 4] == EMPTY ? tId : vertex_neighbor[6 * tId + 4];
		edge[6 * tId + 4] = TopologyModule::Edge(tId, nindex);
		nindex = vertex_neighbor[6 * tId + 5] == EMPTY ? tId : vertex_neighbor[6 * tId + 5];
		edge[6 * tId + 5] = TopologyModule::Edge(tId, nindex);
	}

	template <typename Real>
	__global__ void AGVN_ComputeLeafValue(
		DArray<Real> leafs_value,
		DArray<int> n2v,
		DArray<Real> vertex_value)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= leafs_value.size()) return;

		leafs_value[tId] = (vertex_value[n2v[8 * tId]] + vertex_value[n2v[8 * tId + 1]] + vertex_value[n2v[8 * tId + 2]] + vertex_value[n2v[8 * tId + 3]] +
			vertex_value[n2v[8 * tId + 4]] + vertex_value[n2v[8 * tId + 5]] + vertex_value[n2v[8 * tId + 6]] + vertex_value[n2v[8 * tId + 7]]) * 0.125;
	}

	template<typename TDataType>
	void GLAdaptiveGridVisualNode<TDataType>::updateStates()
	{
		auto m_octree = this->inAdaptiveVolume()->getDataPtr();
		if (this->stateLeafs()->isEmpty()) this->stateLeafs()->allocate();
		if (this->stateLeafsValue()->isEmpty()) this->stateLeafsValue()->allocate();
		if (this->stateLeafsEdge()->isEmpty()) this->stateLeafsEdge()->allocate();
		auto& leaf_value = this->stateLeafsValue()->getData();

		DArray<Coord> vertex;
		DArray<int> vertex_neighbor, node2ver;
		m_octree->extractVertex(vertex, vertex_neighbor, node2ver);

		DArray<Coord> points_pos(vertex.size());
		DArray<TopologyModule::Edge> edges(vertex_neighbor.size());
		cuExecute(vertex.size(),
			AGVN_ComputeVertex,
			points_pos,
			edges,
			vertex,
			vertex_neighbor);
		vertex.clear();
		vertex_neighbor.clear();

		this->stateLeafsEdge()->getDataPtr()->setPoints(points_pos);
		this->stateLeafsEdge()->getDataPtr()->setEdges(edges);

		if (this->varType()->currentKey() == PointData::AGrid_Vertex)
		{
			if (!this->inAGridSDF()->isEmpty())
				leaf_value.assign(this->inAGridSDF()->getData());
			else
			{
				leaf_value.resize(points_pos.size());
				leaf_value.reset();
			}
		}
		else if (this->varType()->currentKey() == PointData::AGrid_Node)
		{
			DArray<AdaptiveGridNode> nodes;
			m_octree->extractLeafs(nodes);

			points_pos.resize(nodes.size());
			cuExecute(nodes.size(),
				AGVN_ComputeLeafPos,
				points_pos,
				nodes);
			nodes.clear();

			leaf_value.resize(points_pos.size());
			if (this->inAGridSDF()->isEmpty())
				leaf_value.reset();
			else
				cuExecute(leaf_value.size(),
					AGVN_ComputeLeafValue,
					leaf_value,
					node2ver,
					this->inAGridSDF()->getData());

		}
		this->stateLeafs()->getDataPtr()->setPoints(points_pos);

		node2ver.clear();
		points_pos.clear();
		edges.clear();
	}

	template<typename TDataType>
	bool GLAdaptiveGridVisualNode<TDataType>::validateInputs()
	{
		if (this->inAdaptiveVolume()->isEmpty())
			return false;
		return true;
	}

	DEFINE_CLASS(GLAdaptiveGridVisualNode);
}
