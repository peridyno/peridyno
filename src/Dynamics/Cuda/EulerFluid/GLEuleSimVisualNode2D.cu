#include "GLEuleSimVisualNode2D.h"
#include "Algorithm/Reduction.h"

#include "Topology/PointSet.h"
#include "GLPointVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "ColorMapping.h"
#include <thrust/sort.h>

namespace dyno
{
	IMPLEMENT_TCLASS(GLEuleSimVisualNode2D, TDataType)

	template<typename TDataType>
	GLEuleSimVisualNode2D<TDataType>::GLEuleSimVisualNode2D()
		: Node()
	{
		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		this->inLeafsValue()->connect(colorMapper->inScalar());
		colorMapper->varMin()->setValue(-1.0f);
		colorMapper->varMax()->setValue(1.0f);

		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->setColor(Color(1, 0, 0));
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
		ptRender->varPointSize()->setValue(0.002);
		this->inLeafs()->connect(ptRender->inPointSet());
		colorMapper->outColor()->connect(ptRender->inColor());

		auto wRender = std::make_shared<GLWireframeVisualModule>();
		wRender->setColor(Color(0, 0, 1));
		this->stateEdgeSet()->connect(wRender->inEdgeSet());

		this->graphicsPipeline()->pushModule(colorMapper);
		this->graphicsPipeline()->pushModule(ptRender);
		this->graphicsPipeline()->pushModule(wRender);
	}

	template<typename TDataType>
	GLEuleSimVisualNode2D<TDataType>::~GLEuleSimVisualNode2D()
	{
		printf("GLEuleSimVisualNode released \n");
	}

	template <typename Real, typename Coord>
	__global__ void SO_CountEdge2D(
		DArray<Coord> edge_pos,
		DArray<TopologyModule::Edge> edge_edge,
		DArray<Coord> pos,
		DArray<Coord> velocity,
		Real scale)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= pos.size()) return;

		edge_pos[tId] = pos[tId];
		edge_pos[(tId + pos.size())] = pos[tId] + scale * velocity[tId];
		edge_edge[tId] = TopologyModule::Edge(tId, (tId + pos.size()));

		//Real vl= velocity[tId].norm();
		//if (scale * vl > 0.036f)
		//	printf("Velocity Visual: %d %f %f %f; %f %f %f; %f \n", tId, pos[tId][0], pos[tId][1], pos[tId][2], velocity[tId][0], velocity[tId][1], velocity[tId][2], vl);
	}

	template<typename TDataType>
	void GLEuleSimVisualNode2D<TDataType>::resetStates()
	{
		this->update();
	}

	template<typename TDataType>
	void GLEuleSimVisualNode2D<TDataType>::updateStates()
	{
		auto& points_pos = this->inLeafs()->getDataPtr()->getPoints();

		Real edge_scale = 0.3f;
		DArray<Coord> edge_pos(2 * points_pos.size());
		edge_pos.reset();
		DArray<TopologyModule::Edge> edge_edge(points_pos.size());
		edge_edge.reset();
		cuExecute(points_pos.size(),
			SO_CountEdge2D,
			edge_pos,
			edge_edge,
			points_pos,
			this->inVelocity()->getData(),
			edge_scale);

		this->stateEdgeSet()->allocate();
		this->stateEdgeSet()->getDataPtr()->setPoints(edge_pos);
		this->stateEdgeSet()->getDataPtr()->setEdges(edge_edge);

		std::printf("GLEuleSimVisualNode2D: %d %d\n", points_pos.size(), edge_pos.size());
		edge_pos.clear();
		edge_edge.clear();
	}

	//template<typename TDataType>
	//bool GLEuleSimVisualNode2D<TDataType>::validateInputs()
	//{
	//	if (this->inAdaptiveVolume()->getDataPtr() == nullptr) 
	//	{
	//		return false;
	//	}

	//	return Node::validateInputs();
	//}

	DEFINE_CLASS(GLEuleSimVisualNode2D);
}
