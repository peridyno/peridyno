#include "GLAGridVisualNode.h"
#include "Algorithm/Reduction.h"

#include "Topology/PointSet.h"
#include "GLPointVisualModule.h"
#include "ColorMapping.h"
#include "../GUI/ImWidgets/ImColorbar.h"
#include <thrust/sort.h>

namespace dyno
{
	IMPLEMENT_TCLASS(GLAGridVisualNode, TDataType)

	template<typename TDataType>
	GLAGridVisualNode<TDataType>::GLAGridVisualNode()
		: Node()
	{
		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		this->stateLeafsValue()->connect(colorMapper->inScalar());
		colorMapper->varMin()->setValue(-1.5f);
		colorMapper->varMax()->setValue(2.5f);

		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->setColor(Color(1, 0, 0));
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
		ptRender->varPointSize()->setValue(0.003);

		this->stateLeafs()->connect(ptRender->inPointSet());
		colorMapper->outColor()->connect(ptRender->inColor());

		//// A simple color bar widget for node
		//auto colorBar = std::make_shared<ImColorbar>();
		//this->stateLeafsValue()->connect(colorBar->inScalar());
		//colorBar->varIsfix()->setValue(true);
		//colorBar->varMin()->setValue(-1.5f);
		//colorBar->varMax()->setValue(2.5f);

		this->graphicsPipeline()->pushModule(colorMapper);
		this->graphicsPipeline()->pushModule(ptRender);
		//this->graphicsPipeline()->pushModule(colorBar);
	}

	template<typename TDataType>
	GLAGridVisualNode<TDataType>::~GLAGridVisualNode()
	{
		printf("GLAGridVisualNode released \n");
	}
	 
	template<typename TDataType>
	void GLAGridVisualNode<TDataType>::resetStates()
	{
		this->update();
	}

	template <typename Coord>
	__global__ void SO_ComputeLeafPos(
		DArray<Coord> leafs_pos,
		DArray<AdaptiveGridNode> leafs)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= leafs.size()) return;

		leafs_pos[tId] = leafs[tId].m_position;
	}
	template<typename TDataType>
	void GLAGridVisualNode<TDataType>::updateStates()
	{
		auto m_octree = this->inAdaptiveVolume()->getDataPtr();
		DArray<AdaptiveGridNode> nodes;
		m_octree->extractLeafs(nodes);

		DArray<Coord> points_pos(nodes.size());
		cuExecute(nodes.size(),
			SO_ComputeLeafPos,
			points_pos,
			nodes);
		nodes.clear();

		this->stateLeafs()->allocate();
		this->stateLeafs()->getDataPtr()->setPoints(points_pos);
		DArray<Real> points_sdf(points_pos.size());
		points_sdf.reset();
		this->stateLeafsValue()->allocate();
		this->stateLeafsValue()->assign(points_sdf);
		//this->stateLeafPosition()->allocate();
		//this->stateLeafPosition()->assign(points_pos);
		//this->stateLeafScale()->allocate();
		//this->stateLeafScale()->assign(points_scale);

		points_pos.clear();
		points_sdf.clear();
	}

	template<typename TDataType>
	bool GLAGridVisualNode<TDataType>::validateInputs()
	{
		if (this->inAdaptiveVolume()->isEmpty()) {
			return false;
		}

		return Node::validateInputs();
	}

	DEFINE_CLASS(GLAGridVisualNode);
}
