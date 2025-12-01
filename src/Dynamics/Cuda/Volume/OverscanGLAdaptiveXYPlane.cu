#include "OverscanGLAdaptiveXYPlane.h"
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
	IMPLEMENT_TCLASS(OverscanGLAdaptiveXYPlane, TDataType)

		template<typename TDataType>
	OverscanGLAdaptiveXYPlane<TDataType>::OverscanGLAdaptiveXYPlane()
		: Node()
	{


		auto wRender = std::make_shared<GLWireframeVisualModule>();
		this->stateGrids()->connect(wRender->inEdgeSet());

		this->graphicsPipeline()->pushModule(wRender);

	}

	template<typename TDataType>
	OverscanGLAdaptiveXYPlane<TDataType>::~OverscanGLAdaptiveXYPlane()
	{
		printf("OverscanGLAdaptiveXYPlane released \n");
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

		edge_points[4 * tId] = pos + Coord(-0.5 * up_dx, -0.5 * up_dx, 0.0);
		edge_points[4 * tId + 1] = pos + Coord(-0.5 * up_dx, 0.5 * up_dx, 0.0);
		edge_points[4 * tId + 2] = pos + Coord(0.5 * up_dx, 0.5 * up_dx, 0.0);
		edge_points[4 * tId + 3] = pos + Coord(0.5 * up_dx, -0.5 * up_dx, 0.0);

		edge[4 * tId] = TopologyModule::Edge(4 * tId, 4 * tId + 1);
		edge[4 * tId + 1] = TopologyModule::Edge(4 * tId + 1, 4 * tId + 2);
		edge[4 * tId + 2] = TopologyModule::Edge(4 * tId + 2, 4 * tId + 3);
		edge[4 * tId + 3] = TopologyModule::Edge(4 * tId + 3, 4 * tId);
	}

	template<typename TDataType>
	void OverscanGLAdaptiveXYPlane<TDataType>::resetStates()
	{
		mFrameNumber = 0;
		mTempPlane = this->varLowerBound()->getValue();

		DArray<AdaptiveGridNode> node;
		DArrayList<int> neighbor;
		auto volumeSet = this->inAdaptiveVolume()->getDataPtr();
		Real zpos = this->varZPos()->getData();
		volumeSet->extractLeafs2D(node, neighbor, zpos);
		Level level = volumeSet->adaptiveGridLevelNum();
		Real dx = volumeSet->adaptiveGridDx();

		DArray<Coord> edge_points(4 * node.size());
		DArray<TopologyModule::Edge> edges(4 * node.size());
		cuExecute(node.size(),
			ESS_CountNodes2D,
			edge_points,
			edges,
			node,
			dx,
			level,
			zpos);


		this->stateGrids()->allocate();
		this->stateGrids()->getDataPtr()->setPoints(edge_points);
		this->stateGrids()->getDataPtr()->setEdges(edges);

		edge_points.clear();

		node.clear();
		neighbor.clear();

		edges.clear();
	}




	template<typename TDataType>
	void OverscanGLAdaptiveXYPlane<TDataType>::updateStates()
	{
		DArray<AdaptiveGridNode> node;
		DArrayList<int> neighbor;
		auto volumeSet = this->inAdaptiveVolume()->getDataPtr();

		Real zpos = this->varZPos()->getData();
		Real offset = this->varLowerBound()->getValue();
		std::cout << mFrameNumber << ", " << this->varFirstFrame()->getValue() << std::endl;

		Real lower = this->varLowerBound()->getValue();
		Real upper = this->varUpperBound()->getValue();


		if ((mFrameNumber > this->varFirstFrame()->getValue())
			&& (mFrameNumber % this->varMovingIntervalFrame()->getValue()) == 0)
		{
			Real step_value = (mTempPlane - this->varLowerBound()->getValue()) /
				abs(this->varUpperBound()->getValue() - this->varLowerBound()->getValue());

			if ((mTempPlane < this->varUpperBound()->getValue()) && (mScanDirection))
			{
				//mTempPlane += this->varMovingStep()->getValue();
				mTempPlane += Interim(step_value);
			}
			else if ((mTempPlane > this->varLowerBound()->getValue()) && (!mScanDirection))
			{
				//mTempPlane -= this->varMovingStep()->getValue();
				mTempPlane -= Interim(step_value);
			}
			else if (mTempPlane > this->varUpperBound()->getValue())
			{
				mScanDirection = false;
				//mTempPlane = this->varLowerBound()->getValue();
			}
			else if (mTempPlane < this->varLowerBound()->getValue())
			{
				mScanDirection = true;
				//mTempPlane = this->varLowerBound()->getValue();
			}

			zpos = mTempPlane;
			std::cout << "Scan: " << zpos << std::endl;
		}


		volumeSet->extractLeafs2D(node, neighbor, zpos);
		Level level = volumeSet->adaptiveGridLevelNum();
		Real dx = volumeSet->adaptiveGridDx();

		DArray<Coord> edge_points(4 * node.size());
		DArray<TopologyModule::Edge> edges(4 * node.size());
		cuExecute(node.size(),
			ESS_CountNodes2D,
			edge_points,
			edges,
			node,
			dx,
			level,
			zpos);


		this->stateGrids()->allocate();
		this->stateGrids()->getDataPtr()->setPoints(edge_points);
		this->stateGrids()->getDataPtr()->setEdges(edges);

		edge_points.clear();

		node.clear();
		neighbor.clear();

		edges.clear();
		mFrameNumber++;
	}

	template<typename TDataType>
	bool OverscanGLAdaptiveXYPlane<TDataType>::validateInputs()
	{
		if (this->inAdaptiveVolume()->isEmpty()) {
			return false;
		}

		return Node::validateInputs();
	}

	DEFINE_CLASS(OverscanGLAdaptiveXYPlane);
}
