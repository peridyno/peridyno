#include "GLAdaptiveWaterVisualNode.h"
#include "Algorithm/Reduction.h"

#include "Module/CalculateNorm.h"
#include "Topology/PointSet.h"
//#include "Topology/GridSet.h"
#include "GLPointVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLSurfaceVisualModule.h"
#include "ColorMapping.h"
#include "Mapping/Extract.h"

#include <thrust/sort.h>

namespace dyno
{
	IMPLEMENT_TCLASS(GLAdaptiveWaterVisualNode, TDataType)

	template<typename TDataType>
	GLAdaptiveWaterVisualNode<TDataType>::GLAdaptiveWaterVisualNode()
		: Node()
	{
		auto exES = std::make_shared<ExtractTriangleSetFromPolygonSet<DataType3f>>();
		this->statePolygonSet()->connect(exES->inPolygonSet());
		this->graphicsPipeline()->pushModule(exES);

		auto surfaceRender = std::make_shared<GLSurfaceVisualModule>();
		exES->outTriangleSet()->connect(surfaceRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(surfaceRender);
	}

	template<typename TDataType>
	GLAdaptiveWaterVisualNode<TDataType>::~GLAdaptiveWaterVisualNode()
	{
	}

	template <typename Real, typename Coord2D, typename Coord3D>
	__global__ void AW_CountNodes2D(
		DArray<Coord3D> edge_points,
		DArray<TopologyModule::Edge> edge,
		DArray<OcKey> seeds,
		Coord2D origin,
		Real dx)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= seeds.size()) return;

		OcIndex gnx, gny;
		RecoverFromMortonCode2D(seeds[tId], gnx, gny);

		Coord3D pos(origin[0] + gnx * dx, 8.0f, origin[1] + gny * dx);

		edge_points[4 * tId] = pos;
		edge_points[4 * tId + 1] = pos + Coord3D(dx, 0.0, 0.0);
		edge_points[4 * tId + 2] = pos + Coord3D(dx, 0.0, dx);
		edge_points[4 * tId + 3] = pos + Coord3D(0.0, 0.0, dx);

		edge[4 * tId] = TopologyModule::Edge(4 * tId, 4 * tId + 1);
		edge[4 * tId + 1] = TopologyModule::Edge(4 * tId + 1, 4 * tId + 2);
		edge[4 * tId + 2] = TopologyModule::Edge(4 * tId + 2, 4 * tId + 3);
		edge[4 * tId + 3] = TopologyModule::Edge(4 * tId + 3, 4 * tId);		
	}

	template<typename TDataType>
	void GLAdaptiveWaterVisualNode<TDataType>::resetStates()
	{
		auto volumeSet = this->inAGridSet()->constDataPtr();
		DArray<Coord3D> edge_points;
		DArray<TopologyModule::Edge> edges;
		if (!this->stateSeedMorton()->isEmpty())
		{
			auto dx = volumeSet->adaptiveGridDx2D();
			auto origin = volumeSet->adaptiveGridOrigin2D();
			auto& m_seed = this->stateSeedMorton()->getData();
			edge_points.resize(4 * m_seed.size());
			edges.resize(4 * m_seed.size());
			cuExecute(m_seed.size(),
				AW_CountNodes2D,
				edge_points,
				edges,
				m_seed,
				origin,
				dx);

			this->stateSeeds()->allocate();
			this->stateSeeds()->getDataPtr()->setPoints(edge_points);
			this->stateSeeds()->getDataPtr()->setEdges(edges);
		}
		edge_points.clear();
		edges.clear();

		this->statePolygonSet()->allocate();
		generateWaterSurface();
	}

	template<typename TDataType>
	void GLAdaptiveWaterVisualNode<TDataType>::updateStates()
	{
		this->resetStates();
	}

	template<typename Real, typename Coord2D, typename Coord4D>
	__global__ void AW_InterpolateVertex(
		DArray<Real> vheight,
		DArray<int> vnum,
		DArray<uint> snum,
		DArray<Coord2D> nodes,
		DArray<Coord4D> heights,
		DArrayList<int> neighbor,
		DArray<int> n2v)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		//if (abs(heights[tId][0] + heights[tId][3]) < REAL_EPSILON) return;

		atomicAdd(&vheight[n2v[4 * tId]], (heights[tId][0] + heights[tId][3]));
		atomicAdd(&vheight[n2v[4 * tId+1]], (heights[tId][0] + heights[tId][3]));
		atomicAdd(&vheight[n2v[4 * tId+2]], (heights[tId][0] + heights[tId][3]));
		atomicAdd(&vheight[n2v[4 * tId+3]], (heights[tId][0] + heights[tId][3]));
		atomicAdd(&vnum[n2v[4 * tId]], 1);
		atomicAdd(&vnum[n2v[4 * tId + 1]], 1);
		atomicAdd(&vnum[n2v[4 * tId + 2]], 1);
		atomicAdd(&vnum[n2v[4 * tId + 3]], 1);

		snum[tId] = 4;
		// -x
		if (neighbor[4 * tId].size() > 1)
		{
			snum[tId]++;
			int nindex = neighbor[4 * tId][0];
			if (nodes[nindex][1] > nodes[tId][1])
			{
				atomicAdd(&vheight[n2v[4 * nindex + 1]], (heights[tId][0] + heights[tId][3]));
				atomicAdd(&vnum[n2v[4 * nindex + 1]], 1);	
			}
			else
			{
				atomicAdd(&vheight[n2v[4 * nindex + 2]], (heights[tId][0] + heights[tId][3]));
				atomicAdd(&vnum[n2v[4 * nindex + 2]], 1);
			}
		}
		// +x
		if (neighbor[4 * tId + 1].size() > 1)
		{
			snum[tId]++;
			int nindex = neighbor[4 * tId + 1][0];
			if (nodes[nindex][1] > nodes[tId][1])
			{
				atomicAdd(&vheight[n2v[4 * nindex]], (heights[tId][0] + heights[tId][3]));
				atomicAdd(&vnum[n2v[4 * nindex]], 1);
			}
			else
			{
				atomicAdd(&vheight[n2v[4 * nindex + 3]], (heights[tId][0] + heights[tId][3]));
				atomicAdd(&vnum[n2v[4 * nindex + 3]], 1);
			}
		}
		// -y
		if (neighbor[4 * tId + 2].size() > 1)
		{
			snum[tId]++;
			int nindex = neighbor[4 * tId + 2][0];
			if (nodes[nindex][0] > nodes[tId][0])
			{
				atomicAdd(&vheight[n2v[4 * nindex + 3]], (heights[tId][0] + heights[tId][3]));
				atomicAdd(&vnum[n2v[4 * nindex + 3]], 1);
			}
			else
			{
				atomicAdd(&vheight[n2v[4 * nindex + 2]], (heights[tId][0] + heights[tId][3]));
				atomicAdd(&vnum[n2v[4 * nindex + 2]], 1);
			}
		}
		// +y
		if (neighbor[4 * tId + 3].size() > 1)
		{
			snum[tId]++;
			int nindex = neighbor[4 * tId + 3][0];
			if (nodes[nindex][0] > nodes[tId][0])
			{
				atomicAdd(&vheight[n2v[4 * nindex]], (heights[tId][0] + heights[tId][3]));
				atomicAdd(&vnum[n2v[4 * nindex]], 1);
			}
			else
			{
				atomicAdd(&vheight[n2v[4 * nindex + 1]], (heights[tId][0] + heights[tId][3]));
				atomicAdd(&vnum[n2v[4 * nindex + 1]], 1);
			}
		}
	}

	template<typename Real, typename Coord2D, typename Coord3D>
	__global__ void AW_ComputeVertex(
		DArray<Coord3D> tri_points,
		DArray<Real> vheight,
		DArray<int> vnum,
		DArray<Coord2D> vertexs,
		Real offset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vertexs.size()) return;

		Real py;
		if (vnum[tId] == 0)	py = 0.0f;
		else py = vheight[tId] / vnum[tId];

		Coord3D pos(vertexs[tId][0], py + offset, vertexs[tId][1]);
		tri_points[tId] = pos;
	}

	template<typename Coord2D>
	GPU_FUNC void AW_StartFromXMinus(
		List<uint>& list_i,
		int tId,
		DArray<Coord2D>& nodes,
		DArrayList<int>& neighbor,
		DArray<int>& n2v)
	{
		// -x
		if (neighbor[4 * tId].size() > 1)
		{
			int nindex = neighbor[4 * tId][0];
			int v = nodes[nindex][1] > nodes[tId][1] ? n2v[4 * nindex + 1] : n2v[4 * nindex + 2];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 0]);
		// -y
		if (neighbor[4 * tId + 2].size() > 1)
		{
			int nindex = neighbor[4 * tId + 2][0];
			int v = nodes[nindex][0] > nodes[tId][0] ? n2v[4 * nindex + 3] : n2v[4 * nindex + 2];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 1]);
		//+x
		if (neighbor[4 * tId + 1].size() > 1)
		{
			int nindex = neighbor[4 * tId + 1][0];
			int v = nodes[nindex][1] > nodes[tId][1] ? n2v[4 * nindex] : n2v[4 * nindex + 3];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 2]);
		// +y
		if (neighbor[4 * tId + 3].size() > 1)
		{
			int nindex = neighbor[4 * tId + 3][0];
			int v = nodes[nindex][0] > nodes[tId][0] ? n2v[4 * nindex] : n2v[4 * nindex + 1];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 3]);
	}
	template<typename Coord2D>
	GPU_FUNC void AW_StartFromYMinus(
		List<uint>& list_i,
		int tId,
		DArray<Coord2D>& nodes,
		DArrayList<int>& neighbor,
		DArray<int>& n2v)
	{
		// -y
		if (neighbor[4 * tId + 2].size() > 1)
		{
			int nindex = neighbor[4 * tId + 2][0];
			int v = nodes[nindex][0] > nodes[tId][0] ? n2v[4 * nindex + 3] : n2v[4 * nindex + 2];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 1]);
		//+x
		if (neighbor[4 * tId + 1].size() > 1)
		{
			int nindex = neighbor[4 * tId + 1][0];
			int v = nodes[nindex][1] > nodes[tId][1] ? n2v[4 * nindex] : n2v[4 * nindex + 3];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 2]);
		// +y
		if (neighbor[4 * tId + 3].size() > 1)
		{
			int nindex = neighbor[4 * tId + 3][0];
			int v = nodes[nindex][0] > nodes[tId][0] ? n2v[4 * nindex] : n2v[4 * nindex + 1];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 3]);
		// -x
		if (neighbor[4 * tId].size() > 1)
		{
			int nindex = neighbor[4 * tId][0];
			int v = nodes[nindex][1] > nodes[tId][1] ? n2v[4 * nindex + 1] : n2v[4 * nindex + 2];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 0]);
	}
	template<typename Coord2D>
	GPU_FUNC void AW_StartFromXPlus(
		List<uint>& list_i,
		int tId,
		DArray<Coord2D>& nodes,
		DArrayList<int>& neighbor,
		DArray<int>& n2v)
	{
		//+x
		if (neighbor[4 * tId + 1].size() > 1)
		{
			int nindex = neighbor[4 * tId + 1][0];
			int v = nodes[nindex][1] > nodes[tId][1] ? n2v[4 * nindex] : n2v[4 * nindex + 3];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 2]);
		// +y
		if (neighbor[4 * tId + 3].size() > 1)
		{
			int nindex = neighbor[4 * tId + 3][0];
			int v = nodes[nindex][0] > nodes[tId][0] ? n2v[4 * nindex] : n2v[4 * nindex + 1];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 3]);
		// -x
		if (neighbor[4 * tId].size() > 1)
		{
			int nindex = neighbor[4 * tId][0];
			int v = nodes[nindex][1] > nodes[tId][1] ? n2v[4 * nindex + 1] : n2v[4 * nindex + 2];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 0]);
		// -y
		if (neighbor[4 * tId + 2].size() > 1)
		{
			int nindex = neighbor[4 * tId + 2][0];
			int v = nodes[nindex][0] > nodes[tId][0] ? n2v[4 * nindex + 3] : n2v[4 * nindex + 2];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 1]);
	}
	template<typename Coord2D>
	GPU_FUNC void AW_StartFromYPlus(
		List<uint>& list_i,
		int tId,
		DArray<Coord2D>& nodes,
		DArrayList<int>& neighbor,
		DArray<int>& n2v)
	{
		// +y
		if (neighbor[4 * tId + 3].size() > 1)
		{
			int nindex = neighbor[4 * tId + 3][0];
			int v = nodes[nindex][0] > nodes[tId][0] ? n2v[4 * nindex] : n2v[4 * nindex + 1];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 3]);
		// -x
		if (neighbor[4 * tId].size() > 1)
		{
			int nindex = neighbor[4 * tId][0];
			int v = nodes[nindex][1] > nodes[tId][1] ? n2v[4 * nindex + 1] : n2v[4 * nindex + 2];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 0]);
		// -y
		if (neighbor[4 * tId + 2].size() > 1)
		{
			int nindex = neighbor[4 * tId + 2][0];
			int v = nodes[nindex][0] > nodes[tId][0] ? n2v[4 * nindex + 3] : n2v[4 * nindex + 2];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 1]);
		//+x
		if (neighbor[4 * tId + 1].size() > 1)
		{
			int nindex = neighbor[4 * tId + 1][0];
			int v = nodes[nindex][1] > nodes[tId][1] ? n2v[4 * nindex] : n2v[4 * nindex + 3];
			list_i.insert(v);
		}
		list_i.insert(n2v[4 * tId + 2]);
	}

	template<typename Coord2D>
	__global__ void AW_SetupPolySetIndices(
		DArrayList<uint> indices,
		DArray<uint> snum,
		DArray<Coord2D> nodes,
		DArrayList<int> neighbor,
		DArray<int> n2v)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		auto& list_i = indices[tId];
		if (list_i.max_size() == 0) return;

		// -x
		if (neighbor[4 * tId].size() > 1)
			AW_StartFromXMinus(list_i, tId, nodes, neighbor, n2v);
		// -y
		else if (neighbor[4 * tId + 2].size() > 1)
			AW_StartFromYMinus(list_i, tId, nodes, neighbor, n2v);
		// +x
		else if (neighbor[4 * tId + 1].size() > 1)
			AW_StartFromXPlus(list_i, tId, nodes, neighbor, n2v);
		// +y
		else if (neighbor[4 * tId + 3].size() > 1)
			AW_StartFromYPlus(list_i, tId, nodes, neighbor, n2v);
		else
		{
			list_i.insert(n2v[4 * tId]);
			list_i.insert(n2v[4 * tId + 1]);
			list_i.insert(n2v[4 * tId + 2]);
			list_i.insert(n2v[4 * tId + 3]);
		}


		//if (tId == 12)
		//	printf("what hapend! %d %f %f, %d %d %d %d %d %d \n", tId, nodes[tId][0], nodes[tId][1], list_i.size(),
		//		list_i[0], list_i[1], list_i[2], list_i[3], list_i[4]);
	}

	template<typename TDataType>
	void GLAdaptiveWaterVisualNode<TDataType>::generateWaterSurface()
	{
		auto& heights = this->inGrid()->constData();

		auto volumeSet = this->inAGridSet()->constDataPtr();
		DArray<Coord2D> nodes;
		DArrayList<int> neighbor;
		volumeSet->extractLeafs(nodes, neighbor);
		if (nodes.size() == 0) return;
		DArray<Coord2D> gridVertices;
		DArray<int> n2v;
		DArrayList<int> v2n;
		volumeSet->extractVertexs(gridVertices, n2v, v2n);
		v2n.clear();

		DArray<Real> vheight(gridVertices.size());
		DArray<int> vnum(gridVertices.size());
		vheight.reset();
		vnum.reset();

		DArray<uint> polygonVertexCounter(nodes.size());
		polygonVertexCounter.reset();
		DArray<Coord3D> polygonVertices(gridVertices.size());
		polygonVertices.reset();
		cuExecute(nodes.size(),
			AW_InterpolateVertex,
			vheight,
			vnum,
			polygonVertexCounter,
			nodes,
			heights,
			neighbor,
			n2v);

		cuExecute(gridVertices.size(),
			AW_ComputeVertex,
			polygonVertices,
			vheight,
			vnum,
			gridVertices,
			this->varWaterOffset()->getData());

		DArrayList<uint> polygonIndices;
		polygonIndices.resize(polygonVertexCounter);

		cuExecute(nodes.size(),
			AW_SetupPolySetIndices,
			polygonIndices,
			polygonVertexCounter,
			nodes,
			neighbor,
			n2v);

		auto ps = this->statePolygonSet()->getDataPtr();
		ps->setPoints(polygonVertices);
		ps->setPolygons(polygonIndices);
		ps->update();

		nodes.clear();
		neighbor.clear();
		n2v.clear();

		polygonVertexCounter.clear();
		polygonVertices.clear();
		polygonIndices.clear();
		gridVertices.clear();
		vheight.clear();
		vnum.clear();
	}

	template<typename TDataType>
	bool GLAdaptiveWaterVisualNode<TDataType>::validateInputs()
	{
		return Node::validateInputs();
	}

	DEFINE_CLASS(GLAdaptiveWaterVisualNode);
}
