#include "Normal.h"
#include "Topology/PointSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"
#include "Topology.h"

namespace dyno
{
	template<typename TDataType>
	Normal<TDataType>::Normal()
		: ParametricModel<TDataType>()
	{
		this->varLength()->setRange(0, 02);
		this->varLineWidth()->setRange(0,1);
		this->stateNormalSet()->setDataPtr(std::make_shared<EdgeSet<DataType3f>>());

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&Normal<TDataType>::varChanged, this));

		this->varLength()->attach(callback);

		auto render_callback = std::make_shared<FCallBackFunc>(std::bind(&Normal<TDataType>::renderChanged, this));
		this->varLineMode()->attach(render_callback);
		this->varShowEdges()->attach(render_callback);
		this->varShowEdges()->attach(render_callback);
		this->varLineWidth()->attach(render_callback);
		this->varShowWireframe()->attach(render_callback);

		glpoint = std::make_shared<GLPointVisualModule>();
		this->stateNormalSet()->connect(glpoint->inPointSet());
		glpoint->varPointSize()->setValue(0.01);
		this->graphicsPipeline()->pushModule(glpoint);

		gledge = std::make_shared<GLWireframeVisualModule>();
		this->stateNormalSet()->connect(gledge->inEdgeSet());
		this->graphicsPipeline()->pushModule(gledge);

		this->inTopology()->tagOptional(true);
		this->inInNormal()->tagOptional(true);
	}

	template<typename TDataType>
	void Normal<TDataType>::renderChanged() 
	{
		glpoint->varVisible()->setValue(this->varShowPoints()->getValue());
		gledge->varVisible()->setValue(this->varShowEdges()->getValue());

		gledge->varRenderMode()->setCurrentKey(this->varLineMode()->getDataPtr()->currentKey());
		gledge->varRadius()->setValue(this->varLineWidth()->getValue());
		
	}


	template<typename TDataType>
	void Normal<TDataType>::resetStates()
	{
		this->varChanged();
		this->renderChanged();
	}

	template<typename TDataType>
	void Normal<TDataType>::updateStates()
	{
		this->varChanged();
		this->renderChanged();
	}

	template<typename TDataType>
	void Normal<TDataType>::varChanged()
	{
			
		auto normalSet = this->stateNormalSet()->getDataPtr();

		////**********************************  build Normal by CPU **********************************////
		//CArray<Triangle> c_triangles;
		//c_triangles.assign(d_triangles);
		//CArray<Coord> c_points;
		//c_points.assign(d_points);
		//std::vector<Coord> normalPt;
		//std::vector<TopologyModule::Edge> edges;

		//for (size_t i = 0; i < c_triangles.size(); i++)
		//{
		//	int a = c_triangles[i][0];
		//	int b = c_triangles[i][1];
		//	int c = c_triangles[i][2];
		//	Real x = (c_points[a][0] + c_points[b][0] + c_points[c][0]) / 3;
		//	Real y = (c_points[a][1] + c_points[b][1] + c_points[c][1]) / 3;
		//	Real z = (c_points[a][2] + c_points[b][2] + c_points[c][2]) / 3;
		//	
		//	Coord ca = c_points[b] - c_points[a];
		//	Coord cb = c_points[b] - c_points[c];
		//	Coord dirNormal = ca.cross(cb).normalize() * -1 * this->varLength()->getValue();


		//	normalPt.push_back(Coord(x, y, z));
		//	normalPt.push_back(Coord(x, y, z)+ dirNormal);
		//	
		//	edges.push_back(TopologyModule::Edge(i * 2, i * 2 + 1));
		//	
		//}
		// 
		//normalSet->setPoints(normalPt);
		//normalSet->setEdges(edges);
		//normalSet->update();

		DArray<TopologyModule::Edge> d_edges;
		DArray<Coord> d_normalPt;


		auto inTriSet = TypeInfo::cast<TriangleSet<DataType3f>>(this->inTopology()->getDataPtr());
		if (inTriSet != nullptr) 
		{
			if (this->inTopology()->isEmpty())
			{
				printf("Normal Node: Need input!\n");
				return;
			}
			////**********************************  build Normal by Cuda **********************************////
			{
				DArray<TopologyModule::Triangle>& d_triangles = inTriSet->getTriangles();
				DArray<Coord>& d_points = inTriSet->getPoints();

				d_edges.resize(d_triangles.size());
				d_normalPt.resize(d_triangles.size() * 2);
				cuExecute(d_triangles.size(),
					UpdateTriangleNormal,
					d_triangles,
					d_points,
					d_edges,
					d_normalPt,
					this->varLength()->getValue(),
					this->varNormalize()->getValue()
				);
			}
			normalSet->setPoints(d_normalPt);
			normalSet->setEdges(d_edges);
			normalSet->update();
		}
		else 
		{
			if (TypeInfo::cast<PointSet<DataType3f>>(this->inTopology()->getDataPtr()) != nullptr)
			{
				auto ptSet = TypeInfo::cast<PointSet<DataType3f>>(this->inTopology()->getDataPtr());
				if (this->inInNormal()->isEmpty() | this->inTopology()->isEmpty())
				{
					printf("Normal Node: Need input!\n");
					return;
				}
				DArray<Coord>& d_points = ptSet->getPoints();
				d_edges.resize(d_points.size());
				d_normalPt.resize(d_points.size() * 2);
				cuExecute(d_points.size(),
					UpdatePointNormal,
					ptSet->getPoints(),
					d_normalPt, 
					this->inInNormal()->getData(),
					d_edges,
					this->varLength()->getValue(),
					this->varNormalize()->getValue()
				);
			}
			normalSet->setPoints(d_normalPt);
			normalSet->setEdges(d_edges);
			normalSet->update();
		}
	}

	template< typename Coord>
	__global__ void UpdatePointNormal(
		DArray<Coord> d_point,
		DArray<Coord> normal_points,
		DArray<Coord> normal,
		DArray<TopologyModule::Edge> edges,
		float length,
		bool normallization)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= d_point.size()) return;
		Coord dirNormal = normal[pId];

		if(normallization)
			dirNormal = dirNormal.normalize() * length;
		else
			dirNormal = dirNormal * length;

		normal_points[2 * pId] = d_point[pId];
		normal_points[2 * pId + 1] = d_point[pId] + dirNormal;

		edges[pId] = TopologyModule::Edge(2 * pId, 2 * pId + 1);
	}


	template< typename Coord>
	__global__ void UpdateTriangleNormal(
		DArray<TopologyModule::Triangle> d_triangles,
		DArray<Coord> d_points,
		DArray<TopologyModule::Edge> edges,
		DArray<Coord> normal_points,
		float length,
		bool normalization)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= d_triangles.size()) return;

		int a = d_triangles[pId][0];
		int b = d_triangles[pId][1];
		int c = d_triangles[pId][2];

		float x = (d_points[a][0] + d_points[b][0] + d_points[c][0]) / 3;
		float y = (d_points[a][1] + d_points[b][1] + d_points[c][1]) / 3;
		float z = (d_points[a][2] + d_points[b][2] + d_points[c][2]) / 3;
		 
		Coord ca = d_points[b] - d_points[a];
		Coord cb = d_points[b] - d_points[c];
		Coord dirNormal;

		if(normalization)
			dirNormal = ca.cross(cb).normalize() * -1 *length;
		else
			dirNormal = ca.cross(cb) * -1 * length;

		normal_points[2 * pId] = Coord(x, y, z);
		normal_points[2 * pId + 1] = Coord(x, y, z) + dirNormal;

		edges[pId] = TopologyModule::Edge(2 * pId, 2 * pId + 1);
	}


	DEFINE_CLASS(Normal);
}