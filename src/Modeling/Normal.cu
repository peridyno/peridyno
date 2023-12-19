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

		//this->varLocation()->attach(callback);
		//this->varScale()->attach(callback);
		//this->varRotation()->attach(callback);

		this->varLength()->attach(callback);
		this->inTriangleSetIn()->attach(callback);


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

		glsource = std::make_shared<GLWireframeVisualModule>();
		this->inTriangleSetIn()->connect(glsource->inEdgeSet());
		this->graphicsPipeline()->pushModule(glsource);

	}

	template<typename TDataType>
	void Normal<TDataType>::renderChanged() 
	{
		printf("renderchange\n");

		glpoint->varVisible()->setValue(this->varShowPoints()->getValue());
		gledge->varVisible()->setValue(this->varShowEdges()->getValue());
		glsource->varVisible()->setValue(this->varShowWireframe()->getValue());

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
	void Normal<TDataType>::varChanged()
	{
		auto triSet = this->inTriangleSetIn()->getDataPtr();
		auto normalSet = this->stateNormalSet()->getDataPtr();
		int triNum = triSet->getTriangles().size();
		this->stateNormal()->resize(triNum);

		DArray<Triangle>& d_triangles = triSet->getTriangles();
		CArray<Triangle> c_triangles;
		c_triangles.assign(d_triangles);
		DArray<Coord>& d_points = triSet->getPoints();
		CArray<Coord> c_points;
		c_points.assign(d_points);


		////**********************************  build Normal by CPU **********************************////
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

		////**********************************  build Normal by Cuda **********************************////
		DArray<TopologyModule::Edge> d_edges;
		DArray<Coord> d_normalPt;
		{	
			d_edges.resize(d_triangles.size());
			d_normalPt.resize(d_triangles.size() * 2) ;
			cuExecute(d_triangles.size(),
				UpdateNormal,
				d_triangles,
				d_points,
				d_edges,
				d_normalPt,
				this->varLength()->getValue()
				);
		}

		normalSet->setPoints(d_normalPt);
		normalSet->setEdges(d_edges);
		normalSet->update();

	}


	template< typename Coord>
	__global__ void UpdateNormal(
		DArray<TopologyModule::Triangle> d_triangles,
		DArray<Coord> d_points,
		DArray<TopologyModule::Edge> edges,
		DArray<Coord> normal_points,
		float length)
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
		Coord dirNormal = ca.cross(cb).normalize() * -1 *length;
		normal_points[2 * pId] = Coord(x, y, z);
		normal_points[2 * pId + 1] = Coord(x, y, z) + dirNormal;

		edges[pId] = TopologyModule::Edge(2 * pId, 2 * pId + 1);
	}


	DEFINE_CLASS(Normal);
}