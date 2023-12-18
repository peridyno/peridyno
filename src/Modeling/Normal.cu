#include "Normal.h"
#include "Topology/PointSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"
#include "Topology.h"
#include "CylinderModel.h"
#include "ConeModel.h"
#include "ColorMapping.h"

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
		this->varDebug()->attach(callback);
		this->varLineWidth()->attach(callback);
		this->varArrowResolution()->attach(callback);
		this->varLineMode()->attach(callback);

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

		this->stateArrow()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		glArrow = std::make_shared<GLSurfaceVisualModule>();
		//glArrow->varColorMode()->setCurrentKey(GLSurfaceVisualModule::CM_Vertex);
		this->stateArrow()->connect(glArrow->inTriangleSet());
		this->graphicsPipeline()->pushModule(glArrow);

		this->varArrowResolution()->setRange(4,15);
		this->varLength()->setRange(0.1,10);
		this->inTopology()->tagOptional(true);
		this->inInNormal()->tagOptional(true);
		this->inColor()->tagOptional(true);


		//auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		//colorMapper->varMin()->setValue(-0.5);
		//colorMapper->varMax()->setValue(0.5);
		//this->inColor()->connect(colorMapper->inScalar());
		//this->graphicsPipeline()->pushModule(colorMapper);
		//// 
		//// 
		//auto surfaceVisualizer = std::make_shared<GLSurfaceVisualModule>();
		//surfaceVisualizer->varColorMode()->getDataPtr()->setCurrentKey(1);
		//colorMapper->outColor()->connect(surfaceVisualizer->inColor());
		//this->stateArrow()->connect(surfaceVisualizer->inTriangleSet());
		//this->graphicsPipeline()->pushModule(surfaceVisualizer);

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
		printf("reset\n");
		this->varChanged();
		this->renderChanged();
	}

	template<typename TDataType>
	void Normal<TDataType>::updateStates()
	{
		printf("update\n");
		this->varChanged();
		//this->renderChanged();
	}

	template<typename TDataType>
	void Normal<TDataType>::varChanged()
	{		
		printf("varChanged\n");
		auto normalSet = this->stateNormalSet()->getDataPtr();

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
				d_points = inTriSet->getPoints();

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
				d_points = ptSet->getPoints();
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

		if (this->varLineMode()->getValue() != this->Arrow) 
		{
			auto arrowTriSet = this->stateArrow()->getDataPtr();
			arrowTriSet->getPoints().clear();
			arrowTriSet->getTriangles().clear();
			arrowTriSet->update();
			return;
		}
			

		auto cylinder = std::make_shared<CylinderModel<DataType3f>>();
		cylinder->varColumns()->setValue(this->varArrowResolution()->getValue());
		cylinder->varEndSegment()->setValue(1);
		cylinder->varRow()->setValue(1);
		cylinder->varRadius()->setValue(this->varLineWidth()->getValue());
		cylinder->varHeight()->setValue(1.0f);
		cylinder->varLocation()->setValue(Coord(0.0f, 0.5f, 0.0f));

		auto cylinderTriangles = cylinder->stateTriangleSet()->getDataPtr()->getTriangles();
		auto cylinderPoints = cylinder->stateTriangleSet()->getDataPtr()->getPoints();




		
		DArray<Coord> CylinderPoint;
		DArray<TopologyModule::Triangle> CylinderTriangle;
		DArray<Coord> singleCylinderPoint;
		DArray<TopologyModule::Triangle> singleCylinderTriangle;

		singleCylinderPoint.assign(cylinder->stateTriangleSet()->getDataPtr()->getPoints());
		singleCylinderTriangle.assign(cylinder->stateTriangleSet()->getDataPtr()->getTriangles());

		//CylinderPoint.assign(merge->getPoints());
		//CylinderTriangle.assign(merge->getTriangles());

		int normalNum = d_normalPt.size()/2;
		int singleCylinderPtNum = singleCylinderPoint.size();
		CylinderPoint.resize(singleCylinderPtNum * normalNum);

		//printf("normalNum : %d \nsingleArrowPtNum : %d\n", normalNum, singleArrowPtNum);
		//printf("arrowPoint : %d\n", arrowPoint.size());
		//printf("arrowTriangle : %d\n", arrowTriangle.size());

		cuExecute(CylinderPoint.size(),
			UpdateArrowPoint,
			CylinderPoint,
			d_normalPt,
			singleCylinderPoint,
			singleCylinderPtNum,
			0,
			this->varDebug()->getValue(),
			false,
			0
		);


		CylinderTriangle.resize(singleCylinderTriangle.size()* normalNum);
		printf("Num : %d \n ResizeNum : %d\n", singleCylinderTriangle.size(), CylinderTriangle.size());

		cuExecute(CylinderTriangle.size(),
			UpdateArrowTriangles,
			CylinderTriangle,
			singleCylinderTriangle,
			singleCylinderTriangle.size(),
			singleCylinderPoint.size(),
			this->varDebug()->getValue()
		);

		printf("pt : %d \n", CylinderPoint.size());
		printf("tri : %d\n", CylinderTriangle.size());


		auto cone = std::make_shared<ConeModel<DataType3f>>();
		cone->varColumns()->setValue(this->varArrowResolution()->getValue());
		cone->varRadius()->setValue(this->varLineWidth()->getValue() * 2); 
		cone->varHeight()->setValue(this->varLineWidth()->getValue() * 2 * 2);
		cone->varLocation()->setValue(Vec3f(0,this->varLineWidth()->getValue(),0));
		cone->varRow()->setValue(1);

		auto coneTriangles = cone->stateTriangleSet()->getDataPtr()->getTriangles();
		auto conePoints = cone->stateTriangleSet()->getDataPtr()->getPoints();

		DArray<Coord> conePoint;
		DArray<TopologyModule::Triangle> coneTriangle;
		DArray<Coord> singleConePoint;
		DArray<TopologyModule::Triangle> singleConeTriangle;

		singleConePoint.assign(conePoints);
		singleConeTriangle.assign(coneTriangles);

		int singleConePtNum = singleConePoint.size();
		conePoint.resize(singleConePtNum* normalNum);

		cuExecute(conePoint.size(),
			UpdateArrowPoint,
			conePoint,
			d_normalPt,
			singleConePoint,
			singleConePtNum,
			0,
			this->varDebug()->getValue(),
			true,
			1
		);




		coneTriangle.resize(singleConeTriangle.size()* normalNum);
		printf("Num : %d \n ResizeNum : %d\n", singleConeTriangle.size(), coneTriangle.size());

		cuExecute(coneTriangle.size(),
			UpdateArrowTriangles,
			coneTriangle,
			singleConeTriangle,
			singleConeTriangle.size(),
			singleConePoint.size(),
			this->varDebug()->getValue()
		);

		auto triset1 = std::make_shared<TriangleSet<DataType3f>>();
		triset1->setPoints(CylinderPoint);
		triset1->setTriangles(CylinderTriangle);

		auto triset2 = std::make_shared<TriangleSet<DataType3f>>();
		triset2->setPoints(conePoint);
		triset2->setTriangles(coneTriangle);

		auto merge = std::make_shared<TriangleSet<DataType3f>>();
		merge->copyFrom(*triset1->merge(*triset2));


		auto arrowTriSet = this->stateArrow()->getDataPtr();
		arrowTriSet->setPoints(merge->getPoints());
		arrowTriSet->setTriangles(merge->getTriangles());
		arrowTriSet->update();

	}

	template< typename Coord>
	__global__ void UpdatePointNormal(
		DArray<Coord> d_point,
		DArray<Coord> normal_points,
		DArray<Coord> normal,
		DArray<TopologyModule::Edge> edges,
		float length,
		bool normallization
		)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= d_point.size()) return;
		Coord dirNormal = normal[pId];

		if(normallization)
			dirNormal = dirNormal.normalize() * length;
		else
			dirNormal = dirNormal * length;

		normal_points[2 * pId] = d_point[pId] ;
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

	template< typename Coord>
	__global__ void UpdateArrowPoint(
		DArray<Coord> arrowPoints,
		DArray<Coord> NormalPt,
		DArray<Coord> singleArrowPoints,
		int arrowPtNum,
		int offest,
		int debug,
		bool moveToTop,
		float lengthOverride
		)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= arrowPoints.size()) return;

		int i = int(pId / arrowPtNum);
		/*if (i != debug)
			return;*/

		Coord root = NormalPt[i * 2];
		Coord head = NormalPt[i * 2 + 1];
		Vec3f defaultDirection = Vec3f(0, 1, 0);
		Vec3f direction = head - root;
		float length = direction.norm();
		float distance = direction.norm();

		direction.normalize();
		defaultDirection.normalize();
		Real angle;
		Vec3f Axis = direction.cross(defaultDirection);
		if (Axis == Vec3f(0, 0, 0) && direction[1] < 0)
		{
			Axis = Vec3f(1, 0, 0);
			angle = M_PI;
		}
		Axis.normalize();

		angle = -1 * acos(direction.dot(defaultDirection));

		Quat<Real> q = Quat<Real>(angle, Axis);

		Real x, y, z, w;
		x = q.x;
		y = q.y;
		z = q.z;
		w = q.w;

		Coord u(x, y, z);
		Real s = w;
		Coord tempPt;

		if (lengthOverride != 0)
			length = lengthOverride;
		
		if (moveToTop)
		{
			tempPt = Coord(singleArrowPoints[pId - i * arrowPtNum][0],
				singleArrowPoints[pId - i * arrowPtNum][1] * length + offest + distance,
				singleArrowPoints[pId - i * arrowPtNum][2]);
		}
		else 
		{
			tempPt = Coord(singleArrowPoints[pId - i * arrowPtNum][0],
				singleArrowPoints[pId - i * arrowPtNum][1] * length + offest,
				singleArrowPoints[pId - i * arrowPtNum][2]);
		}
	
		arrowPoints[pId] = 2.0f * u.dot(tempPt) * u+ (s * s - u.dot(u)) * tempPt + 2.0f * s * u.cross(tempPt) + root;
	
	}


	__global__ void UpdateArrowTriangles(
		DArray<TopologyModule::Triangle> arrowTriangles,
		DArray<TopologyModule::Triangle> singleArrowTriangles,
		int arrowTriNum,
		int singlePointNum,
		int debug
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= arrowTriangles.size()) return;

		int i = pId % arrowTriNum;
		int k = (pId / arrowTriNum);
		int z = (pId / arrowTriNum) * singlePointNum;
		TopologyModule::Triangle temp = singleArrowTriangles[i];

		arrowTriangles[pId] = TopologyModule::Triangle(temp[0] + z, temp[1] + z, temp[2] + z);
	}




	DEFINE_CLASS(Normal);
}