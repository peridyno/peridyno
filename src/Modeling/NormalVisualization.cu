#include "NormalVisualization.h"
#include "Topology/PointSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"
#include "BasicShapes/CylinderModel.h"
#include "BasicShapes/ConeModel.h"
#include "ColorMapping.h"

namespace dyno
{
	template<typename TDataType>
	NormalVisualization<TDataType>::NormalVisualization()
		: Node()
	{
		this->varLength()->setRange(0, 02);
		this->varLineWidth()->setRange(0,1);
		this->stateNormalSet()->setDataPtr(std::make_shared<EdgeSet<DataType3f>>());

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&NormalVisualization<TDataType>::varChanged, this));

		this->varLength()->attach(callback);
		this->varLineWidth()->attach(callback);
		this->varArrowResolution()->attach(callback);


		auto render_callback = std::make_shared<FCallBackFunc>(std::bind(&NormalVisualization<TDataType>::renderChanged, this));
		this->varLineWidth()->attach(render_callback);
		this->varShowWireframe()->attach(render_callback);


		this->varArrowResolution()->setRange(4,15);
		this->varLength()->setRange(0.1,10);

		this->inInNormal()->tagOptional(true);
		this->inScalar()->tagOptional(true);


		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		colorMapper->varMin()->setValue(-0.5);
		colorMapper->varMax()->setValue(0.5);
		this->inScalar()->connect(colorMapper->inScalar());
		this->graphicsPipeline()->pushModule(colorMapper);

		//cylinder
		this->stateArrowCylinder()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->stateTriangleCenter()->setDataPtr(std::make_shared<PointSet<TDataType>>());

		glInstanceCylinder = std::make_shared<GLInstanceVisualModule>();
		glInstanceCylinder->setColor(Color(0, 1, 0));

		this->stateArrowCylinder()->connect(glInstanceCylinder->inTriangleSet());
		this->stateTransformsCylinder()->connect(glInstanceCylinder->inInstanceTransform());
		colorMapper->outColor()->connect(glInstanceCylinder->inInstanceColor());
		this->graphicsPipeline()->pushModule(glInstanceCylinder);

		//cone
		this->stateArrowCone()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());


		glInstanceCone = std::make_shared<GLInstanceVisualModule>();
		glInstanceCone->setColor(Color(0, 1, 0));

		this->stateArrowCone()->connect(glInstanceCone->inTriangleSet());
		this->stateTransformsCone()->connect(glInstanceCone->inInstanceTransform());
		colorMapper->outColor()->connect(glInstanceCone->inInstanceColor());
		this->graphicsPipeline()->pushModule(glInstanceCone);

		this->stateNormal()->promoteOuput();
		this->stateTriangleCenter()->promoteOuput();
	}

	template<typename TDataType>
	void NormalVisualization<TDataType>::renderChanged()
	{

	}


	template<typename TDataType>
	void NormalVisualization<TDataType>::resetStates()
	{
		this->varChanged();
		this->renderChanged();
	}

	template<typename TDataType>
	void NormalVisualization<TDataType>::updateStates()
	{
		this->varChanged();
		//this->renderChanged();
	}

	template<typename TDataType>
	void NormalVisualization<TDataType>::varChanged()
	{		
		printf("varChanged\n");

		if (this->inTriangleSet()->isEmpty())
		{
			printf("Normal Node: Need input!\n");
			return;
		}

		auto normalSet = this->stateNormalSet()->getDataPtr();
		auto inTriSet = this->inTriangleSet()->getDataPtr();

		////**********************************  build Normal by Cuda **********************************////
		{
			DArray<TopologyModule::Triangle>& d_triangles = inTriSet->triangleIndices();
			d_points = inTriSet->getPoints();
			d_edges.resize(d_triangles.size());
			d_normalPt.resize(d_triangles.size() * 2);

			d_normal.resize(d_triangles.size());
			triangleCenter.resize(d_triangles.size());

			cuExecute(d_triangles.size(),
				UpdateTriangleNormal,
				d_triangles,
				d_points,
				d_edges,
				d_normalPt,
				d_normal,
				triangleCenter,
				this->varLength()->getValue(),
				this->varNormalize()->getValue()
			);

			normalSet->setPoints(d_normalPt);
			normalSet->setEdges(d_edges);
			normalSet->update();

			auto triCenter = this->stateTriangleCenter()->getDataPtr();
			triCenter->setPoints(triangleCenter);
			triCenter->update();

			this->stateNormal()->assign(d_normal);


		}


		//Build Arrow
		{	
			auto cylinder = std::make_shared<CylinderModel<DataType3f>>();
			cylinder->varColumns()->setValue(this->varArrowResolution()->getValue());
			cylinder->varEndSegment()->setValue(1);
			cylinder->varRow()->setValue(1);
			cylinder->varRadius()->setValue(this->varLineWidth()->getValue());
			cylinder->varHeight()->setValue(1.0f);
			cylinder->varLocation()->setValue(Coord(0.0f, 0.5f, 0.0f));
			auto cylinderTriSet = cylinder->stateTriangleSet()->getDataPtr();

			auto cone = std::make_shared<ConeModel<DataType3f>>();
			cone->varColumns()->setValue(this->varArrowResolution()->getValue());
			cone->varRadius()->setValue(this->varLineWidth()->getValue() * 2);
			cone->varHeight()->setValue(this->varLineWidth()->getValue() * 2 * 2);
			cone->varLocation()->setValue(Vec3f(0, this->varLineWidth()->getValue(), 0));
			cone->varRow()->setValue(1);
			auto coneTriSet = cone->stateTriangleSet()->getDataPtr();

			auto merge = std::make_shared<TriangleSet<DataType3f>>();
			merge->copyFrom(*coneTriSet->merge(*cylinderTriSet));

			this->stateArrowCylinder()->setDataPtr(cylinder->stateTriangleSet()->getDataPtr());

			DArray<Transform3f> transformCylinder;
			transformCylinder.resize(d_normalPt.size() / 2);
			cuExecute(d_normalPt.size()/2,
				UpdateArrowPoint,
				d_normalPt,
				transformCylinder,
				0,
				false,
				0
			);

			this->stateTransformsCylinder()->assign(transformCylinder);
			this->stateArrowCone()->setDataPtr(cone->stateTriangleSet()->getDataPtr());

			DArray<Transform3f> transformCone;
			transformCone.resize(d_normalPt.size() / 2);
			cuExecute(d_normalPt.size() / 2,
				UpdateArrowPoint,
				d_normalPt,
				transformCone,
				0,
				true,
				0
			);

			this->stateTransformsCone()->assign(transformCone);
		}

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
		DArray<Coord> normal,
		DArray<Coord> triangleCenter,
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
		triangleCenter[pId] = Coord(x, y, z);

		edges[pId] = TopologyModule::Edge(2 * pId, 2 * pId + 1);
		normal[pId] = dirNormal.normalize();
	}

	template< typename Coord>
	__global__ void UpdateArrowPoint(
		DArray<Coord> NormalPt,
		DArray<Transform3f> transform,
		int offest,
		bool moveToTop,
		float lengthOverride
		)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= NormalPt.size()/2) return;

		int i = pId;

		Coord root = NormalPt[i * 2];
		Coord head = NormalPt[i * 2 + 1];
		Vec3f defaultDirection = Vec3f(0, 1, 0);
		Vec3f direction = head - root;

		float length = direction.norm();
		if (lengthOverride != 0)
			length = lengthOverride;

		Vec3f distance = Vec3f(0,direction.norm(),0);

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

		Quat<Real> quat = Quat<Real>(angle, Axis);
		Vec3f location;
		Vec3f scale;

		if (moveToTop) 
		{
			location = root + offest + quat.rotate(distance);
			scale = Vec3f(1, 1, 1);
		}
		else 
		{
			location = root + offest;
			scale = Vec3f(1, length, 1);
		}

		transform[pId].translation() = location;
		transform[pId].scale() = scale;
		transform[pId].rotation() = quat.toMatrix3x3();
	}


	DEFINE_CLASS(NormalVisualization);
}