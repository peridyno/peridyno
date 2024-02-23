#include "Transform.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"



namespace dyno
{
	template<typename TDataType>
	TransformModel<TDataType>::TransformModel()
		: ParametricModel<TDataType>()
	{
		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->statePointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		this->stateEdgeSet()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());

		glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.52f, 0.25f));
		glModule->setVisible(true);

		glWireModule = std::make_shared<GLWireframeVisualModule>();
		glWireModule->varLineWidth()->setValue(1);

		glPointModule = std::make_shared<GLPointVisualModule>();
		glPointModule->varPointSize()->setValue(0.007);

		this->stateTriangleSet()->connect(glModule->inTriangleSet());
		this->stateEdgeSet()->connect(glWireModule->inEdgeSet());
		this->statePointSet()->connect(glPointModule->inPointSet());

		this->graphicsPipeline()->pushModule(glModule);
		this->graphicsPipeline()->pushModule(glWireModule);
		this->graphicsPipeline()->pushModule(glPointModule);
		
	}

	template<typename TDataType>
	void TransformModel<TDataType>::resetStates()
	{
		Transform();
	}

	template<typename TDataType>
	void TransformModel<TDataType>::disableRender() {
		glModule->setVisible(false);
	};

	template<typename TDataType>
	void TransformModel<TDataType>::Transform() {
		auto TriangleSetIn = TypeInfo::cast<TriangleSet<TDataType>>(this->inTopology()->getDataPtr());
		auto EdgeSetIn = TypeInfo::cast<EdgeSet<TDataType>>(this->inTopology()->getDataPtr());
		auto PointSetIn = TypeInfo::cast<PointSet<TDataType>>(this->inTopology()->getDataPtr());
		printf("Cast\n");
		if (TriangleSetIn != nullptr) { inType = Triangle_; }
		else if (EdgeSetIn != nullptr) { inType = Edge_; }
		else if (PointSetIn != nullptr) { inType = Point_; }
		else { inType = Null_; }


		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		auto triangleSet = this->stateTriangleSet()->getDataPtr();
		auto pointSet = this->statePointSet()->getDataPtr();
		auto edgeSet = this->stateEdgeSet()->getDataPtr();


		std::vector<TopologyModule::Edge> edge;
		CArray<TopologyModule::Triangle> c_triangle;
		CArray<Coord> c_point;
		CArray<TopologyModule::Edge> c_edge;
		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangle;

		int numT;
		int numV;
		int numE;
		switch (inType)
		{
		case Point_:
			printf("*********   Point   ********\n");
			c_point.assign(PointSetIn->getPoints());

			break;

		case Edge_:
			printf("*********   Edge   ********\n");
			c_point.assign(EdgeSetIn->getPoints());
			c_edge.assign(EdgeSetIn->getEdges());

			break;

		case Triangle_:
			printf("*********   Triangle   ********\n");
			c_point.assign(TriangleSetIn->getPoints());
			c_edge.assign(TriangleSetIn->getEdges());
			c_triangle.assign(TriangleSetIn->getTriangles());

			break;

		case Null_:
			printf("*********   Null   ********\n");
			break;

		}

		if (!c_point.isEmpty())
		{
			numV = c_point.size();
			for (int i = 0; i < numV; i++)
			{
				vertices.push_back(c_point[i]);
			}

		}

		if (!c_edge.isEmpty())
		{
			numE = c_edge.size();
			for (int i = 0; i < numE; i++)
			{
				edge.push_back(TopologyModule::Edge(c_edge[i]));
			}
		}

		if (!c_triangle.isEmpty())
		{
			numT = c_triangle.size();
			for (int i = 0; i < numT; i++)
			{
				triangle.push_back(TopologyModule::Triangle(c_triangle[i]));
			}
		}




		//±ä»»

		Quat<Real> q = Quat<Real>(M_PI * rot[0] / 180, Coord(1, 0, 0))
			* Quat<Real>(M_PI * rot[1] / 180, Coord(0, 1, 0))
			* Quat<Real>(M_PI * rot[2] / 180, Coord(0, 0, 1));

		q.normalize();

		auto RV = [&](const Coord& v)->Coord {
			return center + q.rotate(v - center);
		};

		int numpt = vertices.size();

		for (int i = 0; i < numpt; i++)
		{
			vertices[i] = RV(vertices[i] * scale + RV(center));
		}

		if (inType == Triangle_)
		{
			triangleSet->setPoints(vertices);
			triangleSet->setTriangles(triangle);
			triangleSet->update();

			edgeSet->setPoints(vertices);
			edgeSet->setEdges(edge);
			pointSet->setPoints(vertices);

			//glModule->setVisible(true);
			//glWireModule->setVisible(true);
			//glPointModule->setVisible(true);
		}
		if (inType == Edge_)
		{
			edgeSet->setPoints(vertices);

			edgeSet->setEdges(edge);
			pointSet->setPoints(vertices);

			//glModule->setVisible(false);
			//glWireModule->setVisible(true);
			//glPointModule->setVisible(true);
			triangleSet->setTriangles(triangle);

		}
		if (inType == Point_)
		{
			pointSet->setPoints(vertices);

			//glModule->setVisible(false);
			//glWireModule->setVisible(false);
			//glPointModule->setVisible(true);
			triangleSet->setTriangles(triangle);
			edgeSet->setEdges(edge);
		}

		vertices.clear();
		triangle.clear();
		c_point.clear();
		c_edge.clear();
		c_triangle.clear();
	
	
	}


	DEFINE_CLASS(TransformModel);
}