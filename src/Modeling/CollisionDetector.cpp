#include "CollisionDetector.h"

#include "Primitive/Primitive3D.h"

#include "Collision/CollisionDetectionAlgorithm.h"

#include "BasicShapes/PlaneModel.h"
#include "BasicShapes/CubeModel.h"
#include "BasicShapes/SphereModel.h"
#include "BasicShapes/CapsuleModel.h"

#include "GLPointVisualModule.h"
#include "GLWireframeVisualModule.h"

namespace dyno
{
	IMPLEMENT_TCLASS(CollisionDetector, TDataType)

	template<typename TDataType>
	CollisionDetector<TDataType>::CollisionDetector() {
		this->setAutoSync(true);

		auto pointRender = std::make_shared<GLPointVisualModule>();
		pointRender->varPointSize()->setValue(0.02);
		pointRender->varBaseColor()->setValue(Color(1.0f, 0.0f, 0.0f));
		this->stateContacts()->connect(pointRender->inPointSet());
		this->graphicsPipeline()->pushModule(pointRender);

		auto wireRender = std::make_shared<GLWireframeVisualModule>();
		wireRender->varRenderMode()->setCurrentKey(GLWireframeVisualModule::CYLINDER);
		this->stateNormals()->connect(wireRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireRender);
	};

	template<typename TDataType>
	void CollisionDetector<TDataType>::resetStates()
	{
		if (this->stateContacts()->isEmpty()) {
			this->stateContacts()->allocate();
		}

		if (this->stateNormals()->isEmpty()) {
			this->stateNormals()->allocate();
		}

		auto shapeA = this->getShapeA();
		auto shapeB = this->getShapeB();

		TManifold<Real> manifold;

		if (shapeA->getShapeType() == BasicShapeType::CUBE && shapeB->getShapeType() == BasicShapeType::CUBE)		//cube-cube
		{
			auto modelA = dynamic_cast<CubeModel<TDataType>*>(shapeA);
			auto modelB = dynamic_cast<CubeModel<TDataType>*>(shapeB);

			auto sA = modelA->outCube()->getValue();
			auto sB = modelB->outCube()->getValue();

			//CollisionDetection<Real>::request(manifold, sA, sB);
			CollisionDetection<Real>::request(manifold, sA, sB, 0.01f, 0.01f);
		}
		else if (shapeA->getShapeType() == BasicShapeType::SPHERE && shapeB->getShapeType() == BasicShapeType::SPHERE)		//sphere-sphere
		{
			auto modelA = dynamic_cast<SphereModel<TDataType>*>(shapeA);
			auto modelB = dynamic_cast<SphereModel<TDataType>*>(shapeB);

			auto sA = modelA->outSphere()->getValue();
			auto sB = modelB->outSphere()->getValue();

			//CollisionDetection<Real>::request(manifold, sA, sB);
			CollisionDetection<Real>::request(manifold, sA, sB, 0.0f, 0.0f);
		}
		else if (shapeA->getShapeType() == BasicShapeType::CAPSULE && shapeB->getShapeType() == BasicShapeType::CAPSULE)		//capsule-capsule
		{
			auto modelA = dynamic_cast<CapsuleModel<TDataType>*>(shapeA);
			auto modelB = dynamic_cast<CapsuleModel<TDataType>*>(shapeB);

			auto sA = modelA->outCapsule()->getValue();
			auto sB = modelB->outCapsule()->getValue();
			
			Segment3D segA = sA.centerline();
			Segment3D segB = sB.centerline();


			//CollisionDetection<Real>::request(manifold, sA, sB);
			CollisionDetection<Real>::request(manifold, segA, segB, sA.radius, sB.radius);
		}
		else if (shapeA->getShapeType() == BasicShapeType::CUBE && shapeB->getShapeType() == BasicShapeType::CAPSULE)	//cube-capsule
		{
			auto modelA = dynamic_cast<CubeModel<TDataType>*>(shapeA);
			auto modelB = dynamic_cast<CapsuleModel<TDataType>*>(shapeB);

			auto sA = modelA->outCube()->getValue();
			auto sB = modelB->outCapsule()->getValue();
			Segment3D seg = sB.centerline();
			Real radius2 = sB.radius;
			//CollisionDetection<Real>::request(manifold, sA, sB);
			CollisionDetection<Real>::request(manifold, sA, seg, 0.01f, radius2);
		}
		else if (shapeA->getShapeType() == BasicShapeType::CAPSULE && shapeB->getShapeType() == BasicShapeType::CUBE)		//capsule-cube
		{
			auto modelA = dynamic_cast<CapsuleModel<TDataType>*>(shapeA);
			auto modelB = dynamic_cast<CubeModel<TDataType>*>(shapeB);

			auto sA = modelA->outCapsule()->getValue();
			auto sB = modelB->outCube()->getValue();
			Segment3D seg = sA.centerline();
			Real radius1 = sA.radius;
			//CollisionDetection<Real>::request(manifold, sA, sB);
			CollisionDetection<Real>::request(manifold, seg, sB, radius1, 0.01f);
		}
		else if (shapeA->getShapeType() == BasicShapeType::CUBE && shapeB->getShapeType() == BasicShapeType::SPHERE)		//cube-sphere
		{
			auto modelA = dynamic_cast<CubeModel<TDataType>*>(shapeA);
			auto modelB = dynamic_cast<SphereModel<TDataType>*>(shapeB);

			auto sA = modelA->outCube()->getValue();
			auto sB = modelB->outSphere()->getValue();

			//CollisionDetection<Real>::request(manifold, sA, sB);
			CollisionDetection<Real>::request(manifold, sA, sB, 0.01f, 0.f);
		}
		else if (shapeA->getShapeType() == BasicShapeType::SPHERE && shapeB->getShapeType() == BasicShapeType::CUBE)		//sphere-cube
		{
			auto modelA = dynamic_cast<SphereModel<TDataType>*>(shapeA);
			auto modelB = dynamic_cast<CubeModel<TDataType>*>(shapeB);

			auto sA = modelA->outSphere()->getValue();
			auto sB = modelB->outCube()->getValue();

			//CollisionDetection<Real>::request(manifold, sA, sB);
			CollisionDetection<Real>::request(manifold, sA, sB, 0.f, 0.01f);
		}
		else if (shapeA->getShapeType() == BasicShapeType::SPHERE && shapeB->getShapeType() == BasicShapeType::CAPSULE)		//sphere-cube
		{
			auto modelA = dynamic_cast<SphereModel<TDataType>*>(shapeA);
			auto modelB = dynamic_cast<CapsuleModel<TDataType>*>(shapeB);

			auto sA = modelA->outSphere()->getValue();
			auto sB = modelB->outCapsule()->getValue();

			Segment3D seg = sB.centerline();
			Real radius2 = sB.radius;

			//CollisionDetection<Real>::request(manifold, sA, sB);
			CollisionDetection<Real>::request(manifold, sA, seg, 0.f, radius2);
		}
		else if (shapeA->getShapeType() == BasicShapeType::CAPSULE && shapeB->getShapeType() == BasicShapeType::SPHERE)		//sphere-cube
		{
			auto modelA = dynamic_cast<CapsuleModel<TDataType>*>(shapeA);
			auto modelB = dynamic_cast<SphereModel<TDataType>*>(shapeB);

			auto sA = modelA->outCapsule()->getValue();
			auto sB = modelB->outSphere()->getValue();
			Segment3D seg = sA.centerline();
			Real radius1 = sA.radius;

			//CollisionDetection<Real>::request(manifold, sA, sB);
			CollisionDetection<Real>::request(manifold, seg, sB, radius1, 0.f);
		}
		else
			std::cout << "Not supported yet" << std::endl;


		std::vector<Coord> points;

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Edge> edges;


		uint num = manifold.contactCount;
		for (uint i = 0; i < num; i++)
		{
			points.push_back(manifold.contacts[i].position);
			vertices.push_back(manifold.contacts[i].position);
			vertices.push_back(manifold.contacts[i].position + manifold.normal * 0.05);
			edges.push_back(TopologyModule::Edge(2 * i, 2 * i + 1));
		}

		auto ptSet = this->stateContacts()->getDataPtr();
		ptSet->setPoints(points);
		ptSet->update();

		auto edgeSet = this->stateNormals()->getDataPtr();
		edgeSet->setPoints(vertices);
		edgeSet->setEdges(edges);
		edgeSet->update();

		points.clear();
		vertices.clear();
		edges.clear();
	}

	template<typename TDataType>
	bool CollisionDetector<TDataType>::validateInputs()
	{
		return this->getShapeA() != nullptr && this->getShapeB() != nullptr;
	}

	DEFINE_CLASS(CollisionDetector);
}