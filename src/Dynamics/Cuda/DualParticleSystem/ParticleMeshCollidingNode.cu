#include "ParticleMeshCollidingNode.h"

namespace dyno
{
	template <typename TDataType>
	ParticleMeshCollidingNode<TDataType>::ParticleMeshCollidingNode()
		:Node()
	{
		auto smoothingLength = std::make_shared<FloatingNumber<TDataType>>();
		smoothingLength->varValue()->setValue(Real(0.012));
		this->animationPipeline()->pushModule(smoothingLength);

		//triangle neighbor
		auto nbrQueryTri = std::make_shared<NeighborTriangleQuery<TDataType>>();
		smoothingLength->outFloating()->connect(nbrQueryTri->inRadius());
		this->inPosition()->connect(nbrQueryTri->inPosition());
 		this->stateTriangleVertex()->connect(nbrQueryTri->inTriPosition());
 		this->stateTriangleIndex()->connect(nbrQueryTri->inTriangles());
		this->inTriangleSet()->connect(nbrQueryTri->inTriangles());
		this->animationPipeline()->pushModule(nbrQueryTri);

		//mesh collision
		auto meshCollision = std::make_shared<TriangularMeshConstraint<TDataType>>();
		this->stateTimeStep()->connect(meshCollision->inTimeStep());
		this->inPosition()->connect(meshCollision->inPosition());
		this->inVelocity()->connect(meshCollision->inVelocity());
 		this->stateTriangleVertex()->connect(meshCollision->inTriangleVertex());
 		this->stateTriangleIndex()->connect(meshCollision->inTriangleIndex());
//		this->inTriangleSet()->connect(meshCollision->inTriangleSet());
		nbrQueryTri->outNeighborIds()->connect(meshCollision->inTriangleNeighborIds());
		this->animationPipeline()->pushModule(meshCollision);

	};

	template <typename TDataType>
	ParticleMeshCollidingNode<TDataType>::~ParticleMeshCollidingNode()
	{}

	template <typename TDataType>
	void ParticleMeshCollidingNode<TDataType>::resetStates()  {
		auto triSet = this->inTriangleSet()->getDataPtr();
		this->stateTriangleVertex()->assign(triSet->getPoints());
		this->stateTriangleIndex()->assign(triSet->getTriangles());
	};

	DEFINE_CLASS(ParticleMeshCollidingNode);
}