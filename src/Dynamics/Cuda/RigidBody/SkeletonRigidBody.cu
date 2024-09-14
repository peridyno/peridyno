#include "SkeletonRigidBody.h"

#include "Module/SimpleVechicleDriver.h"
#include "Module/SharedFuncsForRigidBody.h"
#include "Module/ContactsUnion.h"
#include "Module/TJConstraintSolver.h"
#include "Module/TJSoftConstraintSolver.h"
#include "Module/PJSNJSConstraintSolver.h"
#include "Module/PJSoftConstraintSolver.h"
#include "Module/PJSConstraintSolver.h"
#include "Module/PCGConstraintSolver.h"
#include "Module/CarDriver.h"

#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionBoundingBox.h"
#include "Collision/CollistionDetectionTriangleSet.h"
#include <GLWireframeVisualModule.h>
#include <Mapping/ContactsToEdgeSet.h>

#include <Module/GLPhotorealisticInstanceRender.h>
#include <Mapping/DiscreteElementsToTriangleSet.h>
#include "GLSurfaceVisualModule.h"
#include <cmath>

namespace dyno
{
	IMPLEMENT_TCLASS(SkeletonRigidBody, TDataType)

	template<typename TDataType>
	SkeletonRigidBody<TDataType>::SkeletonRigidBody()
		: RigidBodySystem<TDataType>()
	{
		this->animationPipeline()->clear();

		auto elementQuery = std::make_shared<NeighborElementQuery<TDataType>>();
		elementQuery->varSelfCollision()->setValue(false);
		this->stateTopology()->connect(elementQuery->inDiscreteElements());
		this->stateCollisionMask()->connect(elementQuery->inCollisionMask());
		this->stateAttribute()->connect(elementQuery->inAttribute());
		this->animationPipeline()->pushModule(elementQuery);

		auto convertElement = std::make_shared<DiscreteElementsToTriangleSet<TDataType>>();
		this->stateTopology()->connect(convertElement->inDiscreteElements());

		auto surfaceRender = std::make_shared<GLSurfaceVisualModule>();
		convertElement->outTriangleSet()->connect(surfaceRender->inTriangleSet());

		this->graphicsPipeline()->pushModule(convertElement);
		this->graphicsPipeline()->pushModule(surfaceRender);

	}

	template<typename TDataType>
	SkeletonRigidBody<TDataType>::~SkeletonRigidBody()
	{

	}

	template<typename TDataType>
	void SkeletonRigidBody<TDataType>::resetStates()
	{


		this->clearRigidBodySystem();

		auto topo = this->stateTopology()->constDataPtr();
		auto d_center = this->inElementsCenter()->constDataPtr();
		auto d_rotation = this->inElementsRotation()->constDataPtr();
		auto d_quat = this->inElementsQuaternion()->constDataPtr();

		
		CArray<Vec3f> c_center;
		c_center.assign(*d_center);
		CArray<Quat1f> c_quat;
		c_quat.assign(*d_quat);

		auto d_length = this->inElementsLength()->constDataPtr();

		CArray<Real> c_length;
		c_length.assign(*d_length);

		auto d_radius = this->inElementsRadius()->constDataPtr();

		CArray<Real> c_radius;
		c_radius.assign(*d_radius);



		std::map<int, CapsuleInfo> capsules;
		std::map<int, std::shared_ptr<PdActor>> Actors;




		RigidBodyInfo rigidbody;
		rigidbody.bodyId = 0;

		for (size_t i = 0; i < c_center.size(); i++)
		{
			capsules[i].center = c_center[i];
			capsules[i].rot = c_quat[i];
			capsules[i].halfLength = c_length[i]/3;
			capsules[i].radius = log(c_radius[i]+1)/2;
			Actors[i] = this->addCapsule(capsules[i], rigidbody, 100);
		}

		RigidBodySystem<TDataType>::resetStates();

	}


	template<typename TDataType>
	void SkeletonRigidBody<TDataType>::transform()
	{		
		

	}


	template<typename TDataType>
	void SkeletonRigidBody<TDataType>::updateStates()
	{
		
		this->stateCenter()->getDataPtr()->assign(this->inElementsCenter()->getData());
		this->stateQuaternion()->getDataPtr()->assign(this->inElementsQuaternion()->getData());
		this->stateRotationMatrix()->getDataPtr()->assign(this->inElementsRotation()->getData());




		updateTopology();



	}



	
	template<typename TDataType>
	void SkeletonRigidBody<TDataType>::clearVechicle()
	{
		mBindingPair.clear();
		mBindingPairDevice.clear();
		mBindingTagDevice.clear();
		mInitialRot.clear();
		mActors.clear();
	}

	DEFINE_CLASS(SkeletonRigidBody);

	
}
