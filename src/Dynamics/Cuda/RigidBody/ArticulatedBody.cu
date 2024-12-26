#include "ArticulatedBody.h"

#include <GLSurfaceVisualModule.h>

//Collision
#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionTriangleSet.h"

//RigidBody
#include "Module/ContactsUnion.h"
#include "Module/TJConstraintSolver.h"
#include "Module/InstanceTransform.h"
#include "Module/SharedFuncsForRigidBody.h"

//Rendering
#include "Module/GLPhotorealisticInstanceRender.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ArticulatedBody, TDataType)

	template<typename TDataType>
	ArticulatedBody<TDataType>::ArticulatedBody()
		: ParametricModel<TDataType>()
		, RigidBodySystem<TDataType>()
	{
		this->animationPipeline()->clear();

		auto elementQuery = std::make_shared<NeighborElementQuery<TDataType>>();
		elementQuery->varSelfCollision()->setValue(false);
		this->stateTopology()->connect(elementQuery->inDiscreteElements());
		this->stateCollisionMask()->connect(elementQuery->inCollisionMask());
		this->stateAttribute()->connect(elementQuery->inAttribute());
		this->animationPipeline()->pushModule(elementQuery);

		auto cdBV = std::make_shared<CollistionDetectionTriangleSet<TDataType>>();
		this->stateTopology()->connect(cdBV->inDiscreteElements());
		this->inTriangleSet()->connect(cdBV->inTriangleSet());
		// 		auto cdBV = std::make_shared<CollistionDetectionBoundingBox<TDataType>>();
		// 		this->stateTopology()->connect(cdBV->inDiscreteElements());
		this->animationPipeline()->pushModule(cdBV);


		auto merge = std::make_shared<ContactsUnion<TDataType>>();
		elementQuery->outContacts()->connect(merge->inContactsA());
		cdBV->outContacts()->connect(merge->inContactsB());
		this->animationPipeline()->pushModule(merge);

		auto iterSolver = std::make_shared<TJConstraintSolver<TDataType>>();
		this->stateTimeStep()->connect(iterSolver->inTimeStep());
		this->varFrictionEnabled()->connect(iterSolver->varFrictionEnabled());
		this->varGravityEnabled()->connect(iterSolver->varGravityEnabled());
		this->varGravityValue()->connect(iterSolver->varGravityValue());
		//this->varFrictionCoefficient()->connect(iterSolver->varFrictionCoefficient());
		this->varFrictionCoefficient()->setValue(20.0f);
		this->varSlop()->connect(iterSolver->varSlop());
		this->stateMass()->connect(iterSolver->inMass());
		this->stateCenter()->connect(iterSolver->inCenter());
		this->stateVelocity()->connect(iterSolver->inVelocity());
		this->stateAngularVelocity()->connect(iterSolver->inAngularVelocity());
		this->stateRotationMatrix()->connect(iterSolver->inRotationMatrix());
		this->stateInertia()->connect(iterSolver->inInertia());
		this->stateQuaternion()->connect(iterSolver->inQuaternion());
		this->stateInitialInertia()->connect(iterSolver->inInitialInertia());

		this->stateTopology()->connect(iterSolver->inDiscreteElements());

		merge->outContacts()->connect(iterSolver->inContacts());

		this->animationPipeline()->pushModule(iterSolver);

		/*auto driver = std::make_shared<SimpleVechicleDriver>();

		this->stateFrameNumber()->connect(driver->inFrameNumber());
		this->stateInstanceTransform()->connect(driver->inInstanceTransform());

		this->animationPipeline()->pushModule(driver);*/

		this->inTriangleSet()->tagOptional(true);

		auto transformer = std::make_shared<InstanceTransform<DataType3f>>();
		this->stateCenter()->connect(transformer->inCenter());
		this->stateInitialRotation()->connect(transformer->inInitialRotation());
		this->stateRotationMatrix()->connect(transformer->inRotationMatrix());
		this->stateBindingPair()->connect(transformer->inBindingPair());
		this->stateBindingTag()->connect(transformer->inBindingTag());
		this->stateInstanceTransform()->connect(transformer->inInstanceTransform());
		this->graphicsPipeline()->pushModule(transformer);

		auto prRender = std::make_shared<GLPhotorealisticInstanceRender>();
		this->inTextureMesh()->connect(prRender->inTextureMesh());
		transformer->outInstanceTransform()->connect(prRender->inTransform());
		this->graphicsPipeline()->pushModule(prRender);

		this->setForceUpdate(true);
	}

	template<typename TDataType>
	ArticulatedBody<TDataType>::~ArticulatedBody()
	{

	}


	template<typename TDataType>
	void ArticulatedBody<TDataType>::resetStates()
	{
		RigidBodySystem<TDataType>::resetStates();

		auto topo = this->stateTopology()->constDataPtr();

		int sizeOfRigids = this->stateCenter()->size();

		std::shared_ptr<TextureMesh> texMesh = getTexMeshPtr();

		uint N = 0;
		if (texMesh != NULL);
			N = texMesh->shapes().size();

		CArrayList<Transform3f> tms;
		CArray<uint> instanceNum(N);
		instanceNum.reset();

		//Calculate instance number
		for (uint i = 0; i < mBindingPair.size(); i++)
		{
			instanceNum[mBindingPair[i].first]++;
		}

		if (instanceNum.size() > 0)
			tms.resize(instanceNum);

		//Initialize CArrayList
		for (uint i = 0; i < N; i++)
		{
			for (uint j = 0; j < instanceNum[i]; j++)
			{
				tms[i].insert(Transform3f());
			}
		}

		this->stateInstanceTransform()->assign(tms);

		auto deTopo = this->stateTopology()->constDataPtr();
		auto offset = deTopo->calculateElementOffset();

		std::vector<Pair<uint, uint>> bindingPair(sizeOfRigids);
		std::vector<int> tags(sizeOfRigids, 0);

		for (int i = 0; i < mBindingPair.size(); i++)
		{
			auto actor = mActors[i];
			int idx = actor->idx + offset.checkElementOffset(actor->shapeType);

			bindingPair[idx] = mBindingPair[i];
			tags[idx] = 1;
		}

		this->stateBindingPair()->assign(bindingPair);
		this->stateBindingTag()->assign(tags);
		this->stateInitialRotation()->assign(this->stateRotationMatrix()->constData());

		this->updateInstanceTransform();

		tms.clear();
		bindingPair.clear();
		tags.clear();

		this->transform();
	}


	template<typename TDataType>
	void ArticulatedBody<TDataType>::transform()
	{
		////************************** initial mInitialRot *************************//

		mInitialRot.assign(this->stateRotationMatrix()->constData());

		CArray<Coord> hostCenter;
		hostCenter.assign(this->stateCenter()->constData());

		CArray<Quat<Real>> hostQuaternion;
		hostQuaternion.assign(this->stateQuaternion()->constData());

		CArray<Mat3f> hostRotation;
		hostRotation.assign(this->stateRotationMatrix()->constData());

		{
			//get Elements
			auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->stateTopology()->getDataPtr());

			auto& boxes = topo->boxesInGlobal();
			auto& spheres = topo->spheresInGlobal();
			auto& tets = topo->tetsInGlobal();
			auto& caps = topo->capsulesInGlobal();

			std::vector<Transform3f> vehicleTransform = this->varVehiclesTransform()->getValue();

			int vehicleNum = vehicleTransform.size();



			for (size_t i = 0; i < vehicleNum; i++)
			{

				Quat<Real> q = Quat<Real>(vehicleTransform[i].rotation());

				Vec3f pos = vehicleTransform[i].translation();


				int boxesNum = boxes.size() / vehicleNum;
				int spheresNum = spheres.size() / vehicleNum;
				int tetsNum = tets.size() / vehicleNum;
				int capsNum = caps.size() / vehicleNum;

				//***************************** Copy Translation *************************//
				for (uint j = 0; j < spheresNum; j++)
				{

					hostCenter[i * spheresNum + j] = q.rotate(hostCenter[i * spheresNum + j]) + pos;
				}

				for (uint j = 0; j < boxesNum; j++)
				{
					int offset = spheres.size();
					hostCenter[i * boxesNum + j + offset] = q.rotate(hostCenter[i * boxesNum + j + offset]) + pos;
				}

				for (uint j = 0; j < tetsNum; j++)
				{
					int offset = boxes.size() + spheres.size();
					hostCenter[i * tetsNum + j + offset] = q.rotate(hostCenter[i * tetsNum + j + offset]) + pos;
				}

				for (uint j = 0; j < capsNum; j++)
				{
					int offset = boxes.size() + spheres.size() + tets.size();
					hostCenter[i * capsNum + j + offset] = q.rotate(hostCenter[i * capsNum + j + offset]) + pos;
				}

				//***************************** Copy Rotation *************************//

				for (uint j = 0; j < spheresNum; j++)
				{

					hostQuaternion[i * spheresNum + j] = q * hostQuaternion[i * spheresNum + j];
				}
				for (uint j = 0; j < boxesNum; j++)
				{
					int offset = spheres.size();
					hostQuaternion[i * boxesNum + j + offset] = q * hostQuaternion[i * boxesNum + j + offset];
				}
				for (uint j = 0; j < tetsNum; j++)
				{
					int offset = boxes.size() + spheres.size();
					hostQuaternion[i * tetsNum + j + offset] = q * hostQuaternion[i * tetsNum + j + offset];
				}
				for (uint j = 0; j < capsNum; j++)
				{
					int offset = boxes.size() + spheres.size() + tets.size();
					hostQuaternion[i * capsNum + j + offset] = q * hostQuaternion[i * capsNum + j + offset];
				}


				for (uint j = 0; j < spheresNum; j++)
				{
					hostRotation[i * spheresNum + j] = q.toMatrix3x3() * hostRotation[i * spheresNum + j];
				}
				for (uint j = 0; j < boxesNum; j++)
				{
					int offset = spheres.size();
					hostRotation[i * boxesNum + j + offset] = q.toMatrix3x3() * hostRotation[i * boxesNum + j + offset];
				}
				for (uint j = 0; j < tetsNum; j++)
				{
					int offset = boxes.size() + spheres.size();
					hostRotation[i * tetsNum + j + offset] = q.toMatrix3x3() * hostRotation[i * tetsNum + j + offset];
				}
				for (uint j = 0; j < capsNum; j++)
				{
					int offset = boxes.size() + spheres.size() + tets.size();
					hostRotation[i * capsNum + j + offset] = q.toMatrix3x3() * hostRotation[i * capsNum + j + offset];
				}


			}

		}


		{
			//get varTransform;
			auto quat = this->computeQuaternion();
			Coord location = this->varLocation()->getValue();

			//***************************** Translation *************************//


			for (uint i = 0; i < hostCenter.size(); i++)
			{
				hostCenter[i] = quat.rotate(hostCenter[i]) + location;
			}

			//***************************** Rotation *************************//

			for (uint i = 0; i < hostQuaternion.size(); i++)
			{
				hostQuaternion[i] = quat * hostQuaternion[i];
			}

			for (uint i = 0; i < hostRotation.size(); i++)
			{
				hostRotation[i] = quat.toMatrix3x3() * hostRotation[i];
			}



		}

		this->stateCenter()->assign(hostCenter);
		this->stateQuaternion()->assign(hostQuaternion);
		this->stateRotationMatrix()->assign(hostRotation);

		hostCenter.clear();
		hostQuaternion.clear();
		hostRotation.clear();
	}


	template<typename TDataType>
	void ArticulatedBody<TDataType>::updateStates()
	{
		RigidBodySystem<TDataType>::updateStates();
	}

	template<typename TDataType>
	void ArticulatedBody<TDataType>::updateInstanceTransform()
	{
		ApplyTransform(
			this->stateInstanceTransform()->getData(),
			this->stateCenter()->getData(),
			this->stateRotationMatrix()->getData(),
			this->stateInitialRotation()->constData(),
			this->stateBindingPair()->constData(),
			this->stateBindingTag()->constData());
	}

	template<typename TDataType>
	void ArticulatedBody<TDataType>::bind(std::shared_ptr<PdActor> actor, Pair<uint, uint> shapeId)
	{
		mActors.push_back(actor);
		mBindingPair.push_back(shapeId);
	}

	template<typename TDataType>
	void ArticulatedBody<TDataType>::clearVechicle()
	{
		mBindingPair.clear();
		mActors.clear();
	}

	DEFINE_CLASS(ArticulatedBody);
}