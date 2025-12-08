#include "MultibodySystem.h"

#include "Module/TJSoftConstraintSolver.h"
#include "Module/ContactsUnion.h"

#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionBoundingBox.h"
#include "Collision/CollistionDetectionTriangleSet.h"
#include <GLWireframeVisualModule.h>
#include <Mapping/ContactsToEdgeSet.h>

#include <Module/GLPhotorealisticInstanceRender.h>
#include <Mapping/DiscreteElementsToTriangleSet.h>
#include "GLSurfaceVisualModule.h"

#include "Module/SharedFuncsForRigidBody.h"

#include "Module/CarDriver.h"

namespace dyno
{
	IMPLEMENT_TCLASS(MultibodySystem, TDataType)

	template<typename TDataType>
	MultibodySystem<TDataType>::MultibodySystem()
		: RigidBodySystem<TDataType>()
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
		this->animationPipeline()->pushModule(cdBV);

		auto merge = std::make_shared<ContactsUnion<TDataType>>();
		elementQuery->outContacts()->connect(merge->inContactsA());
		cdBV->outContacts()->connect(merge->inContactsB());
		this->animationPipeline()->pushModule(merge);

		auto iterSolver = std::make_shared<TJSoftConstraintSolver<TDataType>>();
		this->stateTimeStep()->connect(iterSolver->inTimeStep());
		this->varFrictionEnabled()->connect(iterSolver->varFrictionEnabled());
		this->varGravityEnabled()->connect(iterSolver->varGravityEnabled());
		this->varGravityValue()->connect(iterSolver->varGravityValue());
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
		this->stateAttribute()->connect(iterSolver->inAttribute());
		this->stateFrictionCoefficients()->connect(iterSolver->inFrictionCoefficients());

		this->stateTopology()->connect(iterSolver->inDiscreteElements());

		merge->outContacts()->connect(iterSolver->inContacts());

		this->animationPipeline()->pushModule(iterSolver);

		this->inTriangleSet()->tagOptional(true);
	}

	template<typename TDataType>
	MultibodySystem<TDataType>::~MultibodySystem()
	{

	}

	template<typename TDataType>
	void MultibodySystem<TDataType>::resetStates()
	{
//		RigidBodySystem<TDataType>::resetStates();

		auto vehicles = this->getVehicles();

		if (vehicles.size() > 0)
		{
			CArray<std::shared_ptr<DiscreteElements<TDataType>>> topos;
			
			uint sizeOfRigidBodies = 0;
			for (uint i = 0; i < vehicles.size(); i++)
			{
				auto vehicle = vehicles[i];

				auto inTopo = vehicle->stateTopology()->getDataPtr();

				topos.pushBack(inTopo);

				sizeOfRigidBodies += vehicle->stateMass()->size();
			}

			auto curTopo = this->stateTopology()->getDataPtr();

			curTopo->merge(topos);

			topos.clear();

			this->stateMass()->resize(sizeOfRigidBodies);
			this->stateCenter()->resize(sizeOfRigidBodies);
			this->stateVelocity()->resize(sizeOfRigidBodies);
			this->stateAngularVelocity()->resize(sizeOfRigidBodies);
			this->stateRotationMatrix()->resize(sizeOfRigidBodies);
			this->stateInertia()->resize(sizeOfRigidBodies);
			this->stateInitialInertia()->resize(sizeOfRigidBodies);
			this->stateQuaternion()->resize(sizeOfRigidBodies);
			this->stateCollisionMask()->resize(sizeOfRigidBodies);
			this->stateAttribute()->resize(sizeOfRigidBodies);
			this->stateFrictionCoefficients()->resize(sizeOfRigidBodies);

			auto& stateMass = this->stateMass()->getData();
			auto& stateCenter = this->stateCenter()->getData();
			auto& stateVelocity = this->stateVelocity()->getData();
			auto& stateAngularVelocity = this->stateAngularVelocity()->getData();
			auto& stateRotationMatrix = this->stateRotationMatrix()->getData();
			auto& stateInertia = this->stateInertia()->getData();
			auto& stateInitialInertia = this->stateInitialInertia()->getData();
			auto& stateQuaternion = this->stateQuaternion()->getData();
			auto& stateCollisionMask = this->stateCollisionMask()->getData();
			auto& stateAttribute = this->stateAttribute()->getData();
			auto& stateFrictionCoefficients = this->stateFrictionCoefficients()->getData();

			uint offset = 0;
			for (uint i = 0; i < vehicles.size(); i++)
			{
				auto vehicle = vehicles[i];

				uint num = vehicle->stateMass()->size();
				if (num > 0) 
				{
					stateMass.assign(vehicle->stateMass()->constData(), vehicle->stateMass()->size(), offset, 0);
					stateCenter.assign(vehicle->stateCenter()->constData(), vehicle->stateCenter()->size(), offset, 0);
					stateVelocity.assign(vehicle->stateVelocity()->constData(), vehicle->stateVelocity()->size(), offset, 0);
					stateAngularVelocity.assign(vehicle->stateAngularVelocity()->constData(), vehicle->stateAngularVelocity()->size(), offset, 0);
					stateRotationMatrix.assign(vehicle->stateRotationMatrix()->constData(), vehicle->stateRotationMatrix()->size(), offset, 0);
					stateInertia.assign(vehicle->stateInertia()->constData(), vehicle->stateInertia()->size(), offset, 0);
					stateInitialInertia.assign(vehicle->stateInitialInertia()->constData(), vehicle->stateInitialInertia()->size(), offset, 0);
					stateQuaternion.assign(vehicle->stateQuaternion()->constData(), vehicle->stateQuaternion()->size(), offset, 0);
					stateCollisionMask.assign(vehicle->stateCollisionMask()->constData(), vehicle->stateCollisionMask()->size(), offset, 0);
					stateAttribute.assign(vehicle->stateAttribute()->constData(), vehicle->stateAttribute()->size(), offset, 0);
					stateFrictionCoefficients.assign(vehicle->stateFrictionCoefficients()->constData(), vehicle->stateFrictionCoefficients()->size(), offset, 0);
					offset += num;
				}

			}

			//TODO: Replace with a GPU-based algorithm?
			CArray<Attribute> hAttributes;
			hAttributes.assign(stateAttribute);

			uint offsetBodyId = 0;
			uint offsetOfRigidBody = 0;
			for (uint i = 0; i < vehicles.size(); i++)
			{
				auto vehicle = vehicles[i];

				uint num = vehicle->stateMass()->size();

				uint maxBodyId = 0;
				for (uint j = 0; j < num; j++)
				{
					Attribute att = hAttributes[offsetOfRigidBody + j];
					att.setObjectId(att.objectId() + offsetBodyId);

					hAttributes[offsetOfRigidBody + j] = att;

					maxBodyId = std::max(maxBodyId, att.objectId());
				}

				offsetOfRigidBody += num;
				offsetBodyId += (maxBodyId + 1);
			}

			stateAttribute.assign(hAttributes);
			hAttributes.clear();
		}
	}

	template<typename TDataType>
	void MultibodySystem<TDataType>::preUpdateStates()
	{
		auto vehicles = this->getVehicles();

		if (vehicles.size() > 0)
		{
			CArray<std::shared_ptr<DiscreteElements<TDataType>>> topos;

			uint sizeOfRigidBodies = 0;
			for (uint i = 0; i < vehicles.size(); i++)
			{
				auto vehicle = vehicles[i];

				auto inTopo = vehicle->stateTopology()->getDataPtr();

				topos.pushBack(inTopo);

				sizeOfRigidBodies += vehicle->stateMass()->size();
			}

			auto curTopo = this->stateTopology()->getDataPtr();

			curTopo->merge(topos);

			topos.clear();
		}
	}

	template<typename TDataType>
	void MultibodySystem<TDataType>::postUpdateStates()
	{
		auto& vehicles = this->getVehicles();

		if (vehicles.size() > 0)
		{
			uint offset = 0;
			for (uint i = 0; i < vehicles.size(); i++)
			{
				auto vehicle = vehicles[i];

				uint sizeOfInput = vehicle->stateMass()->size();

				auto& stateMass = vehicle->stateMass()->getData();
				auto& stateCenter = vehicle->stateCenter()->getData();
				auto& stateVelocity = vehicle->stateVelocity()->getData();
				auto& stateAngularVelocity = vehicle->stateAngularVelocity()->getData();
				auto& stateRotationMatrix = vehicle->stateRotationMatrix()->getData();
				auto& stateInertia = vehicle->stateInertia()->getData();
				auto& stateInitialInertia = vehicle->stateInitialInertia()->getData();
				auto& stateQuaternion = vehicle->stateQuaternion()->getData();
				auto& stateCollisionMask = vehicle->stateCollisionMask()->getData();
				auto& stateAttribute = vehicle->stateAttribute()->getData();
				auto& stateFrictionCoefficients = vehicle->stateFrictionCoefficients()->getData();


				stateMass.assign(this->stateMass()->constData(), sizeOfInput, 0, offset);
				stateCenter.assign(this->stateCenter()->constData(), sizeOfInput, 0, offset);
				stateVelocity.assign(this->stateVelocity()->constData(), sizeOfInput, 0, offset);
				stateAngularVelocity.assign(this->stateAngularVelocity()->constData(), sizeOfInput, 0, offset);
				stateRotationMatrix.assign(this->stateRotationMatrix()->constData(), sizeOfInput, 0, offset);
				stateInertia.assign(this->stateInertia()->constData(), sizeOfInput, 0, offset);
				stateInitialInertia.assign(this->stateInitialInertia()->constData(), sizeOfInput, 0, offset);
				stateQuaternion.assign(this->stateQuaternion()->constData(), sizeOfInput, 0, offset);
				stateCollisionMask.assign(this->stateCollisionMask()->constData(), sizeOfInput, 0, offset);
				stateAttribute.assign(this->stateAttribute()->constData(), sizeOfInput, 0, offset);
				stateFrictionCoefficients.assign(this->stateFrictionCoefficients()->constData(), sizeOfInput, 0, offset);

				auto topo = vehicle->stateTopology()->getDataPtr();

				auto& topoPos = topo->position();
				auto& topoRot = topo->rotation();

				topoPos.assign(this->stateCenter()->constData(), sizeOfInput, 0, offset);
				topoRot.assign(this->stateRotationMatrix()->constData(), sizeOfInput, 0, offset);
				topo->update();

				offset += sizeOfInput;
			}
		}

		RigidBodySystem<TDataType>::postUpdateStates();
	}

	template<typename TDataType>
	bool MultibodySystem<TDataType>::validateInputs()
	{
		auto& vehicles = this->getVehicles();

		return vehicles.size() > 0;
	}

	DEFINE_CLASS(MultibodySystem);
}
