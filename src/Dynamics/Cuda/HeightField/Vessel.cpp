#include "Vessel.h"

#include "Quat.h"

#include <GLSurfaceVisualModule.h>

namespace dyno
{
	template<typename TDataType>
	Vessel<TDataType>::Vessel()
		: RigidBody<TDataType>()
	{
		this->stateEnvelope()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->stateMesh()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		this->varDensity()->setRange(1.0f, 10000.0f);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&Vessel<TDataType>::transform, this));
		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		auto rigidMeshRender = std::make_shared<GLSurfaceVisualModule>();
		rigidMeshRender->setColor(Color(0.8f, 0.8f, 0.8f));
		this->stateMesh()->promoteOuput()->connect(rigidMeshRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(rigidMeshRender);

		this->allowExported(true);

		auto meshLoader = std::make_shared<FCallBackFunc>(
			[=]() {
				std::string name = this->varMeshName()->getValue().string();
				bool succeed = mInitialMesh.loadObjFile(name);

				if (succeed)
				{
					auto curTopo = this->stateMesh()->getDataPtr();

					curTopo->copyFrom(mInitialMesh);

					curTopo->scale(this->varScale()->getValue());
					curTopo->rotate(this->varRotation()->getValue() * M_PI / 180);
					curTopo->translate(this->varLocation()->getValue());
				}
			}
		);
		meshLoader->update();

		this->varMeshName()->attach(meshLoader);

		auto evenlopeLoader = std::make_shared<FCallBackFunc>(
			[=]() {
				std::string name = this->varEnvelopeName()->getValue().string();
				bool succeed = mInitialEnvelope.loadObjFile(name);

				if (succeed)
				{
					auto envelope = this->stateEnvelope()->getDataPtr();

					envelope->copyFrom(mInitialEnvelope);

					envelope->scale(this->varScale()->getValue());
					envelope->rotate(this->varRotation()->getValue() * M_PI / 180);
					envelope->translate(this->varLocation()->getValue());
				}
			}
		);
		evenlopeLoader->update();

		this->varEnvelopeName()->attach(evenlopeLoader);
	}

	template<typename TDataType>
	Vessel<TDataType>::~Vessel()
	{
		
	}

	template<typename TDataType>
	NBoundingBox Vessel<TDataType>::boundingBox()
	{
		NBoundingBox box;

		this->stateMesh()->constDataPtr()->requestBoundingBox(box.lower, box.upper);

		return box;
	}

	template<typename TDataType>
	void Vessel<TDataType>::resetStates()
	{
		Coord location = this->varLocation()->getValue();
		Coord rot = this->varRotation()->getValue();
		Coord scale = this->varScale()->getValue();

		auto quat = this->computeQuaternion();
		auto offset = this->varBarycenterOffset()->getValue();

		//Initialize states for the rigid body
		{
			Coord lo;
			Coord hi;

			mInitialMesh.requestBoundingBox(lo, hi);

			mShapeCenter = 0.5f * (hi + lo);

			auto envelope = this->stateEnvelope()->getDataPtr();
			envelope->copyFrom(mInitialEnvelope);
			envelope->scale(scale);
			envelope->rotate(quat);
			envelope->translate(location);

			Real lx = hi.x - lo.x;
			Real ly = hi.y - lo.y;
			Real lz = hi.z - lo.z;

			Real rho = this->varDensity()->getData();
			Real mass = rho * lx * ly * lz;

			//Calculate mass using the bounding box
			Matrix inertia = 1.0f / 12.0f * mass
				* Mat3f(ly * ly + lz * lz, 0, 0,
					0, lx * lx + lz * lz, 0,
					0, 0, lx * lx + ly * ly);

			this->stateMass()->setValue(mass);
			this->stateCenter()->setValue(location + mShapeCenter);
			this->stateBarycenter()->setValue(location + mShapeCenter + quat.rotate(offset));
			this->stateVelocity()->setValue(Vec3f(0));
			this->stateAngularVelocity()->setValue(Vec3f(0));
			this->stateInertia()->setValue(inertia);
			this->stateQuaternion()->setValue(quat);
			this->stateInitialInertia()->setValue(inertia);
		}

		auto mesh = this->stateMesh()->getDataPtr();
		mesh->copyFrom(mInitialMesh);
		mesh->scale(scale);
		mesh->rotate(quat);
		mesh->translate(location);

		RigidBody<TDataType>::resetStates();
	}

	template<typename TDataType>
	void Vessel<TDataType>::updateStates()
	{
		RigidBody<TDataType>::updateStates();

		auto center = this->stateCenter()->getValue();
		auto quat = this->stateQuaternion()->getValue();
		auto scale = this->varScale()->getValue();

		auto offset = this->varBarycenterOffset()->getValue();

		this->stateBarycenter()->setValue(center + quat.rotate(offset));
 
		auto buoy = this->stateEnvelope()->getDataPtr();
		buoy->copyFrom(mInitialEnvelope);
		buoy->rotate(quat);
		buoy->scale(scale);
		buoy->translate(center - mShapeCenter);

		auto mesh = this->stateMesh()->getDataPtr();
		mesh->copyFrom(mInitialMesh);
		mesh->rotate(quat);
		mesh->scale(scale);
		mesh->translate(center - mShapeCenter);
	}

	template<typename TDataType>
	void Vessel<TDataType>::transform()
	{
		Coord location = this->varLocation()->getValue();
		Coord rot = this->varRotation()->getValue();
		Coord scale = this->varScale()->getValue();

		auto quat = this->computeQuaternion();

		Coord lo;
		Coord hi;

		mInitialMesh.requestBoundingBox(lo, hi);

		Coord center = 0.5f * (hi + lo);

		auto envelope = this->stateEnvelope()->getDataPtr();
		envelope->copyFrom(mInitialEnvelope);
		envelope->scale(scale);
		envelope->rotate(quat);
		envelope->translate(location);

		auto mesh = this->stateMesh()->getDataPtr();
		mesh->copyFrom(mInitialMesh);
		mesh->scale(scale);
		mesh->rotate(quat);
		mesh->translate(location);
	}

	DEFINE_CLASS(Vessel);
}