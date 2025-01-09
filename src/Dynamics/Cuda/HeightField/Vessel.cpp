#include "Vessel.h"

#include "Quat.h"

#include <GLSurfaceVisualModule.h>
#include "GLPhotorealisticInstanceRender.h"

#include "GltfFunc.h"

namespace dyno
{
	template<typename TDataType>
	Vessel<TDataType>::Vessel()
		: RigidBody<TDataType>()
	{
		this->varDensity()->setRange(1.0f, 10000.0f);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&Vessel<TDataType>::transform, this));
		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		auto EnvelopeRender = std::make_shared<GLSurfaceVisualModule>();
		EnvelopeRender->setColor(Color(0.8f, 0.8f, 0.8f));
		this->stateEnvelope()->promoteOuput()->connect(EnvelopeRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(EnvelopeRender);
		EnvelopeRender->setVisible(false);


		auto texMeshRender = std::make_shared<GLPhotorealisticInstanceRender>();
		this->stateTextureMesh()->connect(texMeshRender->inTextureMesh());
		this->stateInstanceTransform()->connect(texMeshRender->inTransform());
		this->graphicsPipeline()->pushModule(texMeshRender);


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

		auto textureMeshLoader = std::make_shared<FCallBackFunc>(
			[=]() {
				std::string filepath = this->varTextureMeshName()->getValue().string();
				std::shared_ptr<TextureMesh> texMesh = this->stateTextureMesh()->getDataPtr();
				loadGLTFTextureMesh(texMesh, filepath);
			}
		);

		this->varTextureMeshName()->attach(textureMeshLoader);

		this->varDensity()->setValue(150.0f);
		this->varBarycenterOffset()->setValue(Vec3f(0.0f, 0.0f, -0.5f));
	}

	template<typename TDataType>
	Vessel<TDataType>::~Vessel()
	{

	}

	template<typename TDataType>
	NBoundingBox Vessel<TDataType>::boundingBox()
	{
		NBoundingBox box;

		mInitialEnvelope.requestBoundingBox(box.lower, box.upper);

		return box;
	}

	template<typename TDataType>
	void Vessel<TDataType>::resetStates()
	{
		if (this->stateEnvelope()->isEmpty()) this->stateEnvelope()->allocate();
		if (this->stateTextureMesh()->isEmpty()) this->stateTextureMesh()->allocate();

		std::string envFileName = getAssetPath() + "obj/boat_boundary.obj";
		if (this->varEnvelopeName()->getValue() != envFileName) {
			this->varEnvelopeName()->setValue(FilePath(envFileName));
		}

		std::string texMeshName = getAssetPath() + "gltf/SailBoat/SailBoat.gltf";
		if (this->varTextureMeshName()->getValue() != texMeshName) {
			this->varTextureMeshName()->setValue(FilePath(texMeshName));
		}

		this->transform();

		auto texMesh = this->stateTextureMesh()->constDataPtr();

		//Initialize states for the rigid body
		{
			Coord lo;
			Coord hi;

			if (mInitialEnvelope.isEmpty())
				return;

			mInitialEnvelope.requestBoundingBox(lo, hi);

			Coord scale = this->varScale()->getValue();

			mShapeCenter = 0.5f * (hi + lo);


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


			Coord location = this->varLocation()->getValue();
			Coord rot = this->varRotation()->getValue();

			auto quat = this->computeQuaternion();
			auto offset = this->varBarycenterOffset()->getValue();

			this->stateMass()->setValue(mass);
			this->stateCenter()->setValue(location + mShapeCenter);
			this->stateBarycenter()->setValue(location + mShapeCenter + quat.rotate(offset));
			this->stateVelocity()->setValue(Vec3f(0));
			this->stateAngularVelocity()->setValue(Vec3f(0));
			this->stateInertia()->setValue(inertia);
			this->stateQuaternion()->setValue(quat);
			this->stateInitialInertia()->setValue(inertia);
		}

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

		auto texMesh = this->stateTextureMesh()->getDataPtr();
		{

			uint N = texMesh->shapes().size();

			CArrayList<Transform3f> tms;
			tms.assign(this->stateInstanceTransform()->constData());

			for (uint i = 0; i < tms.size(); i++)
			{
				auto& list = tms[i];
				for (uint j = 0; j < list.size(); j++)
				{
					list[j].translation() = center + quat.rotate(texMesh->shapes()[i]->boundingTransform.translation() * scale) - mShapeCenter; //
					list[j].rotation() = quat.toMatrix3x3();
					list[j].scale() = scale;
				}

			}

			auto instantanceTransform = this->stateInstanceTransform()->getDataPtr();
			instantanceTransform->assign(tms);

			tms.clear();

		}
	}

	template<typename TDataType>
	void Vessel<TDataType>::transform()
	{

		Coord location = this->varLocation()->getValue();
		Coord rot = this->varRotation()->getValue();
		Coord scale = this->varScale()->getValue();

		auto quat = this->computeQuaternion();

		auto envelope = this->stateEnvelope()->getDataPtr();
		envelope->copyFrom(mInitialEnvelope);
		envelope->scale(scale);
		envelope->rotate(quat);
		envelope->translate(location);

		if (this->stateTextureMesh()->isEmpty())
			return;

		auto texMesh = this->stateTextureMesh()->constDataPtr();
		{
			uint N = texMesh->shapes().size();

			CArrayList<Transform3f> tms;
			tms.resize(N, 1);

			for (uint i = 0; i < N; i++)
			{
				Transform3f t = texMesh->shapes()[i]->boundingTransform;

				tms[i].insert(Transform3f(t.translation() * scale + location, quat.toMatrix3x3(), t.scale() * scale));
			}

			if (this->stateInstanceTransform()->isEmpty())
			{
				this->stateInstanceTransform()->allocate();
			}

			auto instantanceTransform = this->stateInstanceTransform()->getDataPtr();
			instantanceTransform->assign(tms);

			tms.clear();
		}
	}

	DEFINE_CLASS(Vessel);
}