#include "Gear.h"

#include "Mapping/DiscreteElementsToTriangleSet.h"
#include "GLSurfaceVisualModule.h"

namespace dyno
{
	//Gear
	IMPLEMENT_TCLASS(Gear, TDataType)

	template<typename TDataType>
	Gear<TDataType>::Gear() :
		ArticulatedBody<TDataType>()
	{
		auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
		this->stateTopology()->connect(mapper->inDiscreteElements());
		this->graphicsPipeline()->pushModule(mapper);

		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		sRender->setColor(Color(1, 1, 0));
		sRender->setAlpha(0.5f);
		mapper->outTriangleSet()->connect(sRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(sRender);
	}

	template<typename TDataType>
	Gear<TDataType>::~Gear()
	{

	}

	template<typename TDataType>
	void Gear<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();

		std::string filename = getAssetPath() + "gear/gear_up.obj";
		if (this->varFilePath()->getValue() != filename)
		{
			this->varFilePath()->setValue(FilePath(filename));
		}

		//first gear
		RigidBodyInfo info;
		info.position = Vec3f(0.445f, 1.204f, -0.151);
		info.angularVelocity = Vec3f(1, 0, 0);
		info.motionType = BodyType::Static;
		info.bodyId = 0;
		auto actor = this->createRigidBody(info);

		CapsuleInfo capsule;
		capsule.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
		capsule.radius = 0.05f;
		capsule.halfLength = 0.26f;

		float r = 0.798f;
		for (uint sec = 0; sec < 24; sec++)
		{
			float theta = sec * M_PI / 12 + 0.115;
			float y = r * sin(theta);
			float z = r * cos(theta);

			capsule.center = Vec3f(-0.042f, y, z);
			this->bindCapsule(actor, capsule);
		}

		this->bind(actor, Pair<uint, uint>(0, 0));

		//**************************************************//
		ArticulatedBody<TDataType>::resetStates();
	}

	DEFINE_CLASS(Gear);
}
