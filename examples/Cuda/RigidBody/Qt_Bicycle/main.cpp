#include <QtApp.h>
#include <GlfwGUI/GlfwApp.h>

#include <SceneGraph.h>

#include <BasicShapes/PlaneModel.h>

#include "GltfLoader.h"
#include "BasicShapes/PlaneModel.h"
#include "RigidBody/MultibodySystem.h"
#include "RigidBody/Vehicle.h"

#include "Module/KeyboardInputModule.h"

using namespace std;
using namespace dyno;

template<typename TDataType>
class BycircleDriver : public KeyboardInputModule
{
	DECLARE_TCLASS(BycircleDriver, TDataType);
public:
	typedef typename TDataType::Real Real;

	BycircleDriver() {};
	~BycircleDriver() override {};

public:
	DEF_VAR_IN(Real, Steering, "");
	DEF_VAR_IN(Real, Thrust, "");

protected:
	void onEvent(PKeyboardEvent event) override
	{
		Real stepAngle = M_PI / 50;
		
		if (event.key == PKeyboardType::PKEY_A || event.key == PKeyboardType::PKEY_D)
		{
			auto angle = this->inSteering()->getValue();

			if (event.key == PKeyboardType::PKEY_A)
			{
				angle += stepAngle;
			}
			else
				angle -= stepAngle;

			this->inSteering()->setValue(angle);
		}
		else if (event.key == PKeyboardType::PKEY_W || event.key == PKeyboardType::PKEY_S)
		{
			auto speed = this->inThrust()->getValue();

			switch (event.key)
			{
			case PKeyboardType::PKEY_W:
				speed += 0.5;
				break;

			case PKeyboardType::PKEY_S:
				speed -= 0.5;
				break;
			}
			speed = std::clamp(speed, 0.0f, 5.0f);

			this->inThrust()->setValue(speed);
		}
	}
};

IMPLEMENT_TCLASS(BycircleDriver, TDataType);

DEFINE_CLASS(BycircleDriver);

std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto bike = scn->addNode(std::make_shared<Bicycle<DataType3f>>());


	auto multisystem = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());

	auto driver = std::make_shared<BycircleDriver<DataType3f>>();
	bike->stateSteeringAngle()->connect(driver->inSteering());
	bike->stateThrust()->connect(driver->inThrust());
	bike->animationPipeline()->pushModule(driver);

// 	Key2HingeConfig keyConfig;
// 	//keyConfig.addMap(PKeyboardType::PKEY_W, 0, 1);
// 	//keyConfig.addMap(PKeyboardType::PKEY_S, 0, -1);
// 
// 	keyConfig.addMap(PKeyboardType::PKEY_W, 1, 1);
// 	keyConfig.addMap(PKeyboardType::PKEY_S, 1, -1);
// 
// 	keyConfig.addMap(PKeyboardType::PKEY_D, 2, 1);
// 	keyConfig.addMap(PKeyboardType::PKEY_A, 2, -1);
// 	driver->varHingeKeyConfig()->setValue(keyConfig);

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	bike->connect(multisystem->importVehicles());
	plane->stateTriangleSet()->connect(multisystem->inTriangleSet());
	plane->varLengthX()->setValue(120);
	plane->varLengthZ()->setValue(120);
	plane->varLocation()->setValue(Vec3f(0,-0.5,0));

	return scn;
}

int main()
{
	//QtApp app;
	GlfwApp app;
	app.setSceneGraph(creatCar());
	app.initialize(1280, 768);

	//Set the distance unit for the camera, the fault unit is meter
	//app.renderWindow()->getCamera()->setUnitScale(3.0f);

	app.mainLoop();

	return 0;
}
