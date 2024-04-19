#include "SimpleVechicleDriver.h"

namespace dyno
{
	IMPLEMENT_CLASS(SimpleVechicleDriver)

	SimpleVechicleDriver::SimpleVechicleDriver()
		: ComputeModule()
	{
	}

	SimpleVechicleDriver::~SimpleVechicleDriver()
	{

	}

	void SimpleVechicleDriver::compute()
	{
		CArrayList<Transform3f> tms;
		tms.assign(this->inInstanceTransform()->constData());

		for (uint i = 0; i < tms.size(); i++)
		{
			Transform3f& t = tms[i][0];
			t.translation() += Vec3f(0, 0, 0.01);

			if (i == 0 || i == 1 || i == 2 || i == 3)
			{
				t.rotation() = Quat1f(theta, Vec3f(1, 0, 0)).toMatrix3x3();
			}
		}

		auto instantanceTransform = this->inInstanceTransform()->getDataPtr();
		instantanceTransform->assign(tms);

		theta += 0.01 * M_PI;

		tms.clear();
	}
}