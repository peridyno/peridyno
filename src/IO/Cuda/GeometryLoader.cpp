#include "GeometryLoader.h"

namespace dyno
{
	GeometryLoader::GeometryLoader()
		: Node()
	{
		this->varScale()->setValue(Vec3f(1, 1, 1));
		this->varScale()->setMin(0.01);
		this->varScale()->setMax(100.0f);
	}

	GeometryLoader::~GeometryLoader()
	{
		
	}
}