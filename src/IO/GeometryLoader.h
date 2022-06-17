#pragma once
#include "Node.h"

namespace dyno
{
	/*!
	*	\class	Geometry Loader
	*	\brief	Base class for other geometry loaders
	*/
	class GeometryLoader : public Node
	{
	public:
		GeometryLoader();
		virtual ~GeometryLoader();

		DEF_VAR(Vec3f, Location, 0, "Node location");
		DEF_VAR(Vec3f, Rotation, 0, "Node rotation");
		DEF_VAR(Vec3f, Scale, 0, "Node scale");

		DEF_VAR(std::string, FileName, "", "");
	};
}