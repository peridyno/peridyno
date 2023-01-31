#include "UniformGrid.h"

namespace dyno {

	UniformGrid3D::UniformGrid3D()
	{

	}

	UniformGrid3D::~UniformGrid3D()
	{

	}

	GridInfo UniformGrid3D::getGridInfo()
	{
		GridInfo info;
		info.nx = mDimension.x;
		info.ny = mDimension.y;
		info.nz = mDimension.z;

		info.ox = mOrigin.x;
		info.oy = mOrigin.y;
		info.oz = mOrigin.z;

		info.spacing = mSpacing;

		return info;
	}

}
