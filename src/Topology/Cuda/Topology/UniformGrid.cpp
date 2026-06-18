#include "UniformGrid.h"

#include "Object.h"
#include "DataTypes.h"

namespace dyno {

	template<typename TDataType>
	UniformGrid3D<TDataType>::UniformGrid3D()
		: Topology()
	{

	}

	template<typename TDataType>
	UniformGrid3D<TDataType>::~UniformGrid3D()
	{

	}

	template<typename TDataType>
	void UniformGrid3D<TDataType>::initialize(uint nx, uint ny, uint nz)
	{
		mNx = nx;
		mNy = ny;
		mNz = nz;
	}

	DEFINE_CLASS(UniformGrid3D);
}
