#include "PhaseField.h"

#include "Object.h"
#include "DataTypes.h"

namespace dyno {

	template<typename TDataType>
	PhaseField<TDataType>::PhaseField()
		: UniformGrid3D<TDataType>()
	{

	}

	template<typename TDataType>
	PhaseField<TDataType>::~PhaseField()
	{

	}

	template<typename TDataType>
	void PhaseField<TDataType>::initialize(uint nx, uint ny, uint nz)
	{
		mVolumeFraction.resize(nx, ny, nz);
		mVolumeFraction.reset();

		UniformGrid3D<TDataType>::initialize(nx, ny, nz);
	}

	DEFINE_CLASS(PhaseField)
}
