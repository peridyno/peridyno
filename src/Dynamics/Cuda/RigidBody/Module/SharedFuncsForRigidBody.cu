#include "SharedFuncsForRigidBody.h"

namespace dyno
{
	__global__ void SF_ApplyTransform(
		DArrayList<Transform3f> instanceTransform,
		const DArray<Vec3f> diff,
		const DArray<Vec3f> translate,
		const DArray<Mat3f> rotation,
		const DArray<Mat3f> rotationInit,
		const DArray<Pair<uint, uint>> binding)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= rotation.size())
			return;

		Pair<uint, uint> pair = binding[tId];

		Transform3f ti = Transform3f(translate[tId] + diff[tId], rotation[tId] * rotationInit[tId].transpose());

		instanceTransform[pair.first][pair.second] = ti;
	}

	void ApplyTransform(
		DArrayList<Transform3f>& instanceTransform, 
		const DArray<Vec3f>& diff,
		const DArray<Vec3f>& translate,
		const DArray<Mat3f>& rotation,
		const DArray<Mat3f>& rotationInit,
		const DArray<Pair<uint, uint>>& binding)
	{
		cuExecute(rotation.size(),
			SF_ApplyTransform,
			instanceTransform,
			diff,
			translate,
			rotation,
			rotationInit,
			binding);

	}
}