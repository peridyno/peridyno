#include "AnchorPointToPointSet.h"

namespace dyno
{
	template<typename TDataType>
	AnchorPointToPointSet<TDataType>::AnchorPointToPointSet()
		: TopologyMapping()
	{
	}

	template<typename Coord, typename Matrix, typename Joint>
	__global__ void setUpAnchorPoints(
		DArray<Joint> joints,
		DArray<Matrix> rotMat,
		DArray<Coord> pos,
		DArray<Coord> vertices,
		int begin_index
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= joints.size())
			return;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;

		Coord r1 = rotMat[idx1] * joints[tId].r1;
		Coord r2 = rotMat[idx2] * joints[tId].r2;

		Coord v0 = pos[idx1] + r1;
		Coord v1 = pos[idx2] + r2;
		vertices[2 * tId + begin_index] = v0;
		vertices[2 * tId + 1 + begin_index] = v1;
	}

	template<typename TDataType>
	bool AnchorPointToPointSet<TDataType>::apply()
	{
		auto topo = this->inDiscreteElements()->constDataPtr();

		if (this->outPointSet()->isEmpty()) {
			this->outPointSet()->allocate();
		}

		auto& ballAndSocketJoints = topo->ballAndSocketJoints();
		auto& sliderJoints = topo->sliderJoints();
		auto& hingeJoints = topo->hingeJoints();
		auto& fixedJoints = topo->fixedJoints();

		uint anchorPointSize = 0;
		uint ballAndSocketJoints_size = ballAndSocketJoints.size();
		uint sliderJoints_size = sliderJoints.size();
		uint hingeJoints_size = hingeJoints.size();
		uint fixedJoint_size = fixedJoints.size();
		anchorPointSize += ballAndSocketJoints_size + sliderJoints_size + hingeJoints_size + fixedJoint_size;

		if (anchorPointSize == 0)
			return false;

		auto outset = this->outPointSet()->getDataPtr();

		auto& vertices = outset->getPoints();
		vertices.resize(anchorPointSize * 2);

		if (ballAndSocketJoints_size > 0)
		{
			cuExecute(ballAndSocketJoints_size,
				setUpAnchorPoints,
				ballAndSocketJoints,
				this->inRotationMatrix()->getData(),
				this->inCenter()->getData(),
				vertices,
				0);
		}

		if (sliderJoints_size > 0)
		{
			int begin_index = 2 * ballAndSocketJoints_size;
			cuExecute(sliderJoints_size,
				setUpAnchorPoints,
				sliderJoints,
				this->inRotationMatrix()->getData(),
				this->inCenter()->getData(),
				vertices,
				begin_index);
		}
		
		if (hingeJoints_size > 0)
		{
			int begin_index = 2 * (ballAndSocketJoints_size + sliderJoints_size);
			cuExecute(hingeJoints_size,
				setUpAnchorPoints,
				hingeJoints,
				this->inRotationMatrix()->getData(),
				this->inCenter()->getData(),
				vertices,
				begin_index);
		}

		if (fixedJoint_size > 0)
		{
			int begin_index = 2 * (ballAndSocketJoints_size + sliderJoints_size + hingeJoints_size);
			cuExecute(fixedJoint_size,
				setUpAnchorPoints,
				fixedJoints,
				this->inRotationMatrix()->getData(),
				this->inCenter()->getData(),
				vertices,
				begin_index);
		}

		return true;
	}

	DEFINE_CLASS(AnchorPointToPointSet);
}