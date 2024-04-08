#include "AnchorPointToPointSet.h"

namespace dyno
{
	template<typename TDataType>
	AnchorPointToPointSet<TDataType>::AnchorPointToPointSet()
		: TopologyMapping()
	{
		this->inBallAndSocketJoints()->tagOptional(true);
		this->inSliderJoints()->tagOptional(true);
		//this->inHingeJoints()->tagOptional(true);
		//this->inFixedJoints()->tagOptional(true);
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
		if (this->outPointSet()->isEmpty())
		{
			this->outPointSet()->allocate();
		}
		uint anchorPointSize = 0;
		uint ballAndSocketJoints_size = this->inBallAndSocketJoints()->size();
		uint sliderJoints_size = this->inSliderJoints()->size();
		//uint hingeJoints_size = this->inHingeJoints()->size();
		//uint fixedJoint_size = this->inFixedJoints()->size();
		anchorPointSize += ballAndSocketJoints_size + sliderJoints_size;// +hingeJoints_size + fixedJoint_size;

		if (anchorPointSize == 0)
			return false;

		auto outset = this->outPointSet()->getDataPtr();

		auto& vertices = outset->getPoints();
		vertices.resize(anchorPointSize * 2);

		if (!this->inBallAndSocketJoints()->isEmpty())
		{
			auto& joints = this->inBallAndSocketJoints()->getData();
			cuExecute(ballAndSocketJoints_size,
				setUpAnchorPoints,
				joints,
				this->inRotationMatrix()->getData(),
				this->inCenter()->getData(),
				vertices,
				0);
		}

		if (!this->inSliderJoints()->isEmpty())
		{
			auto& joints = this->inSliderJoints()->getData();
			int begin_index = 2 * ballAndSocketJoints_size;
			cuExecute(sliderJoints_size,
				setUpAnchorPoints,
				joints,
				this->inRotationMatrix()->getData(),
				this->inCenter()->getData(),
				vertices,
				begin_index);
		}
		/*
		if (!this->inHingeJoints()->isEmpty())
		{
			auto& joints = this->inHingeJoints()->getData();
			int begin_index = 2 * (ballAndSocketJoints_size + sliderJoints_size);
			cuExecute(sliderJoints_size,
				setUpAnchorPoints,
				joints,
				this->inRotationMatrix()->getData(),
				this->inCenter()->getData(),
				vertices,
				begin_index);
		}

		if (!this->inFixedJoints()->isEmpty())
		{
			auto& joints = this->inFixedJoints()->getData();
			int begin_index = 2 * (ballAndSocketJoints_size + sliderJoints_size + hingeJoints_size);
			cuExecute(sliderJoints_size,
				setUpAnchorPoints,
				joints,
				this->inRotationMatrix()->getData(),
				this->inCenter()->getData(),
				vertices,
				begin_index);
		}*/

		return true;
	}
	DEFINE_CLASS(AnchorPointToPointSet);
}