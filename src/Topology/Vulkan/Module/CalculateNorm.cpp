#include "CalculateNorm.h"

namespace dyno 
{
	IMPLEMENT_CLASS(CalculateNorm)

	CalculateNorm::CalculateNorm()
	{
		this->addKernel(
			"CalculateNorm",
			std::make_shared<VkProgram>(
				BUFFER(float),				//norm
				BUFFER(Vec3f),				//vector
				CONSTANT(uint))				//num
		);
		kernel("CalculateNorm")->load(getSpvFile("shaders/glsl/topology/CalculateNorm.comp.spv"));
	}

	void CalculateNorm::compute()
	{
		auto& inData = this->inVec()->getData();

		int num = inData.size();

		if (this->outNorm()->isEmpty())
		{
			this->outNorm()->allocate();
		}

		auto& outData = this->outNorm()->getData();
		if (outData.size() != num)
		{
			outData.resize(num);
		}

		VkConstant<uint> constNum(num);

		kernel("CalculateNorm")->flush(
			vkDispatchSize(num, 64),
			outData.handle(),
			inData.handle(),
			&constNum);
	}
}