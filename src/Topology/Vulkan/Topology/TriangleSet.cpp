#include "TriangleSet.h"
#include "VkTransfer.h"

namespace dyno
{

	TriangleSet::TriangleSet()
	{
		this->addKernel(
			"SetupTriangleIndices",
			std::make_shared<VkProgram>(
				BUFFER(uint32_t),			//int: height
				BUFFER(Triangle),       //in:  CapillaryTexture
				CONSTANT(uint)			//in:  horizon & realSize
				)
		);
		kernel("SetupTriangleIndices")->load(getAssetPath() + "shaders/glsl/topology/SetupTriangleIndices.comp.spv");
	}

	TriangleSet::~TriangleSet()
	{

	}

	void TriangleSet::updateTopology()
	{
		this->updateTriangles();

		EdgeSet::updateTopology();
	}

	void TriangleSet::updateTriangles()
	{
		uint num = mTriangleIndex.size();

		mIndex.resize(3 * num);

		kernel("SetupTriangleIndices")->flush(
			vkDispatchSize(num, 64),
			mIndex.handle(),
			mTriangleIndex.handle(),
			&VkConstant<uint>(num));
	}
}