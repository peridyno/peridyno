#include "ColorMapping.h"
#include "Math/SimpleMath.h"

namespace dyno 
{
	IMPLEMENT_CLASS(ColorMapping)

	ColorMapping::ColorMapping()
	{
		this->addKernel(
			"MapHeatColor",
			std::make_shared<VkProgram>(
				BUFFER(Vec3f),				//color
				BUFFER(float),				//scalar
				UNIFORM(float),				//min
				UNIFORM(float),				//max
				CONSTANT(uint))				//num
		);
		kernel("MapHeatColor")->load(getAssetPath() + "shaders/glsl/graphics/MapHeatColor.comp.spv");

		this->addKernel(
			"MapJetColor",
			std::make_shared<VkProgram>(
				BUFFER(Vec3f),				//color
				BUFFER(float),				//scalar
				UNIFORM(float),				//min
				UNIFORM(float),				//max
				CONSTANT(uint))				//num
		);
		kernel("MapJetColor")->load(getAssetPath() + "shaders/glsl/graphics/MapJetColor.comp.spv");
	}

	void ColorMapping::compute()
	{
		auto& inData = this->inScalar()->getData();

		uint num = inData.size();

		if (this->outColor()->isEmpty())
		{
			this->outColor()->allocate();
		}

		auto& outData = this->outColor()->getData();
		if (outData.size() != num)
		{
			outData.resize(num);
		}

		VkUniform<float> uniMin;
		VkUniform<float> uniMax;
		uniMin.setValue(this->varMin()->getValue());
		uniMax.setValue(this->varMax()->getValue());

		VkConstant<uint> constNum(num);

		if(this->varType()->getData() == ColorTable::Jet)
		{
			kernel("MapJetColor")->flush(
				vkDispatchSize(num, 64),
				outData.handle(),
				inData.handle(),
				&uniMin,
				&uniMax,
				&constNum);
		}
		else if(this->varType()->getData() == ColorTable::Heat)
		{
			kernel("MapHeatColor")->flush(
				vkDispatchSize(num, 64),
				outData.handle(),
				inData.handle(),
				&uniMin,
				&uniMax,
				&constNum);
		}
	}
}