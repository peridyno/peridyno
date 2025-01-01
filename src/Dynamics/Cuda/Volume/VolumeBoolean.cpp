#include "VolumeBoolean.h"

#include "Module/FastMarchingMethodGPU.h"

namespace dyno
{
	IMPLEMENT_TCLASS(VolumeBoolean, TDataType)
		
	template<typename TDataType>
	VolumeBoolean<TDataType>::VolumeBoolean()
		: Volume<TDataType>()
	{
		auto fmm = std::make_shared<FastMarchingMethodGPU<TDataType>>();
		this->inA()->connect(fmm->inLevelSetA());
		this->inB()->connect(fmm->inLevelSetB());
		fmm->outLevelSet()->connect(this->outLevelSet());

		this->resetPipeline()->pushModule(fmm);

		this->varSpacing()->attach(std::make_shared<FCallBackFunc>(
			[=]() {
				fmm->varSpacing()->setValue(this->varSpacing()->getValue());
			}));

		this->varBoolType()->attach(std::make_shared<FCallBackFunc>(
			[=]() {
				fmm->varBoolType()->setCurrentKey(this->varBoolType()->currentKey());
			}));
	}

	template<typename TDataType>
	VolumeBoolean<TDataType>::~VolumeBoolean()
	{
	}

	template<typename TDataType>
	std::string VolumeBoolean<TDataType>::getNodeType()
	{
		return "Volume";
	}

	template<typename TDataType>
	void VolumeBoolean<TDataType>::resetStates()
	{
		Node::resetStates();

		this->stateLevelSet()->setDataPtr(this->outLevelSet()->constDataPtr());
	}

	DEFINE_CLASS(VolumeBoolean);
}