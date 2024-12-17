#include "VolumeLoader.h"

#include "Topology/LevelSet.h"

namespace dyno
{
	IMPLEMENT_TCLASS(VolumeLoader, TDataType)

	template<typename TDataType>
	VolumeLoader<TDataType>::VolumeLoader()
		: Volume<TDataType>()
	{
		this->varFileName()->attach(
			std::make_shared<FCallBackFunc>(std::bind(&VolumeLoader<TDataType>::loadFile, this))
		);
	}

	template<typename TDataType>
	VolumeLoader<TDataType>::~VolumeLoader()
	{

	}

	template<typename TDataType>
	void VolumeLoader<TDataType>::resetStates()
	{
		loadFile();
	}

	template<typename TDataType>
	bool VolumeLoader<TDataType>::loadFile()
	{
		// Validate the input filename
		if (this->varFileName()->isModified())
		{
			auto levelset = this->stateLevelSet()->getDataPtr();
			levelset->getSDF().loadSDF(this->varFileName()->getValue().string(), false);

			return true;
		}

		return false;
	}

	DEFINE_CLASS(VolumeLoader);
}