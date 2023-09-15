#include "PointsLoader.h"

namespace dyno
{
	IMPLEMENT_TCLASS(PointsLoader, TDataType)

	template<typename TDataType>
	PointsLoader<TDataType>::PointsLoader()
		: GeometryLoader<TDataType>()
	{
	}

	template<typename TDataType>
	PointsLoader<TDataType>::~PointsLoader()
	{
		
	}

	template<typename TDataType>
	void PointsLoader<TDataType>::resetStates()
	{
		if (this->varFileName()->getData() == "")
		{
			Log::sendMessage(Log::Error, "File name is not set!");
			return;
		}

		if (this->outPointSet()->getDataPtr() == nullptr) {
			this->outPointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		}
		
		auto filename = this->varFileName()->getData();

		auto topo = this->outPointSet()->getDataPtr();

		topo->loadObjFile(filename.string());

		topo->scale(this->varScale()->getData());
		topo->translate(this->varLocation()->getData());
	}

	DEFINE_CLASS(PointsLoader);
}