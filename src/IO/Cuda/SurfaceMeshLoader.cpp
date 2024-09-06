#include "SurfaceMeshLoader.h"

namespace dyno
{
	IMPLEMENT_TCLASS(SurfaceMeshLoader, TDataType)

	template<typename TDataType>
	SurfaceMeshLoader<TDataType>::SurfaceMeshLoader()
		: GeometryLoader<TDataType>()
	{
	}

	template<typename TDataType>
	SurfaceMeshLoader<TDataType>::~SurfaceMeshLoader()
	{
		
	}

	template<typename TDataType>
	void SurfaceMeshLoader<TDataType>::resetStates()
	{
		if (this->varFileName()->getData() == "")
		{
			Log::sendMessage(Log::Error, "File name is not set!");
			return;
		}

		if (this->outTriangleSet()->getDataPtr() == nullptr) {
			this->outTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		}
		
		auto filename = this->varFileName()->getData();

		auto topo = this->outTriangleSet()->getDataPtr();

		topo->loadObjFile(filename.string());

		topo->scale(this->varScale()->getData());
		topo->translate(this->varLocation()->getData());
	}

	DEFINE_CLASS(SurfaceMeshLoader);
}