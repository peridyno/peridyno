#include "SurfaceMeshLoader.h"

namespace dyno
{
	IMPLEMENT_TCLASS(SurfaceMeshLoader, TDataType)

	template<typename TDataType>
	SurfaceMeshLoader<TDataType>::SurfaceMeshLoader()
		: Node()
	{

	}

	template<typename TDataType>
	SurfaceMeshLoader<TDataType>::~SurfaceMeshLoader()
	{
		
	}

	template<typename TDataType>
	void SurfaceMeshLoader<TDataType>::resetStates()
	{
		if (this->outTriangleSet()->getDataPtr() == nullptr) {
			this->outTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		}
		
		auto filename = this->varFileName()->getData();

		auto topo = this->outTriangleSet()->getDataPtr();

		topo->loadObjFile(filename);

		topo->scale(this->varScale()->getData());
		topo->translate(this->varLocation()->getData());
	}

	DEFINE_CLASS(SurfaceMeshLoader);
}