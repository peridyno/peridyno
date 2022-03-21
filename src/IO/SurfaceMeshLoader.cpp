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
		if (this->outTriangularMesh()->getDataPtr() == nullptr) {
			this->outTriangularMesh()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		}
		
		auto filename = this->varFileName()->getData();

		auto topo = this->outTriangularMesh()->getDataPtr();

		topo->loadObjFile(filename);
	}

	DEFINE_CLASS(SurfaceMeshLoader);
}