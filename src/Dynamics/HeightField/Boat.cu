#include "Boat.h"
#include "StaticTriangularMesh.h"
namespace dyno
{
	template<typename TDataType>
	Boat<TDataType>::Boat()
	{
		

	}

	template<typename TDataType>
	Boat<TDataType>::~Boat()
	{

	}

	template<typename TDataType>
	void Boat<TDataType>::resetStates()
	{
		auto staticMesh = std::make_shared<StaticTriangularMesh<DataType3f>>();
		auto meshPath = this->varFileName()->getData();
		if (meshPath.empty()) {
			staticMesh->varFileName()->setValue(getAssetPath() + "bunny/sparse_bunny_mesh.obj");
		}
		else {
			staticMesh->varFileName()->setValue(meshPath);
		}
		
	}

	DEFINE_CLASS(Boat);
}