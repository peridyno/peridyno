#include "BasicShapeToVolume.h"

//Basic shapes
#include "BasicShapes/CubeModel.h"
#include "BasicShapes/SphereModel.h"

namespace dyno
{
	IMPLEMENT_TCLASS(BasicShapeToVolume, TDataType)

	template<typename TDataType>
	BasicShapeToVolume<TDataType>::BasicShapeToVolume()
		: Volume<TDataType>()
	{

	}

	template<typename TDataType>
	BasicShapeToVolume<TDataType>::~BasicShapeToVolume()
	{

	}

	template<typename TDataType>
	void BasicShapeToVolume<TDataType>::resetStates()
	{
		if (this->stateLevelSet()->isEmpty()){
			this->stateLevelSet()->allocate();
		}
		convert();
	}

	template<typename TDataType>
	bool BasicShapeToVolume<TDataType>::validateInputs()
	{
		return this->getShape() != nullptr;
	}

	template<typename TDataType>
	void BasicShapeToVolume<TDataType>::convert()
	{
		auto levelset = this->stateLevelSet()->getDataPtr();

		auto shape = this->getShape();

		bool inverted = this->varInerted()->getValue();

		Real h = this->varGridSpacing()->getValue();

		BasicShapeType type = shape->getShapeType();

		if (type == BasicShapeType::CUBE)
		{
			auto cubeModel = dynamic_cast<CubeModel<TDataType>*>(shape);

			if (cubeModel != nullptr)
			{
				auto obb = cubeModel->outCube()->getValue();

				auto aabb = obb.aabb();

				auto& sdf = levelset->getSDF();

				auto lo = aabb.v0;
				auto hi = aabb.v1;

				int nx = floor((hi[0] - lo[0]) / h);
				int ny = floor((hi[1] - lo[1]) / h);
				int nz = floor((hi[2] - lo[2]) / h);

				uint padding = 5;

				sdf.setSpace(lo - padding * h, hi + padding * h, nx + 2 * padding, ny + 2 * padding, nz + 2 * padding);
				sdf.loadBox(aabb.v0, aabb.v1, inverted);
			}
		}
		else if (type == BasicShapeType::SPHERE)
		{
			auto sphereModel = dynamic_cast<SphereModel<TDataType>*>(shape);

			if (sphereModel != nullptr)
			{
				auto sphere = sphereModel->outSphere()->getValue();

				auto aabb = sphere.aabb();

				auto& sdf = levelset->getSDF();

				auto lo = aabb.v0;
				auto hi = aabb.v1;

				int nx = floor((hi[0] - lo[0]) / h);
				int ny = floor((hi[1] - lo[1]) / h);
				int nz = floor((hi[2] - lo[2]) / h);

				uint padding = 5;

				sdf.setSpace(lo - padding * h, hi + padding * h, nx + 2 * padding, ny + 2 * padding, nz + 2 * padding);
				sdf.loadSphere(sphere.center, sphere.radius, inverted);
			}
		}
		else
		{
			std::cout << "Basic shape is not supported yet " << std::endl;
		}
	}

	DEFINE_CLASS(BasicShapeToVolume);
}