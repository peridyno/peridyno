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
		this->varGridSpacing()->setRange(0.001f, 1.0f);
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

		auto calculateAxisAlignedBoundingBox = [=](Coord& lo, Coord& hi, const Coord v0, const Coord v1, Real h, int padding) {
			int v0_i = std::floor(v0.x / h);
			int v0_j = std::floor(v0.y / h);
			int v0_k = std::floor(v0.z / h);

			int v1_i = std::ceil(v1.x / h);
			int v1_j = std::ceil(v1.y / h);
			int v1_k = std::ceil(v1.z / h);

			lo = h * Coord(v0_i - padding, v0_j - padding, v0_k - padding);
			hi = h * Coord(v1_i + padding, v1_j + padding, v1_k + padding);
			};


		Coord lo;
		Coord hi;

		if (type == BasicShapeType::CUBE)
		{
			auto cubeModel = dynamic_cast<CubeModel<TDataType>*>(shape);

			if (cubeModel != nullptr)
			{
				auto obb = cubeModel->outCube()->getValue();

				auto aabb = obb.aabb();

				auto& sdf = levelset->getSDF();

				uint padding = 5;

				calculateAxisAlignedBoundingBox(lo, hi, aabb.v0, aabb.v1, h, padding);

				sdf.setSpace(lo, hi, h);
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

				uint padding = 5;

				calculateAxisAlignedBoundingBox(lo, hi, aabb.v0, aabb.v1, h, padding);

				sdf.setSpace(lo, hi, h);
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