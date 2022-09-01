#include "PointSetToTriangleSet.h"

#include "PointSetToPointSet.h"

namespace dyno
{
	template<typename TDataType>
	PointSetToTriangleSet<TDataType>::PointSetToTriangleSet()
		: Node()
	{

	}

	template<typename TDataType>
	PointSetToTriangleSet<TDataType>::~PointSetToTriangleSet()
	{

	}

	template<typename TDataType>
	void PointSetToTriangleSet<TDataType>::resetStates()
	{
		if (this->outShape()->isEmpty()) {
			this->outShape()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		}


		auto initial = this->inInitialShape()->getDataPtr();
		auto shape = this->outShape()->getDataPtr();

		shape->copyFrom(*initial);

		mPointMapper = std::make_shared<PointSetToPointSet<TDataType>>();
		mPointMapper->setUpdateAlways(true);

		mPointMapper->setFrom(this->inPointSet()->getDataPtr());
		mPointMapper->setTo(shape);
		mPointMapper->setSearchingRadius(mRadius);
		mPointMapper->initialize();
	}

	template<typename TDataType>
	void PointSetToTriangleSet<TDataType>::updateStates()
	{
		mPointMapper->update();
	}

	DEFINE_CLASS(PointSetToTriangleSet);
}