#include "BasicShapes/CircleModel2D.h"

#include "Primitive/Primitive2D.h"

namespace dyno
{
	template<typename TDataType>
	CircleModel2D<TDataType>::CircleModel2D()
		: BasicShape2D<TDataType>()
	{
		this->varRadius()->setRange(0.001f, 100.0f);
	}


	template<typename TDataType>
	void CircleModel2D<TDataType>::resetStates()
	{
		TCircle2D<Real> circle;
		circle.center = this->varCenter2D()->getData();
		circle.radius = this->varRadius()->getData();

		this->outCircle()->setValue(circle);
	}

	DEFINE_CLASS(CircleModel2D);
}
