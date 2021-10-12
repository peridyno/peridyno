#include "CollidableShape.h"

namespace dyno {

	IMPLEMENT_CLASS_1(CollidableShape, TDataType)

	template<typename TDataType>
	CollidableShape<TDataType>::CollidableShape()
		: CollidableObject(CollidableObject::GEOMETRIC_PRIMITIVE_TYPE)
	{
	}

	DEFINE_CLASS(CollidableShape);
}