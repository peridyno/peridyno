#include "CollidableShape.h"

namespace dyno {

	IMPLEMENT_TCLASS(CollidableShape, TDataType)

	template<typename TDataType>
	CollidableShape<TDataType>::CollidableShape()
		: CollidableObject(CollidableObject::GEOMETRIC_PRIMITIVE_TYPE)
	{
	}

	DEFINE_CLASS(CollidableShape);
}