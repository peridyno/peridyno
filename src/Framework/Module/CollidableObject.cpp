#include "CollidableObject.h"

namespace dyno
{
	CollidableObject::CollidableObject(CType ctype)
	{
		m_type = ctype;
	}

	CollidableObject::~CollidableObject()
	{
	}
}