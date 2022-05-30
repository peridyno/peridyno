#pragma once

#include "Module/CollidableObject.h"

namespace dyno {

	template<typename TDataType>
	class CollidableShape : public CollidableObject
	{
		DECLARE_TCLASS(CollidableShape, TDataType)
	public:
		CollidableShape();

		//should be called before collision is started
		void updateCollidableObject() override {};

		//should be called after the collision is finished
		void updateMechanicalState() override {};
	private:
	};
}