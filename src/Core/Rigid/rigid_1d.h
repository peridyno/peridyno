#pragma once
#include <iostream>
#include "rigid_base.h"

namespace dyno {
	template <typename T>
	class Rigid<T, 1>
	{
	public:
		DYN_FUNC Rigid() {};
		DYN_FUNC ~Rigid() {};

	private:
		
	};

}  //end of namespace dyno

