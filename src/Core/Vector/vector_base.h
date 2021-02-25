#pragma once
#include <iostream>
#include "Platform.h"

namespace dyno {

	template<typename T>
	class VectorBase
	{
	public:
		virtual int size()const = 0;
		virtual T& operator[] (unsigned int) = 0;
		virtual const T& operator[] (unsigned int) const = 0;
	};


	template <typename T, int Dim>
	class Vector
	{
	public:
		DYN_FUNC Vector() {};
		DYN_FUNC ~Vector() {};
	};

}  //end of namespace dyno

