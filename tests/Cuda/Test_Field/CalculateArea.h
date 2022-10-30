#pragma once
#include "Module/ComputeModule.h"

namespace dyno {
	class CalculateArea : public ComputeModule
	{
	public:
		CalculateArea(std::string name);
		~CalculateArea() override {};

		void compute() override;

	public:
		DEF_VAR_IN(float, Width, "Width of a rectangle");
		DEF_VAR_IN(float, Height, "Height of a rectangle");

		DEF_VAR_OUT(float, Area, "Area of a rectangle");
	};
}