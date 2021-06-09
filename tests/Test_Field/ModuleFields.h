#pragma once
#include "Framework/Module.h"

namespace dyno {

#define CALLBACK void

	class ModuleFields : public Module
	{
	public:
		ModuleFields(std::string name);
		~ModuleFields() override {};

		CALLBACK calculateRectangleArea();

	public:
		DEF_VAR_IN(float, Width, "Width of a rectangle");
		DEF_VAR_IN(float, Height, "Height of a rectangle");

		DEF_VAR_OUT(float, Area, "Area of a rectangle");

	private:
		float mDerived;
	};
}