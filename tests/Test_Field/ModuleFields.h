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
		DEF_EMPTY_VAR(Width, float, "Width of a rectangle");
		DEF_EMPTY_VAR(Height, float, "Height of a rectangle");

		DEF_EMPTY_VAR(Area, float, "Area of a rectangle");

	private:
		float mDerived;
	};
}