#pragma once
#include "Module.h"
#include "Base.h"

namespace dyno
{
	class NumericalModel : public Module
	{
	public:
		NumericalModel();
		~NumericalModel() override;

		virtual void step(Real dt) {};

		virtual void updateTopology() {};

		std::string getModuleType() override { return "NumericalModel"; }
	protected:
		
	private:

	};
}

