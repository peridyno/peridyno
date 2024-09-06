#include "Module/VisualModule.h"
#include "Node.h"

namespace dyno
{
	VisualModule::VisualModule()
		: Module()
	{
	}

	VisualModule::~VisualModule()
	{
	}

	void VisualModule::setVisible(bool bVisible)
	{
		this->varVisible()->setValue(bVisible);
	}
}