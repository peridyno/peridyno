#include "ActUpdateGrpahicsContext.h"
#include "Framework/Node.h"

namespace dyno
{
	UpdateGrpahicsContextAct::UpdateGrpahicsContextAct()
	{
	}

	UpdateGrpahicsContextAct::~UpdateGrpahicsContextAct()
	{

	}

	void UpdateGrpahicsContextAct::process(Node* node)
	{
		node->graphicsPipeline()->update();
	}

}