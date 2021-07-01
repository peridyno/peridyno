#pragma once
#include "Action.h"

namespace dyno
{
	class UpdateGrpahicsContextAct : public Action
	{
	public:
		UpdateGrpahicsContextAct();
		virtual ~UpdateGrpahicsContextAct();

	private:
		void process(Node* node) override;
	};
}

