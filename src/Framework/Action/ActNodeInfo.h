#pragma once

#include "Action.h"

namespace dyno
{
	class NodeInfoAct : public Action
	{
	public:
		NodeInfoAct();
		virtual ~NodeInfoAct();

	private:
		void process(Node* node) override;
	};
}
