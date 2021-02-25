#pragma once
#include "Action.h"

namespace dyno
{
	class InitAct : public Action
	{
	public:
		InitAct();
		virtual ~InitAct();

	private:
		void process(Node* node) override;
	};
}
