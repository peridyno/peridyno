#pragma once
#include "Action.h"

namespace dyno
{
	class ResetAct : public Action
	{
	public:
		ResetAct();
		virtual ~ResetAct();

	private:
		void process(Node* node) override;
	};
}
