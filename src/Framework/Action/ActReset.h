#pragma once
#include "Action.h"
#include "Timer.h"

namespace dyno
{
	class ResetAct : public Action
	{
	public:
		ResetAct(bool Timing);

		void process(Node* node) override;

	private:
		bool mTiming = false;
	};
}
