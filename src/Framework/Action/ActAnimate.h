#pragma once
#include "Action.h"

namespace dyno
{
	class AnimateAct : public Action
	{
	public:
		AnimateAct(float dt);
		virtual ~AnimateAct();

	private:
		void process(Node* node) override;

		float m_dt;
	};
}

