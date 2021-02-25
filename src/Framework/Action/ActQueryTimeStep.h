#pragma once
#include "Action.h"

namespace dyno
{
	class QueryTimeStep : public Action
	{
	public:
		QueryTimeStep();
		virtual ~QueryTimeStep();

		float getTimeStep();
		void reset();

	private:
		void process(Node* node) override;

		float m_timestep;
	};
}