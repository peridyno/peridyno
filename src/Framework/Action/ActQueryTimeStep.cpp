#include "ActQueryTimeStep.h"

#include <algorithm>

namespace dyno
{
	
	QueryTimeStep::QueryTimeStep()
		: Action()
		, m_timestep(0.033f)
	{

	}

	QueryTimeStep::~QueryTimeStep()
	{

	}

	float QueryTimeStep::getTimeStep()
	{
		return m_timestep;
	}

	void QueryTimeStep::reset()
	{
		m_timestep = 0.033f;
	}

	void QueryTimeStep::process(Node* node)
	{
		m_timestep = std::min(node->getDt(), m_timestep);
	}

}