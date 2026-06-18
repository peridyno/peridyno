#include "NeighborQuery.h"

namespace dyno
{
	NeighborQuery::NeighborQuery()
		: Module()
	{
	}

	NeighborQuery::~NeighborQuery()
	{
	}

	void NeighborQuery::performBroadPhase()
	{
		if (!this->validateInputs()) {
			return;
		}

		if (!isInitialized())
		{
			bool ret = initialize();
			if (ret == false)
				return;
		}

		if (this->requireUpdate()) {

			this->broadphase();

			//reset parameters
			for (auto param : fields_param)
			{
				param->tack();
			}

			//reset input fields
			for (auto f_in : fields_input)
			{
				f_in->tack();
			}
		}
	}

	void NeighborQuery::performNarrowPhase()
	{
		if (!this->validateInputs()) {
			return;
		}

		if (!isInitialized())
		{
			bool ret = initialize();
			if (ret == false)
				return;
		}

		this->narrowphase();

		//tag all output fields as modifed
		for (auto f_out : fields_output)
		{
			f_out->tick();
		}
	}

	void NeighborQuery::updateImpl()
	{
		this->performBroadPhase();
		this->performNarrowPhase();
	}

}