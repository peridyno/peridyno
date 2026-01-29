#include "LightController.h"

namespace dyno
{
	IMPLEMENT_CLASS(LightController);

	LightController::LightController()
		: KeyboardInputModule()
	{


	}

	void LightController::onEvent(PKeyboardEvent event)
	{

		CArrayList<uint> idList;
		idList.assign(this->inShapeVehicleID()->constData());
		
		CArray<uint> num(idList.size());
		for (size_t i = 0; i < idList.size(); i++)
		{
			num[i] = idList[i].size();
		}

		this->outLightDirection()->setValue(this->varLightDirection()->getValue());
		
		uint id = this->varVehicleID()->getValue();

		switch (event.key)
		{
		case PKeyboardType::PKEY_Q:
		{
			if (event.action == PActionType::AT_PRESS) 
			{
				CArrayList<float> head;
				head.resize(num);
				for (size_t i = 0; i < idList.size(); i++)
				{
					const auto& list = idList[i];
					for (size_t j = 0; j < list.size(); j++)
					{
						if (list[j] == id)
							head[i][j] = active ? 0 : 1;
					}
				}

				active = !active;

				if (this->outHeadLight()->isEmpty())
					this->outHeadLight()->allocate();
				this->outHeadLight()->getDataPtr()->assign(head);
			}
		}
		break;
		case PKeyboardType::PKEY_E:
		{
			CArrayList<float> turn;
			turn.resize(num);
			for (size_t i = 0; i < idList.size(); i++)
			{
				const auto& list = idList[i];
				for (size_t j = 0; j < list.size(); j++)
				{		
					turn[i][j] = 0;
				}
			}

			if (this->outTurnSignal()->isEmpty())
				this->outTurnSignal()->allocate();
			this->outTurnSignal()->getDataPtr()->assign(turn);
		}
		break;
		case PKeyboardType::PKEY_W:
		{
			CArrayList<float> brak;
			brak.resize(num);
			for (size_t i = 0; i < idList.size(); i++)
			{
				const auto& list = idList[i];
				for (size_t j = 0; j < list.size(); j++)
				{
					brak[i][j] = 0;
				}
			}
			if (this->outBrakeLight()->isEmpty())
				this->outBrakeLight()->allocate();
			this->outBrakeLight()->getDataPtr()->assign(brak);
		}
		break;
		case PKeyboardType::PKEY_S:
		{
			CArrayList<float> brak;
			brak.resize(num);
			for (size_t i = 0; i < idList.size(); i++)
			{
				const auto& list = idList[i];
				for (size_t j = 0; j < list.size(); j++)
				{
					if (list[j] == id)
						brak[i][j] = 1;

				}
			}
			if (this->outBrakeLight()->isEmpty())
				this->outBrakeLight()->allocate();
			this->outBrakeLight()->getDataPtr()->assign(brak);
		}
		break;
		case PKeyboardType::PKEY_A:
		{		
			CArrayList<float> turn;
			turn.resize(num);
			for (size_t i = 0; i < idList.size(); i++)
			{
				const auto& list = idList[i];
				for (size_t j = 0; j < list.size(); j++)
				{
					if (list[j] == id)
						turn[i][j] = -1;
				}
			}
			if (this->outTurnSignal()->isEmpty())
				this->outTurnSignal()->allocate();
			this->outTurnSignal()->getDataPtr()->assign(turn);
		}
		break;
		case PKeyboardType::PKEY_D:
		{
			CArrayList<float> turn;
			turn.resize(num);
			for (size_t i = 0; i < idList.size(); i++)
			{
				const auto& list = idList[i];
				for (size_t j = 0; j < list.size(); j++)
				{
					if (list[j] == id)
						turn[i][j] = 1;

				}
			}

			if (this->outTurnSignal()->isEmpty())
				this->outTurnSignal()->allocate();
			this->outTurnSignal()->getDataPtr()->assign(turn);
		}
		break;
		default:
			break;
		}




	}
	
}