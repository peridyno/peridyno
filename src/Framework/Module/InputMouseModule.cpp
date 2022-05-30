#include "Module/InputMouseModule.h"

namespace dyno
{
	InputMouseModule::InputMouseModule()
		: Module()
	{

	}

	InputMouseModule::~InputMouseModule()
	{
	}

	void InputMouseModule::enqueueEvent(PMouseEvent event)
	{
		mEventQueue.push(event);
	}

	void InputMouseModule::updateImpl()
	{
		if (!mEventQueue.empty())
		{
			onEvent(mEventQueue.front());

			mEventQueue.pop();
		}
	}

}