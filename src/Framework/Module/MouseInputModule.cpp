#include "MouseInputModule.h"

namespace dyno
{
	MouseInputModule::MouseInputModule()
		: InputModule()
	{

	}

	MouseInputModule::~MouseInputModule()
	{
	}

	void MouseInputModule::enqueueEvent(PMouseEvent event)
	{
		if (!this->varCacheEvent()->getData()) {
			while (!mEventQueue.empty()) mEventQueue.pop();
		}

		mEventQueue.push(event);
	}

	void MouseInputModule::updateImpl()
	{
		if (!mEventQueue.empty())
		{
			onEvent(mEventQueue.front());

			mEventQueue.pop();
		}
	}

	bool MouseInputModule::requireUpdate()
	{
		bool required = !mEventQueue.empty();

		return required || Module::requireUpdate();
	}

}