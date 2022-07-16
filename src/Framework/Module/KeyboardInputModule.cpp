#include "KeyboardInputModule.h"

namespace dyno
{
	KeyboardInputModule::KeyboardInputModule()
		: InputModule()
	{

	}

	KeyboardInputModule::~KeyboardInputModule()
	{
	}

	void KeyboardInputModule::enqueueEvent(PKeyboardEvent event)
	{
		if (!this->varCacheEvent()->getData()) {
			while (!mEventQueue.empty()) mEventQueue.pop();
		}

		mEventQueue.push(event);
	}

	void KeyboardInputModule::updateImpl()
	{
	}

	bool KeyboardInputModule::requireUpdate()
	{
		bool required = !mEventQueue.empty();

		return required || Module::requireUpdate();
	}

}