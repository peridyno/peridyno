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
		mMutex.lock();

		if (!this->varCacheEvent()->getData()) {
			while (!mEventQueue.empty()) mEventQueue.pop();
		}

		mEventQueue.push(event);

		mMutex.unlock();
	}

	void KeyboardInputModule::updateImpl()
	{
		mMutex.lock();
		if (!mEventQueue.empty())
		{
			onEvent(mEventQueue.front());

			mEventQueue.pop();
		}
		mMutex.unlock();
	}

	bool KeyboardInputModule::requireUpdate()
	{
		bool required = !mEventQueue.empty();

		return required || Module::requireUpdate();
	}

}