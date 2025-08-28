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
		std::cout << "C++ enqueueEvent" << std::endl;
		mMutex.lock();
		std::cout << "C++ enqueueEvent lock" << std::endl;

		if (!this->varCacheEvent()->getData()) {
			while (!mEventQueue.empty()) mEventQueue.pop();
		}

		std::cout << "C++ enqueueEvent push" << std::endl;
		mEventQueue.push(event);

		mMutex.unlock();
		std::cout << "C++ enqueueEvent unlock" << std::endl;
	}

	void KeyboardInputModule::updateImpl()
	{
		std::cout << "C++ updateImpl" << std::endl;
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
		std::cout << "C++ requireUpdate" << std::endl;
		bool required = !mEventQueue.empty();

		return required || Module::requireUpdate();
	}
}