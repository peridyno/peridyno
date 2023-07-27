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
		mMutex.lock();

		if (this->varCacheEvent()->getValue()) {
			while (!mEventQueue.empty()) {
				auto e = mEventQueue.back();
				if (e == event)
				{
					mEventQueue.pop_back();
				}
				else
					break;
			}	
		}
		else
		{
			while (!mEventQueue.empty()) mEventQueue.pop_front();
		}

		mEventQueue.push_back(event);

		mMutex.unlock();
	}

	void MouseInputModule::updateImpl()
	{
		mMutex.lock();
		if (!mEventQueue.empty())
		{
			onEvent(mEventQueue.front());

			mEventQueue.pop_front();
		}
		mMutex.unlock();
	}

	bool MouseInputModule::requireUpdate()
	{
		bool required = !mEventQueue.empty();

		return required || Module::requireUpdate();
	}

}