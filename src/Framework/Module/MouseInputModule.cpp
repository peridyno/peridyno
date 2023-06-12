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

		if (!this->varCacheEvent()->getData()) {
			while (!mEventQueue.empty()) {
				auto e = mEventQueue.front();
				if (e == event)
				{
					mEventQueue.pop();
				}
				else
					break;
			}	
		}

		mEventQueue.push(event);

		mMutex.unlock();
	}

	void MouseInputModule::updateImpl()
	{
		mMutex.lock();
		if (!mEventQueue.empty())
		{
			onEvent(mEventQueue.front());

			mEventQueue.pop();
		}
		mMutex.unlock();
	}

	bool MouseInputModule::requireUpdate()
	{
		bool required = !mEventQueue.empty();

		return required || Module::requireUpdate();
	}

}