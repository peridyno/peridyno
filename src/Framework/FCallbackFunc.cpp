#include "FCallbackFunc.h"

#include "FBase.h"

namespace dyno
{

	FCallBackFunc::FCallBackFunc(std::function<void()> func)
	{
		mCallback = func;
	}

	void FCallBackFunc::update()
	{
		for(auto f : mInputs)
		{
			if (f->isEmpty())
				return;
		}

		mCallback();
	}

	void FCallBackFunc::addInput(FBase* f)
	{
		mInputs.push_back(f);
	}

}