#pragma once
#include "AppBase.h"

namespace dyno
{
	class WtApp : public AppBase
	{
	public:
		WtApp(int argc = 1, char** argv = NULL);
		~WtApp();

		void mainLoop() override;

	private:
		int argc_ = 1;
		char** argv_ = nullptr;
	};
}

