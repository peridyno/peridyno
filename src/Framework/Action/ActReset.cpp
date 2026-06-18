#include "ActReset.h"
#include "Node.h"

#include <sstream>
#include <iomanip>

namespace dyno
{
	ResetAct::ResetAct(bool Timing)
	{
		mTiming = Timing;
	}

	void ResetAct::process(Node* node)
	{
		if(node == NULL) {
			Log::sendMessage(Log::Error, "Node is invalid!");
			return;
		}

		CTimer timer;

		if (mTiming) {
			timer.start();
		}

		node->reset();

		if (mTiming) {
			timer.stop();

			std::stringstream name;
			std::stringstream ss;
			name << std::setw(40) << node->getClassInfo()->getClassName();
			ss << std::setprecision(10) << timer.getElapsedTime();

			std::string info = "Node reset: \t" + name.str() + ": \t " + ss.str() + "ms \n";
			Log::sendMessage(Log::Info, info);
		}
	}

}