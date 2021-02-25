#ifndef FRAMEWORK_ACTION_H
#define FRAMEWORK_ACTION_H

#include "Framework/Node.h"

namespace dyno
{
	class Action
	{
	public:
		Action();
		virtual ~Action();

		virtual void start(Node* node);
		virtual void process(Node* node);
		virtual void end(Node* node);
	private:

	};
}

#endif
