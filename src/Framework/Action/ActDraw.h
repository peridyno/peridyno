#ifndef FRAMEWORK_NODEINFOACT_H
#define FRAMEWORK_NODEINFOACT_H

#include "Action.h"

namespace dyno
{
	class DrawAct : public Action
	{
	public:
		DrawAct();
		virtual ~DrawAct();

	private:
		void process(Node* node) override;
	};
}

#endif
