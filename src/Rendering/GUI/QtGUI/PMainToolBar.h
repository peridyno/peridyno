#pragma once

#include "ToolBar/TabToolbar.h"

namespace dyno
{
	class Node;

	class PMainToolBar : public tt::TabToolbar
	{
		Q_OBJECT
	public:
		PMainToolBar(QWidget* parent = nullptr, unsigned _groupMaxHeight = 75, unsigned _groupRowCount = 3);


	Q_SIGNALS:
		void nodeCreated(std::shared_ptr<Node> node);

	private:
	};
}