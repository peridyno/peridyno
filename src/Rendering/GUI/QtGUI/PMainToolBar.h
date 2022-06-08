#pragma once

#include "ToolBar/TabToolbar.h"
#include "NodeEditor/QtNodeFlowWidget.h"

namespace dyno
{
	class Node;

	class PMainToolBar : public tt::TabToolbar
	{
		Q_OBJECT
	public:
		PMainToolBar(Qt::QtNodeFlowWidget* nodeFlow, QWidget* parent = nullptr, unsigned _groupMaxHeight = 75, unsigned _groupRowCount = 3);


	Q_SIGNALS:
		void nodeCreated(std::shared_ptr<Node> node);

	private:

		Qt::QtNodeFlowWidget* mNodeFlow = nullptr;
	};
}