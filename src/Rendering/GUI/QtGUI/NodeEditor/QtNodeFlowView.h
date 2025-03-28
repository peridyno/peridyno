#pragma once

#include "nodes/QNode"
#include "nodes/QFlowView"

namespace Qt
{
	class QtNodeFlowView
		: public QtFlowView
	{
		Q_OBJECT
	public:

		QtNodeFlowView(QWidget *parent = Q_NULLPTR);
		QtNodeFlowView(QtFlowScene *scene, QWidget *parent = Q_NULLPTR);

		QtNodeFlowView(const QtNodeFlowView&) = delete;
		QtNodeFlowView operator=(const QtNodeFlowView&) = delete;

	public Q_SLOTS:
		void showPortContextMenu(QtNode& n, const PortIndex index, const QPointF& pos);
	};
}
