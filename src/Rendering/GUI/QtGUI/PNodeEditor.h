#pragma once

#include <QMainWindow>

#include "NodeEditor/QtNodeWidget.h"

namespace dyno
{
	class PNodeEditor :
		public QMainWindow
	{
		Q_OBJECT
	public:
		PNodeEditor(Qt::QtNodeWidget* node_widget);

	private:
	};
}

