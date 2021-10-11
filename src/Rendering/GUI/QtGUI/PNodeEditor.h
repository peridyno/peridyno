#pragma once

#include <QMainWindow>

#include "Nodes/QtNodeWidget.h"

namespace dyno
{
	class PNodeEditor :
		public QMainWindow
	{
		Q_OBJECT
	public:
		PNodeEditor(QtNodes::QtNodeWidget* node_widget);

	private:
	};
}

