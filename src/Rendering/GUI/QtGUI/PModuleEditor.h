#pragma once

#include <QMainWindow>

#include "NodeEditor/QtNodeWidget.h"

namespace dyno
{
	class PModuleEditor :
		public QMainWindow
	{
		Q_OBJECT
	public:
		PModuleEditor(Qt::QtNodeWidget* node_widget);

	private:
	};
}

