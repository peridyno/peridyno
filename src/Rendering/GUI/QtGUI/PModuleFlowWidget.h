/*=========================================================================

  Program:   Scene Flow Widget
  Module:    PSceneFlowWidget.h

  Copyright (c) Xiaowei He
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
=========================================================================*/
#pragma once

#include <QWidget>

#include "Nodes/QtModuleFlowScene.h"
#include "Nodes/QtNodeWidget.h"

QT_FORWARD_DECLARE_CLASS(QGridLayout)


using QtNodes::QtModuleFlowScene;

namespace dyno
{
	class PModuleFlowWidget : public QWidget
	{
		Q_OBJECT

	public:
		explicit PModuleFlowWidget(QWidget *parent = nullptr, QtNodes::QtNodeWidget* node_widget = nullptr);
		~PModuleFlowWidget();

		//void addActor(vtkActor *actor);
		QtModuleFlowScene* getModuleFlowScene() { return module_scene; }

	signals:

	public:
		QGridLayout*		m_MainLayout;

		QtModuleFlowScene* module_scene = nullptr;
	};

}
