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

#include "QtModuleFlowScene.h"
#include "QtNodeWidget.h"

QT_FORWARD_DECLARE_CLASS(QGridLayout)

namespace Qt
{
	class QtModuleFlowWidget : public QWidget
	{
		Q_OBJECT

	public:
		explicit QtModuleFlowWidget(QWidget *parent = nullptr, QtNodeWidget* node_widget = nullptr);
		~QtModuleFlowWidget();

		//void addActor(vtkActor *actor);
		QtModuleFlowScene* getModuleFlowScene() { return mModuleFlow; }

	signals:

	public:
		QGridLayout*		mLayout;

		QtModuleFlowScene* mModuleFlow = nullptr;
	};

}
