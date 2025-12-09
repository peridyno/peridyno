/*=========================================================================

  Program:   Scene Flow Widget
  Module:    PSceneFlowWidget.h

  Copyright (c) Yuzhong Guo
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
=========================================================================*/
#pragma once

#include <QWidget>

#include "QtMaterialFlowScene.h"
#include "Topology/MaterialManager.h"

QT_FORWARD_DECLARE_CLASS(QGridLayout)

namespace Qt
{
	class QtMaterialFlowWidget : public QWidget
	{
		Q_OBJECT

	public:
		explicit QtMaterialFlowWidget(std::shared_ptr<dyno::CustomMaterial> src , QWidget *parent = nullptr );
		~QtMaterialFlowWidget();

		//void addActor(vtkActor *actor);
		QtMaterialFlowScene* getModuleFlowScene() { return mMaterialFlow; }

	signals:

	public:
		QGridLayout*		mLayout;

		QtMaterialFlowScene* mMaterialFlow = nullptr;
	};

}
