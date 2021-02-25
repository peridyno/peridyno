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

#include "Nodes/QtNodeFlowScene.h"

QT_FORWARD_DECLARE_CLASS(QGridLayout)


using QtNodes::QtNodeFlowScene;

namespace dyno
{
	class PNodeFlowWidget : public QWidget
	{
		Q_OBJECT

	public:
		explicit PNodeFlowWidget(QWidget *parent = nullptr);
		~PNodeFlowWidget();

	signals:

	public:
		QGridLayout*		m_MainLayout;

		QtNodeFlowScene* node_scene = nullptr;
	};

}
