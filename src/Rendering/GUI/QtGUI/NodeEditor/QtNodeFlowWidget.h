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

#include "QtNodeFlowScene.h"

QT_FORWARD_DECLARE_CLASS(QGridLayout)

namespace Qt
{
	class QtNodeFlowWidget : public QWidget
	{
		Q_OBJECT

	public:
		explicit QtNodeFlowWidget(QWidget *parent = nullptr);
		~QtNodeFlowWidget();

		inline QtNodeFlowScene* flowScene() { return node_scene; }

	signals:

	private:
		QGridLayout*		m_MainLayout;

		QtNodeFlowScene* node_scene = nullptr;
	};
}
