/*=========================================================================

  Program:   VTK-based Visualization Widget
  Module:    VTKOpenGLWidget.h

  Copyright (c) Xiaowei He
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
=========================================================================*/
#ifndef VTKOPENGLWIDGET_H
#define VTKOPENGLWIDGET_H

#include <QWidget>

//VTK
#include <vtkSmartPointer.h>

QT_FORWARD_DECLARE_CLASS(QGridLayout)

class vtkActor;
class vtkRenderer;
class vtkOrientationMarkerWidget;
class QVTKOpenGLWidget;

namespace dyno
{
	class PVTKOpenGLWidget : public QWidget
	{
		Q_OBJECT

	public:
		explicit PVTKOpenGLWidget(QWidget *parent = nullptr);
		~PVTKOpenGLWidget();

		//void addActor(vtkActor *actor);

	signals:

	public slots:
		void showAxisWidget();
		void prepareRenderingContex();
		void redisplay();

	public:
		QGridLayout*		m_MainLayout;

		QVTKOpenGLWidget*						m_OpenGLWidget;
		vtkOrientationMarkerWidget*				m_axisWidget;

	public:
		static vtkRenderer* getCurrentRenderer() {	return g_current_renderer;	}
		static vtkRenderer*		g_current_renderer;
	};

}

#endif // VTKOPENGLWIDGET_H
