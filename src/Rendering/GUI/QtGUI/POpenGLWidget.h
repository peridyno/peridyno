/**
 * Program:   OpenGL-based Visualization Widget
 * Module:    POpenGLWidget.h
 *
 * Copyright 2021 Xiaowei He
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
//Qt
#include <QWidget>
#include <QOpenGLExtraFunctions>
#include <QOpenGLWidget>
#include <QTimer>

//#include <GL/glu.h>
#include <Rendering.h>
#include <ImWindow.h>

namespace dyno
{
	class ImWidget;
	class RenderEngine;
	class Camera;

	enum QButtonState
	{
		QBUTTON_DOWN = 0,
		QBUTTON_UP
	};

	enum QButtonType
	{
		QBUTTON_LETF = 0,
		QBUTTON_RIGHT,
		QBUTTON_UNKNOWN
	};

	class POpenGLWidget
		: public QOpenGLWidget
		, private QOpenGLExtraFunctions
	{
		Q_OBJECT
	public:
		POpenGLWidget(RenderEngine* engine);
		~POpenGLWidget();

	protected:
		void initializeGL() override;		
		void paintGL() override;		
		void resizeGL(int w, int h) override;
		
		void mousePressEvent(QMouseEvent *event) override;
		void mouseReleaseEvent(QMouseEvent *event) override;
		void mouseMoveEvent(QMouseEvent *event) override;
		void wheelEvent(QWheelEvent *event) override;

	public slots:
		void updateGraphicsContext();

	private:
		std::shared_ptr<Camera> activeCamera();

		RenderEngine* mRenderEngine;

		QButtonState mButtonState = QButtonState::QBUTTON_UP;
		// Qt
		QTimer timer;

		// 
		ImWindow mImWindow;
	};

}
