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
#if QT_VERSION >= QT_VERSION_CHECK(6,0,0)
	#include <QtOpenGLWidgets/QtOpenGLWidgets>
#else
	#include <QOpenGLWidget>
#endif
#include <QTimer>

//#include <GL/glu.h>
#include <RenderEngine.h>
#include <RenderWindow.h>
#include <ImWindow.h>

namespace dyno
{
	class Node;

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
		, public RenderWindow
	{
		Q_OBJECT
	public:
		POpenGLWidget(QWidget* parent = nullptr);
		~POpenGLWidget();

		void mainLoop() override {};

	protected:
		void initializeGL() override;		
		void paintGL() override;		
		void resizeGL(int w, int h) override;
		
		void mousePressEvent(QMouseEvent *event) override;
		void mouseReleaseEvent(QMouseEvent *event) override;
		void mouseMoveEvent(QMouseEvent *event) override;
		void wheelEvent(QWheelEvent *event) override;

		void keyPressEvent(QKeyEvent* event) override;
		void keyReleaseEvent(QKeyEvent* event) override;

		void onSelected(const Selection& s) override;

	public slots:
		void updateGrpahicsContext();
		void updateGraphicsContext(Node* node);

	signals:
		void nodeSelected(std::shared_ptr<Node> node);


	private:

		QButtonState mButtonState = QButtonState::QBUTTON_UP;
		int			 mCursorX = -1;
		int			 mCursorY = -1;
		
		// Qt
		QTimer timer;

		// 
		ImWindow mImWindow;
	};

}
