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

#include <QOpenGLFramebufferObject>

#include <QTimer>

//#include <GL/glu.h>
#include <RenderEngine.h>
#include <RenderWindow.h>
#include <ImWindow.h>
#include <unordered_set>

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

		void setDefaultAnimationOption(bool op);

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

		void onSaveScreen(const std::string& filename) override;

		void paintEvent(QPaintEvent* event) override;

	public slots:
		void updateGraphicsContext();
		void updateGraphicsContext(Node* node);
		void resetSceneFrame();
		void onNodeUpdated(std::shared_ptr<Node> node);
		void onModuleUpdated(std::shared_ptr<Module> node);
		void nodeNodeRenderingKeyUpdated(std::shared_ptr<Node> node);

		void updateOneFrame(int frame);

	signals:
		void nodeSelected(std::shared_ptr<Node> node);


	private:

		QButtonState mButtonState = QButtonState::QBUTTON_UP;
		int			 mCursorX = -1;
		int			 mCursorY = -1;
		int          mtempCursorX = -1;
		// Qt
		QTimer timer;

		QOpenGLFramebufferObject* mFBO = nullptr;

		// 
		ImWindow mImWindow;

		static std::unordered_set<int> mPressedKeys;
		static std::unordered_set<int> mPressedMouseButtons;
		static bool mRreshape;
		static bool mMouseButtonRelease;
		static bool mScroll;
		bool mNeedUpdate = false;
		bool mBlockFieldUpdate = false;

	private:
		bool isAnyKeyPressed() {
			return !mPressedKeys.empty();
		}
		bool isAnyMouseButtonPressed() {
			return !mPressedMouseButtons.empty();
		}

	};

}
