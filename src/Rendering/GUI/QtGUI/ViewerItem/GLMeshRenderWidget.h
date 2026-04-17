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

#include <QTimer>
#include <RenderWindow.h>

#include "Topology/TriangleSet.h"
#include "Topology/TextureMesh.h"
#include "GLMeshRenderEngine.h"

#include <QOpenGLFramebufferObject>
#include <QWidget>
#include <QOpenGLExtraFunctions>
#if QT_VERSION >= QT_VERSION_CHECK(6,0,0)
#include <QtOpenGLWidgets/QtOpenGLWidgets>
#else
#include <QOpenGLWidget>
#endif


namespace dyno
{

	class GLMeshRenderEngine;

	class GLMeshRenderWidget
		: public QOpenGLWidget
		, private QOpenGLExtraFunctions
		, public RenderWindow
	{
		Q_OBJECT

	public:
		GLMeshRenderWidget(QWidget* parent = nullptr) ;
		~GLMeshRenderWidget() override ;

		//Set TextureMesh for rendering
		void setTextureMesh(const std::vector<FInstance<TextureMesh>*>& mesh) { mTextureMesh = mesh; }

		//Set TriangleSet for rendering
		void setTriangleSet(const std::vector<FInstance<TriangleSet<DataType3f>>*>& mesh) { mTriangleSet = mesh; }

		void setDefaultAnimationOption(bool op) {}

		void setTexMeshShapesID(std::vector<uint> shapes);


	public slots:
		void updateModuleGL() { mRenderEngine->updateModuleGL(); };
		void setTransparency(bool t)
		{
			mTransparency = t;
		};

	protected:
		void initializeGL() override ;
		void resizeGL(int w, int h) override ;
		void paintGL() override;
		void mousePressEvent(QMouseEvent* event) override;
		void mouseMoveEvent(QMouseEvent* event) override;
		void mouseReleaseEvent(QMouseEvent* event) override;
		void wheelEvent(QWheelEvent* event) override;

	private:
		void cleanup() ;

		// Render engine
		std::unique_ptr<GLMeshRenderEngine> mRenderEngine;

		// Mesh data
		std::vector<FInstance<TextureMesh>*> mTextureMesh;
		std::vector<FInstance<TriangleSet<DataType3f>>*> mTriangleSet;

		// Rendering state
		bool mGLInitialized = false;

		// Camera parameters
		glm::vec3 mCameraPosition = glm::vec3(0.0f, 0.0f, 5.0f);
		glm::vec3 mCameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
		glm::vec3 mCameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

		// Mouse event variables
		bool mIsDragging = false;
		Qt::MouseButton mCurrentButton = Qt::NoButton;
		QPoint mLastMousePos;
		bool mAltPressed = false;

		bool mTransparency = false;
	};
}
