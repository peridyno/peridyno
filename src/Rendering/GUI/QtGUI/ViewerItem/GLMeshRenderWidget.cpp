#include "GLMeshRenderWidget.h"

#include <QMatrix4x4>
#include <QVector3D>
#include <QMouseEvent>
#include <QWheelEvent>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "GLMeshRenderEngine.h"
#include "GLShapeRender.h"

namespace dyno
{
	GLMeshRenderWidget::GLMeshRenderWidget(QWidget* parent) : QOpenGLWidget(parent)
	{
		QSurfaceFormat format;
		format.setDepthBufferSize(24);
		format.setMajorVersion(4);
		format.setMinorVersion(6);
		format.setSamples(4);
		format.setProfile(QSurfaceFormat::CoreProfile);
		setFormat(format);

		// Create render engine
		mRenderEngine = std::make_unique<GLMeshRenderEngine>();
		mRenderEngine->realisticRenderModule = std::make_shared<GLShapeRender>();

	}

	GLMeshRenderWidget::~GLMeshRenderWidget()
	{
		makeCurrent();
		cleanup();
		doneCurrent();
	}

	void GLMeshRenderWidget::setTexMeshShapesID(std::vector<uint> shapes)
	{
		auto shapeRender = std::dynamic_pointer_cast<GLShapeRender>(mRenderEngine->realisticRenderModule);
		shapeRender->varShapes()->setValue(shapes);
	}

	void GLMeshRenderWidget::initializeGL()
	{
		initializeOpenGLFunctions();
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glEnable(GL_DEPTH_TEST);

		// Initialize render engine
		if (mRenderEngine)
		{
			mRenderEngine->initialize();
			// Set default environment style
			mRenderEngine->setEnvStyle(EEnvStyle::Studio);
		}

		mGLInitialized = true;
	}

	void GLMeshRenderWidget::resizeGL(int w, int h)
	{
		glViewport(0, 0, w, h);
	}

	void GLMeshRenderWidget::paintGL()
	{
		if (!mGLInitialized || !mRenderEngine)
			return;

		// Clear color and depth buffers
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Set up render parameters
		RenderParams rparams;
		rparams.width = width();
		rparams.height = height();

		// Set up camera matrices
		// Calculate view matrix from camera parameters
		glm::mat4 view = glm::lookAt(mCameraPosition, mCameraTarget, mCameraUp);

		// Calculate projection matrix
		float aspect = float(width()) / float(height());
		glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);

		rparams.transforms.view = view;
		rparams.transforms.proj = projection;
		rparams.transforms.model = glm::mat4(1.0f);

		// Set up light
		rparams.light.mainLightDirection = glm::vec3(0.0f, 1.0f, 1.0f);
		rparams.light.mainLightColor = glm::vec3(1.0f, 1.0f, 1.0f);

		// Set other parameters
		rparams.unitScale = 1.0f;

		// Render the meshes
		mRenderEngine->renderMesh(mTextureMesh, mTriangleSet, rparams, mTransparency);
	}

	void GLMeshRenderWidget::mousePressEvent(QMouseEvent* event)
	{
		// Check if Alt key is pressed
		mAltPressed = (event->modifiers() & Qt::AltModifier);
		if (mAltPressed)
		{
			mIsDragging = true;
			mCurrentButton = event->button();
			mLastMousePos = event->pos();
		}
		QOpenGLWidget::mousePressEvent(event);
	}

	void GLMeshRenderWidget::mouseMoveEvent(QMouseEvent* event)
	{
		if (mIsDragging && mAltPressed)
		{
			QPoint delta = event->pos() - mLastMousePos;
			float sensitivity = 0.01f;

			if (mCurrentButton == Qt::LeftButton)
			{
				// Rotate camera around target
				glm::vec3 cameraDirection = glm::normalize(mCameraPosition - mCameraTarget);
				glm::vec3 cameraRight = glm::normalize(glm::cross(mCameraUp, cameraDirection));

				glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
				glm::mat4 yawMatrix = glm::rotate(glm::mat4(1.0f), -delta.x() * sensitivity, worldUp);
				glm::mat4 pitchMatrix = glm::rotate(glm::mat4(1.0f), -delta.y() * sensitivity, cameraRight);

				// Apply rotations
				glm::vec3 newCameraDirection = glm::vec3(yawMatrix * pitchMatrix * glm::vec4(cameraDirection, 0.0f));
				float distance = glm::length(mCameraPosition - mCameraTarget);
				mCameraPosition = mCameraTarget + newCameraDirection * distance;

				// Update camera up vector
				mCameraUp = glm::normalize(glm::cross(glm::cross(cameraDirection, worldUp), cameraDirection));
			}
			else if (mCurrentButton == Qt::MiddleButton)
			{
				// Pan camera
				glm::vec3 cameraDirection = glm::normalize(mCameraPosition - mCameraTarget);
				glm::vec3 cameraRight = glm::normalize(glm::cross(mCameraUp, cameraDirection));
				glm::vec3 cameraUp = mCameraUp;

				float panSpeed = 0.01f;
				glm::vec3 panVector = -cameraRight * (delta.x() * panSpeed) + cameraUp * (delta.y() * panSpeed);

				mCameraPosition += panVector;
				mCameraTarget += panVector;
			}
			else if (mCurrentButton == Qt::RightButton)
			{
				// Zoom camera
				float zoomSpeed = 0.1f;
				glm::vec3 cameraDirection = glm::normalize(mCameraPosition - mCameraTarget);
				mCameraPosition += cameraDirection * (-delta.y() * zoomSpeed);
			}

			mLastMousePos = event->pos();
			update();
		}
		QOpenGLWidget::mouseMoveEvent(event);
	}

	void GLMeshRenderWidget::mouseReleaseEvent(QMouseEvent* event)
	{
		mIsDragging = false;
		mCurrentButton = Qt::NoButton;
		QOpenGLWidget::mouseReleaseEvent(event);
	}

	void GLMeshRenderWidget::wheelEvent(QWheelEvent* event)
	{
		if (mAltPressed)
		{
			// Zoom camera with mouse wheel
			float zoomSpeed = 0.1f;
			glm::vec3 cameraDirection = glm::normalize(mCameraPosition - mCameraTarget);
			mCameraPosition -= cameraDirection * (event->angleDelta().y() * zoomSpeed * 0.01f);
			update();
		}
		QOpenGLWidget::wheelEvent(event);
	}

	void GLMeshRenderWidget::cleanup()
	{
		if (mRenderEngine)
		{
			mRenderEngine->terminate();
			mRenderEngine.reset();
		}

		mGLInitialized = false;
	}


}
