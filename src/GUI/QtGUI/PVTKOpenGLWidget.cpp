#include "PVTKOpenGLWidget.h"

#include "Framework/SceneGraph.h"
#include "PSimulationThread.h"

//VTK
#include <vtkActor.h>
#include <vtkAxesActor.h>
#include <vtkSphereSource.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkRenderWindowInteractor.h>
#include <QVTKOpenGLWidget.h>

//QT
#include <QGridLayout>

namespace dyno
{
	vtkRenderer* PVTKOpenGLWidget::g_current_renderer = vtkRenderer::New();

	PVTKOpenGLWidget::PVTKOpenGLWidget(QWidget *parent) :
		QWidget(parent)
	{
		m_MainLayout = new QGridLayout();
		this->setLayout(m_MainLayout);

		m_OpenGLWidget = new QVTKOpenGLWidget();
		m_MainLayout->addWidget(m_OpenGLWidget, 0, 0);

		auto renderWindow = m_OpenGLWidget->GetRenderWindow();

		renderWindow->AddRenderer(PVTKOpenGLWidget::getCurrentRenderer());

		vtkSmartPointer<vtkSphereSource> sphereSource = vtkSmartPointer<vtkSphereSource>::New();
		sphereSource->SetCenter(0.0, 0.0, 0.0);
		sphereSource->SetRadius(1.0);
		sphereSource->Update();

		vtkPolyData* polydata = sphereSource->GetOutput();

		// Create a mapper
		vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
		mapper->SetInputData(polydata);

		// Create an actor
		vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
		actor->SetMapper(mapper);

		// Add the actors to the scene
//		g_renderer->AddActor(actor);
		g_current_renderer->SetBackground(.2, .3, .4);


		vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();

		m_axisWidget = vtkOrientationMarkerWidget::New();
		m_axisWidget->SetOutlineColor(0.9300, 0.5700, 0.1300);
		m_axisWidget->SetOrientationMarker(axes);
		m_axisWidget->SetCurrentRenderer(PVTKOpenGLWidget::getCurrentRenderer());
		m_axisWidget->SetInteractor(m_OpenGLWidget->GetInteractor());
		m_axisWidget->SetViewport(0.0, 0.0, 0.2, 0.2);
		m_axisWidget->SetEnabled(1);
		m_axisWidget->InteractiveOn();

		PVTKOpenGLWidget::getCurrentRenderer()->ResetCamera();
	}

	PVTKOpenGLWidget::~PVTKOpenGLWidget()
	{
		delete m_MainLayout;
		m_MainLayout = nullptr;

// 		m_renderer->Delete();
// 		m_renderer = nullptr;

		m_axisWidget->Delete();
		m_axisWidget = nullptr;
	}

// 	void PVTKOpenGLWidget::addActor(vtkActor *actor)
// 	{
// 		m_renderer->AddActor(actor);
// 	}

	void PVTKOpenGLWidget::showAxisWidget()
	{

	}

	void PVTKOpenGLWidget::prepareRenderingContex()
	{
		PSimulationThread::instance()->startRendering();

		SceneGraph::getInstance().draw();
		m_OpenGLWidget->GetRenderWindow()->Render();

		PSimulationThread::instance()->stopRendering();
	}

	void PVTKOpenGLWidget::redisplay()
	{
		m_OpenGLWidget->GetRenderWindow()->Render();
	}

}