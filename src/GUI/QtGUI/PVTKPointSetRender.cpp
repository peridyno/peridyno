#pragma once
#include "PVTKPointSetRender.h"

#include "PVTKPolyDataSource.h"

#include "Framework/Node.h"
#include "PVTKOpenGLWidget.h"
#include "PVTKPointSetSource.h"

//VTK
#include <vtkActor.h>
#include <vtkPolyData.h>
#include <vtkRenderer.h>
#include <vtkSphereSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkGlyph3DMapper.h>
#include <vtkNamedColors.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>

namespace dyno
{
	IMPLEMENT_CLASS(PVTKPointSetRender)

	PVTKPointSetRender::PVTKPointSetRender()
		: VisualModule()
		, m_actor(nullptr)
		, mapper(nullptr)
		, pointsetSource(nullptr)
	{
	}

	PVTKPointSetRender::~PVTKPointSetRender()
	{
		if (m_actor != nullptr)
		{
			PVTKOpenGLWidget::getCurrentRenderer()->RemoveActor(m_actor);
			PVTKOpenGLWidget::getCurrentRenderer()->GetRenderWindow()->Render();
		}
	}

	vtkActor* PVTKPointSetRender::getVTKActor()
	{
		return m_actor;
	}

	bool PVTKPointSetRender::initializeImpl()
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Should insert this module into a node!");
			return false;
		}

		auto triSet = TypeInfo::cast<PointSet<DataType3f>>(parent->getTopologyModule());
		if (triSet == nullptr)
		{
			Log::sendMessage(Log::Error, "TriangleModule: The topology module is not supported!");
			return false;
		}

		auto colors = vtkSmartPointer<vtkNamedColors>::New();

		auto points = vtkSmartPointer<vtkPoints>::New();
		points->InsertNextPoint(0, 0, 0);
		points->InsertNextPoint(1, 0, 0);
		points->InsertNextPoint(1, 1, 0);
		points->InsertNextPoint(0, 1, 0);
		points->InsertNextPoint(0, 0, 1);
		points->InsertNextPoint(1, 0, 1);
		points->InsertNextPoint(1, 1, 1);
		points->InsertNextPoint(0, 1, 1);
		points->InsertNextPoint(0.5, 0, 0);
		points->InsertNextPoint(1, 0.5, 0);
		points->InsertNextPoint(0.5, 1, 0);
		points->InsertNextPoint(0, 0.5, 0);
		points->InsertNextPoint(0.5, 0.5, 0);

		auto sphere = vtkSmartPointer<vtkSphereSource>::New();
		sphere->SetPhiResolution(21);
		sphere->SetThetaResolution(21);
		sphere->SetRadius(.0025);


		pointsetSource = PVTKPointSetSource::New();
		//polydataSource->SetInputData(spheredata);
		pointsetSource->setData(triSet);

		pointsetSource->Update();

		vtkPointSet* polydata = pointsetSource->GetOutput();

// 		// vtkPolyData
// 		auto polyData = vtkSmartPointer<vtkPolyData>::New();
// 		polyData->SetPoints(points);

		// vtkGlyph3DMapper: Rendering each vertex as a sphere
		auto pointMapper = vtkSmartPointer<vtkGlyph3DMapper>::New();
		pointMapper->SetInputData(polydata);
		pointMapper->SetSourceConnection(sphere->GetOutputPort());

		m_actor = vtkActor::New();
		m_actor->SetMapper(pointMapper);
		m_actor->GetProperty()->SetColor(colors->GetColor3d("Peacock").GetData());


		PVTKOpenGLWidget::getCurrentRenderer()->AddActor(m_actor);

		return true;
	}


	void PVTKPointSetRender::updateRenderingContext()
	{
		pointsetSource->Update();
		pointsetSource->Modified();

		m_actor->SetVisibility(isVisible());
	}
}