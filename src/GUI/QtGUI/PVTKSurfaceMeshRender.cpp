#pragma once
#include "PVTKSurfaceMeshRender.h"

#include "PVTKPolyDataSource.h"

#include "Framework/Node.h"
#include "PVTKOpenGLWidget.h"

//VTK
#include <vtkActor.h>
#include <vtkPolyData.h>
#include <vtkRenderer.h>
#include <vtkSphereSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>

namespace dyno
{
	IMPLEMENT_CLASS(PVTKSurfaceMeshRender)

	PVTKSurfaceMeshRender::PVTKSurfaceMeshRender()
		: VisualModule()
		, m_actor(nullptr)
		, mapper(nullptr)
		, polydataSource(nullptr)
	{
	}

	PVTKSurfaceMeshRender::~PVTKSurfaceMeshRender()
	{
		if (m_actor != nullptr)
		{
			PVTKOpenGLWidget::getCurrentRenderer()->RemoveActor(m_actor);
			PVTKOpenGLWidget::getCurrentRenderer()->GetRenderWindow()->Render();
		}
	}

	vtkActor* PVTKSurfaceMeshRender::getVTKActor()
	{
		return m_actor;
	}

	bool PVTKSurfaceMeshRender::initializeImpl()
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Should insert this module into a node!");
			return false;
		}

		auto triSet = TypeInfo::cast<TriangleSet<DataType3f>>(parent->getTopologyModule());
		if (triSet == nullptr)
		{
			Log::sendMessage(Log::Error, "TriangleModule: The topology module is not supported!");
			return false;
		}

		polydataSource = PVTKPolyDataSource::New();
		//polydataSource->SetInputData(spheredata);
		polydataSource->setData(triSet);
		
		polydataSource->Update();

		vtkPolyData* polydata = polydataSource->GetOutput();

		// Create a mapper
		mapper = vtkPolyDataMapper::New();
		mapper->SetInputData(polydata);

		// Create an actor
		m_actor = vtkActor::New();
		m_actor->SetMapper(mapper);


		polydataSource->Modified();

		PVTKOpenGLWidget::getCurrentRenderer()->AddActor(m_actor);
		this->setVisible(true);

		return true;
	}


	void PVTKSurfaceMeshRender::updateRenderingContext()
	{
		polydataSource->Update();
		polydataSource->Modified();

		m_actor->SetVisibility(isVisible());
	}
}