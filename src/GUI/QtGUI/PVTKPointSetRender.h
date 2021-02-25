#pragma once

#include "Framework/ModuleVisual.h"

class vtkActor;
class vtkPolyDataMapper;
class PVTKPointSetSource;

namespace dyno
{
	class PVTKPointSetRender : public VisualModule
	{
		DECLARE_CLASS(PVTKPointSetRender)
	public:
		PVTKPointSetRender();
		virtual ~PVTKPointSetRender();


		vtkActor* getVTKActor();

	protected:
		bool  initializeImpl() override;

		void updateRenderingContext() override;

	private:
		vtkActor* m_actor;
		vtkPolyDataMapper* mapper;
		PVTKPointSetSource* pointsetSource;
	};

}