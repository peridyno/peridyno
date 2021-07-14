#pragma once

#include "Module/VisualModule.h"

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

		void updateGraphicsContext() override;

	private:
		vtkActor* m_actor;
		vtkPolyDataMapper* mapper;
		PVTKPointSetSource* pointsetSource;
	};

}