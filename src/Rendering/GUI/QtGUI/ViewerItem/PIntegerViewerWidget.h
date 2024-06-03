#pragma once
#include "PDataViewerWidget.h"

namespace dyno
{
	class PIntegerViewerWidget : public PDataViewerWidget
	{
		Q_OBJECT

	public:

		PIntegerViewerWidget(FBase* field,QWidget* pParent = NULL);

	public slots:

		void updateDataTable()override;

		void buildArrayDataTable(int first, int last)override;

		void buildArrayListDataTable(int first, int last)override ;

		void buildVarDataTable()override;


	protected:
		FBase* mfield;


		
	};



}