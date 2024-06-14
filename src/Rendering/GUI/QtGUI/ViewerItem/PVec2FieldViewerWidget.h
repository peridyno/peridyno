#pragma once
#include "PDataViewerWidget.h"

namespace dyno
{
	class PVec2FieldViewerWidget : public PDataViewerWidget
	{
		Q_OBJECT

	public:

		PVec2FieldViewerWidget(FBase* field,QWidget* pParent = NULL);

	public slots:

		void updateDataTable()override;

		void buildArrayDataTable(int first, int last)override;

		void buildArrayListDataTable(int first, int last)override;

		void buildVarDataTable()override;

	protected:
		FBase* mfield;


		
	};



}