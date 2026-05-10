#pragma once
#include "PDataViewerWidget.h"

namespace dyno
{
	class PInstanceViewerWidget : public QWidget
	{
		Q_OBJECT

	public:

		PInstanceViewerWidget(FBase* field, QWidget* pParent = NULL) { mfield = field; }

	public slots:

		virtual void updateWidget() {};

	protected:
		FBase* mfield;

	};



}