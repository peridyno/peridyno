/*=========================================================================
  Program:   Log Widget
  Module:    PLogWidget.h

  Copyright (c) Xiaowei He
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
=========================================================================*/

#pragma once

#include "Log.h"

#include <QTableWidgetItem>

#include <atomic>
#include <mutex>

namespace dyno
{
	class PLogWidget : public QTableWidget
	{
		Q_OBJECT

	public:
		static PLogWidget* instance();

		QSize sizeHint() const override;
		int sizeHintForColumn(int column) const override;

		void toggleLogging();

		static void RecieveLogMessage(const Log::Message& m);
		static void setOutput(std::string filename);

	protected:
		void contextMenuEvent(QContextMenuEvent* pContextMenuEvent);

		QIcon getIcon(const QString& name);

	public slots:
		void onPrintMessage(const Log::Message& m);
		void onClear(void);
		void onClearAll(void);

	private:
		PLogWidget(QWidget* pParent = NULL);

		static std::atomic<PLogWidget*> gInstance;
		static std::mutex gMutex;

		bool mEnableLogging = false;
	};

}