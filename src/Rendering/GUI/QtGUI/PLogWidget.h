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

namespace dyno
{
	/**
	 * @brief QLogSignal is used to send message from Log to QT PLogWidget.
	 * 
	 */
	class PLogSignal : public QObject
	{
		Q_OBJECT

	public:
		void setMessage(const Log::Message& message);

	signals:
		void sendMessage(const Log::Message& message);
	};

	class QTimeTableWidgetItem : public QTableWidgetItem
	{
	public:
		QTimeTableWidgetItem(void);

		virtual QSize sizeHint() const;
	};

	class PTableItemMessage : public QTableWidgetItem
	{
	public:
		PTableItemMessage(const Log::Message& m);
	};

	class PTableItemProgress : public QTableWidgetItem
	{
	public:
		PTableItemProgress(const QString& Event, const float& Progress);
	};

	class PLogWidget : public QTableWidget
	{
		Q_OBJECT

	public:
		PLogWidget(QWidget* pParent = NULL);

		static PLogSignal logSignal;
		static void RecieveLogMessage(const Log::Message& m);

		QSize sizeHint() const override;

		static void setOutput(std::string filename);

		void toggleLogging();

	protected:
		void contextMenuEvent(QContextMenuEvent* pContextMenuEvent);

		QIcon getIcon(const QString& name);

	public slots:
		void OnLog(const Log::Message& m);
// 		void OnLog(const QString& Message, const QString& Icon);
// 		void OnLogProgress(const QString& Event, const float& Progress);
		void OnClear(void);
		void OnClearAll(void);

	private:
		bool mEnableLogging = false;
	};

}