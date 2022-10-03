#include "PLogWidget.h"

#include "Platform.h"

#include <QHeaderView>

#include <QString>
#include <QTime>
#include <QApplication>

namespace dyno
{
	PLogSignal PLogWidget::logSignal;

	QTimeTableWidgetItem::QTimeTableWidgetItem(void) :
		QTableWidgetItem(QTime::currentTime().toString("hh:mm:ss"))
	{
		setFont(QFont("Arial", 8));
		//	setTextColor(QColor(60, 60, 60));

		setToolTip(text());
		setStatusTip("Message recorded at " + text());
	}

	QSize QTimeTableWidgetItem::sizeHint(void) const
	{
		return QSize(70, 10);
	}

	PTableItemMessage::PTableItemMessage(const Log::Message& m) :
		QTableWidgetItem(QString::fromUtf8(m.text.c_str()))
	{
		QString ToolTipPrefix;

		if (m.type & Log::Error)
			ToolTipPrefix += "Critical error: ";

		setFont(QFont("Arial", 8));

		QColor TextColor;

		switch (m.type)
		{
		case Log::Warning:
		{
			//TextColor = Qt::black;
			break;
		}

		case Log::Error:
		{
			TextColor = Qt::red;
			setTextColor(TextColor);
			break;
		}
		}

		QString text = QString::fromUtf8(m.text.c_str());

		setToolTip(ToolTipPrefix + text);
		setStatusTip(ToolTipPrefix + text);
	}

	PTableItemProgress::PTableItemProgress(const QString& Event, const float& Progress)
	{
		QString ProgressString = Event;

		if (Progress == 100.0f)
			ProgressString += "... Done";
		else
			ProgressString += QString::number(Progress, 'f', 2);

		setText(ProgressString);
		setFont(QFont("Arial", 7));
		//	setTextColor(Qt::blue);
	}

	PLogWidget::PLogWidget(QWidget* pParent /*= NULL*/) :
		QTableWidget(pParent)
	{
		Log::setUserReceiver(PLogWidget::RecieveLogMessage);

		setColumnCount(3);

		QStringList HeaderLabels;

		HeaderLabels << "time" << "" << "message";

		setHorizontalHeaderLabels(HeaderLabels);
		horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
		horizontalHeader()->setSectionResizeMode(1, QHeaderView::Fixed);
		horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
		horizontalHeader()->resizeSection(1, 25);
		horizontalHeader()->setDefaultAlignment(Qt::AlignLeft);
		horizontalHeader()->setVisible(false);

		// Disable vertical header
		verticalHeader()->setVisible(false);

		setGridStyle(Qt::NoPen);

		setAlternatingRowColors(true);

		QObject::connect(&PLogWidget::logSignal, SIGNAL(sendMessage(const Log::Message&)), this, SLOT(OnLog(const Log::Message&)));

		Log::sendMessage(Log::Info, "Finished");
	}

	void PLogWidget::setOutput(std::string filename)
	{
		Log::setOutput(filename);
	}

	void PLogWidget::toggleLogging()
	{
		mEnableLogging = mEnableLogging ? false : true;
	}

	void PLogWidget::OnLog(const Log::Message& m)
	{
		if (!mEnableLogging)
			return;

		insertRow(0);

		QIcon ItemIcon;

		switch (m.type)
		{
		case (int)Log::Warning:
		{
			ItemIcon = getIcon("exclamation");
			break;
		}

		case (int)Log::Error:
		{
			ItemIcon = getIcon("exclamation-red");
			break;
		}

		case (int)Log::Info:
			ItemIcon = getIcon("exclamation-white");
			break;

		case (int)Log::User:
			ItemIcon = getIcon("user");
			break;
		}

		setItem(0, 0, new QTimeTableWidgetItem());
		setItem(0, 1, new QTableWidgetItem(ItemIcon, ""));
		setItem(0, 2, new PTableItemMessage(m));
		setRowHeight(0, 18);
	}
// 
// 	void PLogWidget::OnLog(const QString& Message, const QString& Icon)
// 	{
// 		insertRow(0);
// 
// 		QIcon ItemIcon = GetIcon(Icon);
// 
// 		setItem(0, 0, new QTimeTableWidgetItem());
// 		setItem(0, 1, new QTableWidgetItem(ItemIcon, ""));
// 		setItem(0, 2, new QTableItemMessage(Message, QLogger::Normal));
// 		setRowHeight(0, 18);
// 	}
// 
// 	void PLogWidget::OnLogProgress(const QString& Event, const float& Progress)
// 	{
// 		// Find nearest row with matching event
// 		QList<QTableWidgetItem*> Items = findItems(Event, Qt::MatchStartsWith);
// 
// 		int RowIndex = 0;
// 
// 		if (Items.empty())
// 		{
// 			insertRow(0);
// 			RowIndex = 0;
// 		}
// 		else
// 		{
// 			RowIndex = Items[0]->row();
// 		}
// 
// 		setItem(RowIndex, 0, new QTimeTableWidgetItem());
// 		setItem(RowIndex, 1, new QTableWidgetItem(""));
// 		setItem(RowIndex, 2, new QTableItemProgress(Event, Progress));
// 		setRowHeight(0, 18);
// 	}

	void PLogWidget::OnClear(void)
	{
		if (currentRow() < 0)
			return;

		removeRow(currentRow());
	}

	void PLogWidget::OnClearAll(void)
	{
		clear();
		setRowCount(0);
	}

	void PLogWidget::RecieveLogMessage(const Log::Message& m)
	{
		PLogWidget::logSignal.setMessage(m);
	}

	void PLogWidget::contextMenuEvent(QContextMenuEvent* pContextMenuEvent)
	{
// 		QMenu ContextMenu(this);
// 		ContextMenu.setTitle("Log");
// 
// 		if (currentRow() > 0)
// 			ContextMenu.addAction(GetIcon("cross-small"), "Clear", this, SLOT(OnClear()));
// 
// 		ContextMenu.addAction(GetIcon("cross"), "Clear All", this, SLOT(OnClearAll()));
// 		ContextMenu.exec(pContextMenuEvent->globalPos());
	}

	QIcon PLogWidget::getIcon(const QString& name)
	{
		return QIcon(QString::fromLocal8Bit(getAssetPath().c_str()) + "icon/" + name + ".png");
	}

	QSize PLogWidget::sizeHint() const
	{
		return QSize(100, 100);
	}

	void PLogSignal::setMessage(const Log::Message& message)
	{
		emit sendMessage(message);
	}

}