#include "PLogWidget.h"

#include "Platform.h"

#include <QHeaderView>

#include <QMenu>
#include <QContextMenuEvent>
#include <QString>
#include <QTime>
#include <QApplication>

namespace dyno
{
	std::atomic<PLogWidget*> PLogWidget::gInstance;
	std::mutex PLogWidget::gMutex;

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
		return QTableWidgetItem::sizeHint();
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
			//setTextColor(TextColor);
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

	PLogWidget* PLogWidget::instance()
	{
		PLogWidget* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new PLogWidget();
				gInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
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

	void PLogWidget::onPrintMessage(const Log::Message& m)
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

	void PLogWidget::onClear(void)
	{
		if (currentRow() < 0)
			return;

		removeRow(currentRow());
	}

	void PLogWidget::onClearAll(void)
	{
		clear();
		setRowCount(0);
	}

	void PLogWidget::RecieveLogMessage(const Log::Message& m)
	{
		PLogWidget::instance()->onPrintMessage(m);
	}

	void PLogWidget::contextMenuEvent(QContextMenuEvent* pContextMenuEvent)
	{
		QMenu menu;
		menu.setTitle("Log");

		menu.addAction(getIcon("cross"), "Clear All", this, SLOT(onClearAll()));
		menu.exec(pContextMenuEvent->globalPos());
	}

	QIcon PLogWidget::getIcon(const QString& name)
	{
		return QIcon(QString::fromLocal8Bit(getAssetPath().c_str()) + "icon/" + name + ".png");
	}

	QSize PLogWidget::sizeHint() const
	{
		return QSize(100, 100);
	}

	int PLogWidget::sizeHintForColumn(int column) const {
		ensurePolished();

		//TODO: viewOptions cannot not be recognized for Qt 6 
// 		int row_count = rowCount();
// 		if (row_count > 0 && column == 0) {
// 			auto idx = model()->index(0, 0);
// 			auto vo = viewOptions();
// 			auto hint = itemDelegate(idx)->sizeHint(vo, idx).width();
// 			return hint + 1;
// 		}

		return QTableWidget::sizeHintForColumn(column);
	}
}