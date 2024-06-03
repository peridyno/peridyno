#pragma once
#include <QTableWidgetItem>
#include "Field.h"
#include <QResizeEvent>
#include <QWheelEvent>
#include <QScrollBar>

#include <QScrollBar>
#include <QMouseEvent>
#include <QHBoxLayout>
#include <QKeyEvent>
#include "Format.h"

class QDataViewScrollBar : public QScrollBar
{
	Q_OBJECT
public:
	QDataViewScrollBar(Qt::Orientation, QWidget* parent = nullptr) : QScrollBar(parent) {}

	void setScrollBarParam(int min, int max, int step) 
	{
		this->setRange(min,max);
		this->setPageStep(step);
	}

public slots:

	void updateScrollValue(int v) 
	{
		int deltaValue = pageStep() / 3 >= 1 ? pageStep() / 3 : 1;
		if (v > 0)
			this->setValue(this->value() - deltaValue);
		else if (v < 0)
			this->setValue(this->value() + deltaValue);

	}

private:
	
	QPoint pressPos = QPoint(0,0);

};



namespace dyno
{

	enum TemplateName
	{
		FArrayType = 0,
		FArrayListType = 1,
		FVarType = 2,
		NoneType = 3,
	};

	enum DataType
	{
		FloatType = 0,
		DoubleType = 1,
		UnknownType = 2,
	};

	class PDataViewerWidget : public QTableWidget
	{
		Q_OBJECT

	public:

		PDataViewerWidget(FBase* field,QWidget* pParent = NULL);
		int sizeHintForColumn(int column) const override;
		int getRowNum() { return rowNum; }
		virtual void setRowNum(int size ) { rowNum = size ; }	
		void setTableScrollBar(QScrollBar* s) { mScrollBar = s; }
		QScrollBar* getTableScrollBar(QScrollBar* s) { return mScrollBar; }
		std::string getFieldWindowTitle() { return mfield->getObjectName() + " ( " + mfield->getTemplateName() + " )"; }

	protected:

		void addItemToPosition(const QString& qstr, int row, int column,const QString& rowName,const QString& columnName,const bool resizeRow = false,const int height = 30);
		void resizeEvent(QResizeEvent* event) override;
		void updateMaxViewRowCount();


	Q_SIGNALS:

		void wheelDeltaAngleChange(int value);

	public slots:

		void ClearDataTable();
		virtual void updateDataTable();
		virtual void updateDataTableTo(int value) { updateDataTable(); }
		void updateScrollBarParam();
		virtual void buildArrayListDataTable(int first, int last) {};
		virtual void buildArrayDataTable(int first, int last) {};
		virtual void buildVarDataTable() {};

	protected:

		void wheelEvent(QWheelEvent* wheelEvent) override;

		void keyPressEvent(QKeyEvent* event) override
		{
			if (event->key() == Qt::Key_Alt)
				AltPressed = true;

			QTableWidget::keyPressEvent(event);
		}

		void keyReleaseEvent(QKeyEvent* event)
		{
			if (event->key() == Qt::Key_Alt)
				AltPressed = false;

			QWidget::keyReleaseEvent(event);
		}




	protected:

		TemplateName mTemplateType = TemplateName::NoneType;
		DataType mDataType = DataType::UnknownType;
		int rowsHeight = 30;
		int mMaxViewRowCount = -1;
		int minIndex = -1;
		int maxIndex = -1;

	private:

		FBase* mfield;
		int rowNum = 0;
		QScrollBar* mScrollBar = NULL;

		bool AltPressed = false;
	};


}