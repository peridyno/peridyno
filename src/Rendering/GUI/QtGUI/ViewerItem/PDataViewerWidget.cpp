#include "PDataViewerWidget.h"

#include <QContextMenuEvent>
#include <QHeaderView>
#include "Format.h"
#include "PSimulationThread.h"
#include "qprogressbar.h"

namespace dyno
{

	PDataViewerWidget::PDataViewerWidget(FBase* field, QWidget* pParent):
		QTableWidget(pParent)
	{
		mfield = field;
		rowNum = mfield->size();
		printf("Data Size = %d\n",rowNum);

		this->setEditTriggers(QAbstractItemView::NoEditTriggers);
		this->setAttribute(Qt::WA_DeleteOnClose,true);
		//this->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

		Qt::WindowFlags m_flags = windowFlags();
		setWindowFlags(m_flags | Qt::WindowStaysOnTopHint);
		
		this->setWindowTitle(FormatFieldWidgetName(field->getObjectName()) + " ( " + FormatFieldWidgetName(field->getTemplateName()) + " )");

		horizontalHeader()->setDefaultAlignment(Qt::AlignLeft);
		verticalHeader()->setVisible(true);

		setGridStyle(Qt::SolidLine);

		setAlternatingRowColors(true);

		disconnect(this, 0, 0, 0);

		connect(PSimulationThread::instance(), &PSimulationThread::oneFrameFinished, this, &PDataViewerWidget::updateDataTable);


		if (mfield->getClassName() == std::string("FArrayList")) 
			mTemplateType = TemplateName::FArrayListType;

		if (mfield->getClassName() == std::string("FArray"))
			mTemplateType = TemplateName::FArrayType;

		if (mfield->getClassName() == std::string("FVar"))
			mTemplateType = TemplateName::FVarType;


		switch (mTemplateType)
		{
		default:
			break;
		case TemplateName::FArrayListType:
			this->setMinimumWidth(500);
		}
		


		this->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff); 

		updateMaxViewRowCount();
	}
	
	void PDataViewerWidget::updateDataTable()
	{
		//printf("updateDataTable\n");
		this->ClearDataTable();

		int VisibleRowNum = mMaxViewRowCount > rowNum ? rowNum : mMaxViewRowCount;
		this->setRowCount(VisibleRowNum);

		if (mScrollBar == NULL)
			return;

		
		this->minIndex = mScrollBar->value();
		this->maxIndex = (minIndex + mMaxViewRowCount < rowNum ? minIndex + mMaxViewRowCount : rowNum) - 1;

		printf("mScrollBar:: min: %d -  max: %d \n", int(mScrollBar->minimum()), int(mScrollBar->maximum()));
		printf("minIndex: %d\n", minIndex);
		printf("maxIndex: %d\n", maxIndex);
		printf("maxViewRow: %d\n", mMaxViewRowCount);
		printf("size : %d\n", mfield->size());

		if (minIndex == -1 || maxIndex == -1)
			return;

		switch (mTemplateType)
		{
		default:
			break;

		case TemplateName::FArrayListType:
			buildArrayListDataTable(minIndex,maxIndex);
			break;
		case TemplateName::FArrayType:
			buildArrayDataTable(minIndex, maxIndex);
			break;
		case TemplateName::FVarType:
			buildVarDataTable();
			break;
		}
		




		for (size_t i = 0; i < rowCount(); i++)
		{
			setRowHeight(i,rowsHeight);
		}




	}

	void PDataViewerWidget::updateScrollBarParam()
	{

		if (mScrollBar != NULL)
		{

			mScrollBar->setRange(0, rowNum - mMaxViewRowCount);
			mScrollBar->setPageStep(mMaxViewRowCount);


			this->minIndex = mScrollBar->value();
			this->maxIndex = (minIndex + mMaxViewRowCount < rowNum ? minIndex + mMaxViewRowCount : rowNum) - 1;

		}


	}
	

	void PDataViewerWidget::resizeEvent(QResizeEvent* event)
	{

		QTableView::resizeEvent(event);

		updateMaxViewRowCount();

		updateScrollBarParam();

		updateDataTable();

	}


	void PDataViewerWidget::updateMaxViewRowCount()
	{
		rowsHeight;
		int t_tableH = this->height();
		mMaxViewRowCount = std::clamp(t_tableH / rowsHeight, 1, 100);
	}

	void PDataViewerWidget::wheelEvent(QWheelEvent* wheelEvent)
	{
		if (AltPressed) 
			this->horizontalScrollBar()->setValue(horizontalScrollBar()->value() - (wheelEvent->angleDelta().x()/abs(wheelEvent->angleDelta().x())));
		else
			emit wheelDeltaAngleChange(wheelEvent->angleDelta().x());

	}

	void PDataViewerWidget::addItemToPosition(const QString& qstr, int row, int column,const QString& rowName,const QString& columnName, const bool resizeRow, const int height)
	{

		{
			QTableWidgetItem* header = new QTableWidgetItem;
			header->setText(rowName);
			this->setVerticalHeaderItem(row, header);

			if (resizeRow)
				this->setRowHeight(row,height);
		}

		int curColumn = this->columnCount();
		if (column + 1 > curColumn)
		{
			this->setColumnCount(column + 1);

			QTableWidgetItem* header = new QTableWidgetItem;
			header->setText(columnName);
			this->setHorizontalHeaderItem(column, header);
		}

		auto item = new QTableWidgetItem;
		item->setText(qstr);

		this->setItem(row, column, item);

		
	}



	void PDataViewerWidget::ClearDataTable()
	{
		int rowNums = this->rowCount();
		for (int i = 0; i < rowNums; i++)
		{
			this->removeRow(0);
		}
	}

	int PDataViewerWidget::sizeHintForColumn(int column) const 
	{
		ensurePolished();

		return QTableWidget::sizeHintForColumn(column);
	}






}