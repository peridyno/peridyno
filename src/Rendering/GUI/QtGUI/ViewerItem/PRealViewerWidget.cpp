#include "PRealViewerWidget.h"
#include <QHeaderView>


namespace dyno
{

	PRealViewerWidget::PRealViewerWidget(FBase* field, QWidget* pParent) :
		PDataViewerWidget(field, pParent)
	{
		mfield = field;

		rowsHeight = 30;

		updateDataTable();	
	}
	

	void PRealViewerWidget::updateDataTable()
	{
		PDataViewerWidget::updateDataTable();
	}

	void PRealViewerWidget::buildArrayDataTable(int first, int last)
	{

		// ***************************    float	 ************************
		{
			CArray<float>* dataPtr = NULL;
			CArray<float> cData;

			// ************************		 GPU	   **************************
			FArray<float, DeviceType::GPU>* float_GPU = TypeInfo::cast<FArray<float, DeviceType::GPU>>(mfield);
			if (float_GPU != nullptr)
			{
				std::shared_ptr<Array<float, DeviceType::GPU>>& data = float_GPU->getDataPtr();

				cData.assign(*data);
				dataPtr = &cData;
			}

			// ************************	    CPU	   **************************
			FArray<float, DeviceType::CPU>* float_CPU = TypeInfo::cast<FArray<float, DeviceType::CPU>>(mfield);
			if (float_CPU != nullptr)
			{
				dataPtr = float_CPU->getDataPtr().get();
			}

			if (dataPtr != NULL) 
			{
				uint rowId = first;
				uint rowTarget = 0;
				for (size_t i = first; i <= last; i++)
				{
					QString value = "";
					value = (QString::number((*dataPtr)[i], 'f', 6));

					addItemToPosition(value, rowTarget,0, QString::number(rowId), QString("float"));

					QTableWidgetItem* header = new QTableWidgetItem;
					header->setText(QString::number(i));
					this->setVerticalHeaderItem(i, header);
					rowId++;
					rowTarget++;
				}
			}		
		}

		// ***************************    double	 ************************
		{
			CArray<double>* dataPtr = NULL;
			CArray<double> cData;

			// ************************		 GPU	   **************************
			FArray<double, DeviceType::GPU>* double_GPU = TypeInfo::cast<FArray<double, DeviceType::GPU>>(mfield);
			if (double_GPU != nullptr)
			{
				std::shared_ptr<Array<double, DeviceType::GPU>>& data = double_GPU->getDataPtr();

				cData.assign(*data);
				dataPtr = &cData;
			}

			// ************************	    CPU	   **************************
			FArray<double, DeviceType::CPU>* double_CPU = TypeInfo::cast<FArray<double, DeviceType::CPU>>(mfield);
			if (double_CPU != nullptr)
			{
				dataPtr = double_CPU->getDataPtr().get();
			}

			if (dataPtr != NULL)
			{
				uint rowId = first;
				uint rowTarget = 0;
				for (size_t i = first; i <= last; i++)
				{

					QString value = "";
					value = (QString::number((*dataPtr)[i], 'f', 10));

					addItemToPosition(value, rowTarget, 0, QString::number(rowId), QString("double"));

					QTableWidgetItem* header = new QTableWidgetItem;
					header->setText(QString::number(i));
					this->setVerticalHeaderItem(i, header);

					rowId++;
					rowTarget++;
				}
			}
		}		
	}


	void PRealViewerWidget::buildVarDataTable()
	{

		// ************************		FVar<float>	   **************************
		{
			FVar<float>* f = TypeInfo::cast<FVar<float>>(mfield);
			if (f != nullptr)
			{
				const auto& data = f->getValue();

				QString value = "";
				value = (QString::number(data, 'f', 6));

				addItemToPosition(value, 0, 0, QString::number(0), QString("float"));
				
				QTableWidgetItem* header = new QTableWidgetItem;
				header->setText(QString::number(0));
				this->setVerticalHeaderItem(0, header);

			}
		}

		// ************************		FVar<double>	   **************************
		{
			FVar<double>* f = TypeInfo::cast<FVar<double>>(mfield);
			if (f != nullptr)
			{
				const auto& data = f->getValue();


				QString value = "";
				value = (QString::number(data, 'f', 10));

				addItemToPosition(value, 0, 0, QString::number(0), QString("double"));


				QTableWidgetItem* header = new QTableWidgetItem;
				header->setText(QString::number(0));
				this->setVerticalHeaderItem(0, header);

			}
		}



	}

	void PRealViewerWidget::buildArrayListDataTable(int first, int last)
	{
		{
			// ***************************    float	 ************************
			CArrayList<float>* dataPtr = NULL;
			CArrayList<float> cData;

			// ************************	    GPU	   **************************
			FArrayList<float, DeviceType::GPU>* f_GPU = TypeInfo::cast<FArrayList<float, DeviceType::GPU>>(mfield);
			if (f_GPU != nullptr)
			{
				std::shared_ptr<ArrayList<float, DeviceType::GPU>>& data = f_GPU->getDataPtr();

				cData.assign(*data);
				dataPtr = &cData;
			}

			// ************************	    CPU	   **************************
			FArrayList<float, DeviceType::CPU>* f_CPU = TypeInfo::cast<FArrayList<float, DeviceType::CPU>>(mfield);
			if (f_CPU != nullptr)
			{
				dataPtr = f_CPU->getDataPtr().get();
			}


			if (dataPtr == NULL)
				return;

			//BuildDataTable
			uint rowId = first;
			uint rowTarget = 0;
			for (size_t i = first; i <= last; i++)
			{
				auto it = (*dataPtr)[i];

				for (size_t j = 0; j < it.size(); j++)
				{
					auto trans = it[j];

					QString value;

					value.append(QString::number(trans, 'f', 6));

					addItemToPosition(value, rowTarget, j, QString::number(rowId), QString("[") + QString::number(j) + QString("]") , true, rowsHeight);

					value.clear();
				}
				rowId++;
				rowTarget++;
			}

		}



		{

			// ***************************    double	 ************************
			CArrayList<double>* dataPtr = NULL;
			CArrayList<double> cData;

			// ************************	    GPU	   **************************
			FArrayList<double, DeviceType::GPU>* f_GPU = TypeInfo::cast<FArrayList<double, DeviceType::GPU>>(mfield);
			if (f_GPU != nullptr)
			{
				std::shared_ptr<ArrayList<double, DeviceType::GPU>>& data = f_GPU->getDataPtr();

				cData.assign(*data);
				dataPtr = &cData;
			}

			// ************************	    CPU	   **************************
			FArrayList<double, DeviceType::CPU>* f_CPU = TypeInfo::cast<FArrayList<double, DeviceType::CPU>>(mfield);
			if (f_CPU != nullptr)
			{
				dataPtr = f_CPU->getDataPtr().get();
			}


			if (dataPtr == NULL)
				return;

			//BuildDataTable
			uint rowId = first;
			uint rowTarget = 0;
			for (size_t i = first; i <= last; i++)
			{
				auto it = (*dataPtr)[i];

				for (size_t j = 0; j < it.size(); j++)
				{
					auto trans = it[j];

					QString value;

					value.append(QString::number(trans, 'f', 10));

					addItemToPosition(value, rowTarget, j, QString::number(rowId), QString("[") + QString::number(j) + QString("]"), true, rowsHeight);

					value.clear();
				}
				rowId++;
				rowTarget++;
			}

		}

	}


}