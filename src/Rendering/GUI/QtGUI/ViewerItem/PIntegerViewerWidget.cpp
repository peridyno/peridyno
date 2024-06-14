#include "PIntegerViewerWidget.h"
#include <QHeaderView>


namespace dyno
{

	PIntegerViewerWidget::PIntegerViewerWidget(FBase* field, QWidget* pParent) :
		PDataViewerWidget(field, pParent)
	{
		mfield = field;

		rowsHeight = 30;

		updateDataTable();	
	}
	

	void PIntegerViewerWidget::updateDataTable()
	{
		PDataViewerWidget::updateDataTable();
	}

	void PIntegerViewerWidget::buildArrayDataTable(int first, int last)
	{

		// ***************************    int	 ************************
		{
			CArray<int>* dataPtr = NULL;
			CArray<int> cData;

			// ************************		 GPU	   **************************
			FArray<int, DeviceType::GPU>* int_GPU = TypeInfo::cast<FArray<int, DeviceType::GPU>>(mfield);
			if (int_GPU != nullptr)
			{
				std::shared_ptr<Array<int, DeviceType::GPU>>& data = int_GPU->getDataPtr();

				cData.assign(*data);
				dataPtr = &cData;
			}

			// ************************	    CPU	   **************************
			FArray<int, DeviceType::CPU>* int_CPU = TypeInfo::cast<FArray<int, DeviceType::CPU>>(mfield);
			if (int_CPU != nullptr)
			{
				dataPtr = int_CPU->getDataPtr().get();
			}

			if (dataPtr != NULL) 
			{
				uint rowId = first;
				uint rowTarget = 0;
				for (size_t i = first; i <= last; i++)
				{
					QString value = "";
					value = (QString::number((*dataPtr)[i]));

					addItemToPosition(value, rowTarget,0, QString::number(rowId), QString("int"));

					QTableWidgetItem* header = new QTableWidgetItem;
					header->setText(QString::number(i));
					this->setVerticalHeaderItem(i, header);
					rowId++;
					rowTarget++;
				}

			}		
		}

		// ***************************    uint	 ************************
		{
			CArray<uint>* dataPtr = NULL;
			CArray<uint> cData;

			// ************************		 GPU	   **************************
			FArray<uint, DeviceType::GPU>* uint_GPU = TypeInfo::cast<FArray<uint, DeviceType::GPU>>(mfield);
			if (uint_GPU != nullptr)
			{
				std::shared_ptr<Array<uint, DeviceType::GPU>>& data = uint_GPU->getDataPtr();

				cData.assign(*data);
				dataPtr = &cData;
			}

			// ************************	    CPU	   **************************
			FArray<uint, DeviceType::CPU>* uint_CPU = TypeInfo::cast<FArray<uint, DeviceType::CPU>>(mfield);
			if (uint_CPU != nullptr)
			{
				dataPtr = uint_CPU->getDataPtr().get();
			}

			if (dataPtr != NULL)
			{
				uint rowId = first;
				uint rowTarget = 0;
				for (size_t i = first; i <= last; i++)
				{

					QString value = "";
					value = (QString::number((*dataPtr)[i]));

					addItemToPosition(value, rowTarget, 0, QString::number(rowId), QString("uint"));

					QTableWidgetItem* header = new QTableWidgetItem;
					header->setText(QString::number(i));
					this->setVerticalHeaderItem(i, header);

					rowId++;
					rowTarget++;
				}

			}
		}

		
	}

	void PIntegerViewerWidget::buildVarDataTable()
	{

		// ************************		FVar<Vec3f>	   **************************
		{
			FVar<int>* f = TypeInfo::cast<FVar<int>>(mfield);
			if (f != nullptr)
			{
				const auto& data = f->getValue();

				QString value = "";
				value = (QString::number(data));

				addItemToPosition(value, 0, 0, QString::number(0), QString("int"));
				
				QTableWidgetItem* header = new QTableWidgetItem;
				header->setText(QString::number(0));
				this->setVerticalHeaderItem(0, header);

			}
		}

		// ************************		FVar<Vec3d>	   **************************
		{
			FVar<uint>* f = TypeInfo::cast<FVar<uint>>(mfield);
			if (f != nullptr)
			{
				const auto& data = f->getValue();


				QString value = "";
				value = (QString::number(data));

				addItemToPosition(value, 0, 0, QString::number(0), QString("uint"));


				QTableWidgetItem* header = new QTableWidgetItem;
				header->setText(QString::number(0));
				this->setVerticalHeaderItem(0, header);

			}
		}



	}

	void PIntegerViewerWidget::buildArrayListDataTable(int first, int last)
	{
		{
			// ***************************    int	 ************************
			CArrayList<int>* dataPtr = NULL;
			CArrayList<int> cData;

			// ************************	    GPU	   **************************
			FArrayList<int, DeviceType::GPU>* f_GPU = TypeInfo::cast<FArrayList<int, DeviceType::GPU>>(mfield);
			if (f_GPU != nullptr)
			{
				std::shared_ptr<ArrayList<int, DeviceType::GPU>>& data = f_GPU->getDataPtr();

				cData.assign(*data);
				dataPtr = &cData;
			}

			// ************************	    CPU	   **************************
			FArrayList<int, DeviceType::CPU>* f_CPU = TypeInfo::cast<FArrayList<int, DeviceType::CPU>>(mfield);
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

					value.append(QString::number(trans));

					addItemToPosition(value, rowTarget, j, QString::number(rowId), QString("[") + QString::number(j) + QString("]") , true, rowsHeight);

					value.clear();
				}
				rowId++;
				rowTarget++;
			}

		}



		{

			// ***************************    uint	 ************************
			CArrayList<uint>* dataPtr = NULL;
			CArrayList<uint> cData;

			// ************************	    GPU	   **************************
			FArrayList<uint, DeviceType::GPU>* f_GPU = TypeInfo::cast<FArrayList<uint, DeviceType::GPU>>(mfield);
			if (f_GPU != nullptr)
			{
				std::shared_ptr<ArrayList<uint, DeviceType::GPU>>& data = f_GPU->getDataPtr();

				cData.assign(*data);
				dataPtr = &cData;
			}

			// ************************	    CPU	   **************************
			FArrayList<uint, DeviceType::CPU>* f_CPU = TypeInfo::cast<FArrayList<uint, DeviceType::CPU>>(mfield);
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

					value.append(QString::number(trans));

					addItemToPosition(value, rowTarget, j, QString::number(rowId), QString("[") + QString::number(j) + QString("]"), true, rowsHeight);

					value.clear();
				}
				rowId++;
				rowTarget++;
			}

		}

	}


}