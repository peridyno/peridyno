#include "PVec2iFieldViewerWidget.h"
#include <QHeaderView>
#include "Vector/Vector3D.h"
#include "Vector/Vector2D.h"
#include "Framework/Module/TopologyModule.h"

namespace dyno
{

	PVec2FieldViewerWidget::PVec2FieldViewerWidget(FBase* field, QWidget* pParent) :
		PDataViewerWidget(field, pParent)
	{
		mfield = field;


		updateDataTable();

		
	}
	

	void PVec2FieldViewerWidget::updateDataTable()
	{
		PDataViewerWidget::updateDataTable();
	}


	void PVec2FieldViewerWidget::buildArrayDataTable(int first, int last)
	{


		// ************************		Vec2f      **************************
		{
			CArray<Vec2f>* cVec2f = NULL;
			CArray<Vec2f> cData;

			// ************************	    GPU	   **************************
			FArray<Vec2f, DeviceType::GPU>* f_GPU = TypeInfo::cast<FArray<Vec2f, DeviceType::GPU>>(mfield);
			if (f_GPU != nullptr)
			{
				std::shared_ptr<Array<Vec2f, DeviceType::GPU>>& data = f_GPU->getDataPtr();

				cData.assign(*data);
				cVec2f = &cData;			
			}

			// ************************	    CPU	   **************************
			FArray<Vec2f, DeviceType::CPU>* f_CPU = TypeInfo::cast<FArray<Vec2f, DeviceType::CPU>>(mfield);
			if (f_CPU != nullptr)
			{
				cVec2f = f_CPU->getDataPtr().get();
			}


			//BuildDataTable
			if (cVec2f != NULL)
			{
				uint rowId = first;
				uint rowTarget = 0;
				for (size_t i = first; i <= last; i++)
				{
					for (size_t j = 0; j < 2; j++)
					{
						QString value = "";
						value = (QString::number((*cVec2f)[i][j], 'f', 6));

						addItemToPosition(value, rowTarget, j, QString::number(rowId), QString("float") + QString("[") + QString::number(j) + QString("]"));
					}

					rowId++;
					rowTarget++;
				}
			}
			
		}


		// ************************		Vec2d      **************************
		{
			CArray<Vec2d>* cVec2d = NULL;
			CArray<Vec2d> cData;

			// ************************	    GPU	   **************************
			FArray<Vec2d, DeviceType::GPU>* f_GPU = TypeInfo::cast<FArray<Vec2d, DeviceType::GPU>>(mfield);
			if (f_GPU != nullptr)
			{
				std::shared_ptr<Array<Vec2d, DeviceType::GPU>>& data = f_GPU->getDataPtr();

				cData.assign(*data);
				cVec2d = &cData;
			}

			// ************************	    CPU	   **************************
			FArray<Vec2d, DeviceType::CPU>* f_CPU = TypeInfo::cast<FArray<Vec2d, DeviceType::CPU>>(mfield);
			if (f_CPU != nullptr)
			{
				cVec2d = f_CPU->getDataPtr().get();
			}


			//BuildDataTable
			if (cVec2d != NULL)
			{
				uint rowId = first;
				uint rowTarget = 0;
				for (size_t i = first; i <= last; i++) 
				{
					for (size_t j = 0; j < 2; j++)
					{
						QString value = "";
						value = (QString::number((*cVec2d)[i][j], 'f', 10));

						addItemToPosition(value, rowTarget, j, QString::number(rowId), QString("float") + QString("[") + QString::number(j) + QString("]"));
					}

					//QTableWidgetItem* header = new QTableWidgetItem;
					//header->setText(QString::number(i));
					//this->setVerticalHeaderItem(i, header);

					rowId++;
					rowTarget++;
				}

			}

		}

		// ************************		VectorND<int,2>      **************************
		{
			CArray<VectorND<int, 2>>* cVec2f = NULL;
			CArray<VectorND<int, 2>> cData;

			// ************************	    GPU	   **************************
			FArray<VectorND<int, 2>, DeviceType::GPU>* f_GPU = TypeInfo::cast<FArray<VectorND<int, 2>, DeviceType::GPU>>(mfield);
			if (f_GPU != nullptr)
			{
				std::shared_ptr<Array<VectorND<int, 2>, DeviceType::GPU>>& data = f_GPU->getDataPtr();

				cData.assign(*data);
				cVec2f = &cData;
			}

			// ************************	    CPU	   **************************
			FArray<VectorND<int, 2>, DeviceType::CPU>* f_CPU = TypeInfo::cast<FArray<VectorND<int, 2>, DeviceType::CPU>>(mfield);
			if (f_CPU != nullptr)
			{
				cVec2f = f_CPU->getDataPtr().get();
			}


			//BuildDataTable
			if (cVec2f != NULL)
			{
				uint rowId = first;
				uint rowTarget = 0;
				for (size_t i = first; i <= last; i++)
				{
					for (size_t j = 0; j < 2; j++)
					{
						QString value = "";
						value = (QString::number((*cVec2f)[i][j]));

						addItemToPosition(value, rowTarget, j, QString::number(rowId), QString("int") + QString("[") + QString::number(j) + QString("]"));
					}

					rowId++;
					rowTarget++;
				}
			}

		}

	}

	void PVec2FieldViewerWidget::buildVarDataTable()
	{

		//std::string template_name = mfield->getTemplateName();

		// ************************		FVar<Vec2f>	   **************************
		{
			FVar<Vec2f>* f = TypeInfo::cast<FVar<Vec2f>>(mfield);
			if (f != nullptr)
			{
				const auto& Vec2f = f->getValue();


				for (size_t j = 0; j < 2; j++)
				{
					QString value = "";
					value = (QString::number((Vec2f)[j], 'f', 6));

					addItemToPosition(value, 0, j, QString::number(0), QString("float") + QString("[") + QString::number(j) + QString("]"));
				}

				QTableWidgetItem* header = new QTableWidgetItem;
				header->setText(QString::number(0));
				this->setVerticalHeaderItem(0, header);	

			}
		}

		// ************************		FVar<Vec2d>	   **************************
		{
			FVar<Vec2d>* f = TypeInfo::cast<FVar<Vec2d>>(mfield);
			if (f != nullptr)
			{
				const auto& Vec2d = f->getValue();


				for (size_t j = 0; j < 2; j++)
				{
					QString value = "";
					value = (QString::number((Vec2d)[j], 'f', 10));

					addItemToPosition(value, 0, j, QString::number(0), QString("double") + QString("[") + QString::number(j) + QString("]"));
				}

				QTableWidgetItem* header = new QTableWidgetItem;
				header->setText(QString::number(0));
				this->setVerticalHeaderItem(0, header);

			}
		}


	}


	void PVec2FieldViewerWidget::buildArrayListDataTable(int first, int last)
	{

		{

			//std::string template_name = mfield->getTemplateName();


			//  ***********************  Vec2f ***************************
			CArrayList<Vec2f>* dataPtr = NULL;
			CArrayList<Vec2f> cData;

			// ************************	    GPU	   **************************
			FArrayList<Vec2f, DeviceType::GPU>* f_GPU = TypeInfo::cast<FArrayList<Vec2f, DeviceType::GPU>>(mfield);
			if (f_GPU != nullptr)
			{
				std::shared_ptr<ArrayList<Vec2f, DeviceType::GPU>>& data = f_GPU->getDataPtr();

				cData.assign(*data);
				dataPtr = &cData;
			}

			// ************************	    CPU	   **************************
			FArrayList<Vec2f, DeviceType::CPU>* f_CPU = TypeInfo::cast<FArrayList<Vec2f, DeviceType::CPU>>(mfield);
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
					for (size_t i = 0; i < 2; i++)
					{
						value.append(QString::number(trans[i]) + ", ");

					}
					addItemToPosition(value, rowTarget, j, QString::number(rowId), QString("[") + QString::number(j) + QString("]"), true, rowsHeight);

					value.clear();

				}


				rowId++;
				rowTarget++;
			}

		}



		{
			//  ***********************  Vec2d ***************************
			CArrayList<Vec2d>* dataPtr = NULL;
			CArrayList<Vec2d> cData;

			// ************************	    GPU	   **************************
			FArrayList<Vec2d, DeviceType::GPU>* f_GPU = TypeInfo::cast<FArrayList<Vec2d, DeviceType::GPU>>(mfield);
			if (f_GPU != nullptr)
			{
				std::shared_ptr<ArrayList<Vec2d, DeviceType::GPU>>& data = f_GPU->getDataPtr();

				cData.assign(*data);
				dataPtr = &cData;
			}

			// ************************	    CPU	   **************************
			FArrayList<Vec2d, DeviceType::CPU>* f_CPU = TypeInfo::cast<FArrayList<Vec2d, DeviceType::CPU>>(mfield);
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
					for (size_t i = 0; i < 3; i++)
					{
						value.append(QString::number(trans[i]) + ", ");

					}
					addItemToPosition(value, rowTarget, j, QString::number(rowId), QString("[") + QString::number(j) + QString("]"), true, rowsHeight);

					value.clear();

				}


				rowId++;
				rowTarget++;
			}
		}
		


	}

}