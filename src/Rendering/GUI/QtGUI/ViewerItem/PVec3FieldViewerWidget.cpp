#include "PVec3FieldViewerWidget.h"
#include <QHeaderView>
#include "Vector/Vector3D.h"

namespace dyno
{

	PVec3FieldViewerWidget::PVec3FieldViewerWidget(FBase* field, QWidget* pParent) :
		PDataViewerWidget(field, pParent)
	{
		mfield = field;


		updateDataTable();

		
	}
	

	void PVec3FieldViewerWidget::updateDataTable() 
	{
		PDataViewerWidget::updateDataTable();
	}


	void PVec3FieldViewerWidget::buildArrayDataTable(int first, int last)
	{


		// ************************		Vec3f      **************************
		{
			CArray<Vec3f>* cVec3f = NULL;
			CArray<Vec3f> cData;

			// ************************	    GPU	   **************************
			FArray<Vec3f, DeviceType::GPU>* f_GPU = TypeInfo::cast<FArray<Vec3f, DeviceType::GPU>>(mfield);
			if (f_GPU != nullptr)
			{
				std::shared_ptr<Array<Vec3f, DeviceType::GPU>>& data = f_GPU->getDataPtr();

				cData.assign(*data);
				cVec3f = &cData;			
			}

			// ************************	    CPU	   **************************
			FArray<Vec3f, DeviceType::CPU>* f_CPU = TypeInfo::cast<FArray<Vec3f, DeviceType::CPU>>(mfield);
			if (f_CPU != nullptr)
			{
				cVec3f = f_CPU->getDataPtr().get();
			}


			//BuildDataTable
			if (cVec3f != NULL)
			{
				uint rowId = first;
				uint rowTarget = 0;
				for (size_t i = first; i <= last; i++)
				{
					for (size_t j = 0; j < 3; j++)
					{
						QString value = "";
						value = (QString::number((*cVec3f)[i][j], 'f', 6));

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


		// ************************		Vec3d      **************************
		{
			CArray<Vec3d>* cVec3d = NULL;
			CArray<Vec3d> cData;

			// ************************	    GPU	   **************************
			FArray<Vec3d, DeviceType::GPU>* f_GPU = TypeInfo::cast<FArray<Vec3d, DeviceType::GPU>>(mfield);
			if (f_GPU != nullptr)
			{
				std::shared_ptr<Array<Vec3d, DeviceType::GPU>>& data = f_GPU->getDataPtr();

				cData.assign(*data);
				cVec3d = &cData;
			}

			// ************************	    CPU	   **************************
			FArray<Vec3d, DeviceType::CPU>* f_CPU = TypeInfo::cast<FArray<Vec3d, DeviceType::CPU>>(mfield);
			if (f_CPU != nullptr)
			{
				cVec3d = f_CPU->getDataPtr().get();
			}


			//BuildDataTable
			if (cVec3d != NULL)
			{
				uint rowId = first;
				uint rowTarget = 0;
				for (size_t i = first; i <= last; i++) 
				{
					for (size_t j = 0; j < 3; j++)
					{
						QString value = "";
						value = (QString::number((*cVec3d)[i][j], 'f', 10));

						addItemToPosition(value, rowTarget, j, QString::number(rowId), QString("double") + QString("[") + QString::number(j) + QString("]"));
					}

					//QTableWidgetItem* header = new QTableWidgetItem;
					//header->setText(QString::number(i));
					//this->setVerticalHeaderItem(i, header);

					rowId++;
					rowTarget++;
				}

			}

		}

		// ************************		Vec3i      **************************
		{
			CArray<Vec3i>* cVec3i = NULL;
			CArray<Vec3i> cData;

			// ************************	    GPU	   **************************
			FArray<Vec3i, DeviceType::GPU>* i_GPU = TypeInfo::cast<FArray<Vec3i, DeviceType::GPU>>(mfield);
			if (i_GPU != nullptr)
			{
				std::shared_ptr<Array<Vec3i, DeviceType::GPU>>& data = i_GPU->getDataPtr();

				cData.assign(*data);
				cVec3i = &cData;
			}

			// ************************	    CPU	   **************************
			FArray<Vec3i, DeviceType::CPU>* i_CPU = TypeInfo::cast<FArray<Vec3i, DeviceType::CPU>>(mfield);
			if (i_CPU != nullptr)
			{
				cVec3i = i_CPU->getDataPtr().get();
			}


			//BuildDataTable
			if (cVec3i != NULL)
			{
				uint rowId = first;
				uint rowTarget = 0;
				for (size_t i = first; i <= last; i++)
				{
					for (size_t j = 0; j < 3; j++)
					{
						QString value = "";
						value = (QString::number((*cVec3i)[i][j]));

						addItemToPosition(value, rowTarget, j, QString::number(rowId), QString("int") + QString("[") + QString::number(j) + QString("]"));
					}


					rowId++;
					rowTarget++;
				}

			}
		}

		// ************************		Vec3u      **************************
		{
			CArray<Vec3u>* cVec3u = NULL;
			CArray<Vec3u> cData;

			// ************************	    GPU	   **************************
			FArray<Vec3u, DeviceType::GPU>* u_GPU = TypeInfo::cast<FArray<Vec3u, DeviceType::GPU>>(mfield);
			if (u_GPU != nullptr)
			{
				std::shared_ptr<Array<Vec3u, DeviceType::GPU>>& data = u_GPU->getDataPtr();

				cData.assign(*data);
				cVec3u = &cData;
			}

			// ************************	    CPU	   **************************
			FArray<Vec3u, DeviceType::CPU>* u_CPU = TypeInfo::cast<FArray<Vec3u, DeviceType::CPU>>(mfield);
			if (u_CPU != nullptr)
			{
				cVec3u = u_CPU->getDataPtr().get();
			}


			//BuildDataTable
			if (cVec3u != NULL)
			{
				uint rowId = first;
				uint rowTarget = 0;
				for (size_t i = first; i <= last; i++)
				{
					for (size_t j = 0; j < 3; j++)
					{
						QString value = "";
						value = (QString::number((*cVec3u)[i][j]));

						addItemToPosition(value, rowTarget, j, QString::number(rowId), QString("uint") + QString("[") + QString::number(j) + QString("]"));
					}
					rowId++;
					rowTarget++;
				}
			}
		}


		// ************************		Vector<int,3>      **************************
		{
			CArray<Vector<int, 3>>* cVec3i = NULL;
			CArray<Vector<int, 3>> cData;

			// ************************	    GPU	   **************************
			FArray<Vector<int, 3>, DeviceType::GPU>* i_GPU = TypeInfo::cast<FArray<Vector<int, 3>, DeviceType::GPU>>(mfield);
			if (i_GPU != nullptr)
			{
				std::shared_ptr<Array<Vector<int, 3>, DeviceType::GPU>>& data = i_GPU->getDataPtr();

				cData.assign(*data);
				cVec3i = &cData;
			}

			// ************************	    CPU	   **************************
			FArray<Vector<int, 3>, DeviceType::CPU>* i_CPU = TypeInfo::cast<FArray<Vector<int, 3>, DeviceType::CPU>>(mfield);
			if (i_CPU != nullptr)
			{
				cVec3i = i_CPU->getDataPtr().get();
			}


			//BuildDataTable
			if (cVec3i != NULL)
			{
				uint rowId = first;
				uint rowTarget = 0;
				for (size_t i = first; i <= last; i++)
				{
					for (size_t j = 0; j < 3; j++)
					{
						QString value = "";
						value = (QString::number((*cVec3i)[i][j]));

						addItemToPosition(value, rowTarget, j, QString::number(rowId), QString("int") + QString("[") + QString::number(j) + QString("]"));
					}


					rowId++;
					rowTarget++;
				}

			}
		}

	}

	void PVec3FieldViewerWidget::buildVarDataTable()
	{

		//std::string template_name = mfield->getTemplateName();

		// ************************		FVar<Vec3f>	   **************************
		{
			FVar<Vec3f>* f = TypeInfo::cast<FVar<Vec3f>>(mfield);
			if (f != nullptr)
			{
				const auto& Vec3f = f->getValue();


				for (size_t j = 0; j < 3; j++)
				{
					QString value = "";
					value = (QString::number((Vec3f)[j], 'f', 6));

					addItemToPosition(value, 0, j, QString::number(0), QString("float") + QString("[") + QString::number(j) + QString("]"));
				}

				QTableWidgetItem* header = new QTableWidgetItem;
				header->setText(QString::number(0));
				this->setVerticalHeaderItem(0, header);	

			}
		}

		// ************************		FVar<Vec3d>	   **************************
		{
			FVar<Vec3d>* f = TypeInfo::cast<FVar<Vec3d>>(mfield);
			if (f != nullptr)
			{
				const auto& Vec3d = f->getValue();


				for (size_t j = 0; j < 3; j++)
				{
					QString value = "";
					value = (QString::number((Vec3d)[j], 'f', 10));

					addItemToPosition(value, 0, j, QString::number(0), QString("double") + QString("[") + QString::number(j) + QString("]"));
				}

				QTableWidgetItem* header = new QTableWidgetItem;
				header->setText(QString::number(0));
				this->setVerticalHeaderItem(0, header);

			}
		}


		// ************************		FVar<Vec3i>	   **************************
		{
			FVar<Vec3i>* f = TypeInfo::cast<FVar<Vec3i>>(mfield);
			if (f != nullptr)
			{
				const auto& Vec3i = f->getValue();


				for (size_t j = 0; j < 3; j++)
				{
					QString value = "";
					value = (QString::number((Vec3i)[j]));

					addItemToPosition(value, 0, j, QString::number(0), QString("int") + QString("[") + QString::number(j) + QString("]"));
				}

				QTableWidgetItem* header = new QTableWidgetItem;
				header->setText(QString::number(0));
				this->setVerticalHeaderItem(0, header);

			}
		}

		// ************************		FVar<Vec3i>	   **************************
		{
			FVar<Vec3u>* f = TypeInfo::cast<FVar<Vec3u>>(mfield);
			if (f != nullptr)
			{
				const auto& Vec3u = f->getValue();


				for (size_t j = 0; j < 3; j++)
				{
					QString value = "";
					value = (QString::number((Vec3u)[j]));

					addItemToPosition(value, 0, j, QString::number(0), QString("uint") + QString("[") + QString::number(j) + QString("]"));
				}

				QTableWidgetItem* header = new QTableWidgetItem;
				header->setText(QString::number(0));
				this->setVerticalHeaderItem(0, header);

			}
		}

		// ************************		FVar<VectorND<int,3>>	   **************************
		{
			FVar<VectorND<int, 3>>* f = TypeInfo::cast<FVar<VectorND<int, 3>>>(mfield);
			if (f != nullptr)
			{
				const auto& Vec3i = f->getValue();


				for (size_t j = 0; j < 3; j++)
				{
					QString value = "";
					value = (QString::number((Vec3i)[j]));

					addItemToPosition(value, 0, j, QString::number(0), QString("int") + QString("[") + QString::number(j) + QString("]"));
				}

				QTableWidgetItem* header = new QTableWidgetItem;
				header->setText(QString::number(0));
				this->setVerticalHeaderItem(0, header);

			}
		}

	}


	void PVec3FieldViewerWidget::buildArrayListDataTable(int first, int last)
	{

		{

			//std::string template_name = mfield->getTemplateName();


			//  ***********************  Vec3f ***************************
			CArrayList<Vec3f>* dataPtr = NULL;
			CArrayList<Vec3f> cData;

			// ************************	    GPU	   **************************
			FArrayList<Vec3f, DeviceType::GPU>* f_GPU = TypeInfo::cast<FArrayList<Vec3f, DeviceType::GPU>>(mfield);
			if (f_GPU != nullptr)
			{
				std::shared_ptr<ArrayList<Vec3f, DeviceType::GPU>>& data = f_GPU->getDataPtr();

				cData.assign(*data);
				dataPtr = &cData;
			}

			// ************************	    CPU	   **************************
			FArrayList<Vec3f, DeviceType::CPU>* f_CPU = TypeInfo::cast<FArrayList<Vec3f, DeviceType::CPU>>(mfield);
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



		{
			//  ***********************  Vec3d ***************************
			CArrayList<Vec3d>* dataPtr = NULL;
			CArrayList<Vec3d> cData;

			// ************************	    GPU	   **************************
			FArrayList<Vec3d, DeviceType::GPU>* f_GPU = TypeInfo::cast<FArrayList<Vec3d, DeviceType::GPU>>(mfield);
			if (f_GPU != nullptr)
			{
				std::shared_ptr<ArrayList<Vec3d, DeviceType::GPU>>& data = f_GPU->getDataPtr();

				cData.assign(*data);
				dataPtr = &cData;
			}

			// ************************	    CPU	   **************************
			FArrayList<Vec3d, DeviceType::CPU>* f_CPU = TypeInfo::cast<FArrayList<Vec3d, DeviceType::CPU>>(mfield);
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
		
		{
			//  ***********************  Vec3i ***************************
			CArrayList<Vec3i>* dataPtr = NULL;
			CArrayList<Vec3i> cData;

			// ************************	    GPU	   **************************
			FArrayList<Vec3i, DeviceType::GPU>* i_GPU = TypeInfo::cast<FArrayList<Vec3i, DeviceType::GPU>>(mfield);
			if (i_GPU != nullptr)
			{
				std::shared_ptr<ArrayList<Vec3i, DeviceType::GPU>>& data = i_GPU->getDataPtr();

				cData.assign(*data);
				dataPtr = &cData;
			}

			// ************************	    CPU	   **************************
			FArrayList<Vec3i, DeviceType::CPU>* i_CPU = TypeInfo::cast<FArrayList<Vec3i, DeviceType::CPU>>(mfield);
			if (i_CPU != nullptr)
			{
				dataPtr = i_CPU->getDataPtr().get();
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


		{
			//  ***********************  Vec3u ***************************
			CArrayList<Vec3u>* dataPtr = NULL;
			CArrayList<Vec3u> cData;

			// ************************	    GPU	   **************************
			FArrayList<Vec3u, DeviceType::GPU>* u_GPU = TypeInfo::cast<FArrayList<Vec3u, DeviceType::GPU>>(mfield);
			if (u_GPU != nullptr)
			{
				std::shared_ptr<ArrayList<Vec3u, DeviceType::GPU>>& data = u_GPU->getDataPtr();

				cData.assign(*data);
				dataPtr = &cData;
			}

			// ************************	    CPU	   **************************
			FArrayList<Vec3u, DeviceType::CPU>* u_CPU = TypeInfo::cast<FArrayList<Vec3u, DeviceType::CPU>>(mfield);
			if (u_CPU != nullptr)
			{
				dataPtr = u_CPU->getDataPtr().get();
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

		{
			//  ***********************  VectorND<int,3> ***************************
			CArrayList<VectorND<int,3>>* dataPtr = NULL;
			CArrayList<VectorND<int, 3>> cData;

			// ************************	    GPU	   **************************
			FArrayList<VectorND<int, 3>, DeviceType::GPU>* i_GPU = TypeInfo::cast<FArrayList<VectorND<int, 3>, DeviceType::GPU>>(mfield);
			if (i_GPU != nullptr)
			{
				std::shared_ptr<ArrayList<VectorND<int, 3>, DeviceType::GPU>>& data = i_GPU->getDataPtr();

				cData.assign(*data);
				dataPtr = &cData;
			}

			// ************************	    CPU	   **************************
			FArrayList<VectorND<int, 3>, DeviceType::CPU>* i_CPU = TypeInfo::cast<FArrayList<VectorND<int, 3>, DeviceType::CPU>>(mfield);
			if (i_CPU != nullptr)
			{
				dataPtr = i_CPU->getDataPtr().get();
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