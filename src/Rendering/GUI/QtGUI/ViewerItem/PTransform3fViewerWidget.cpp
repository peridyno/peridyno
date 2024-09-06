#include "PTransform3fViewerWidget.h"



namespace dyno
{

	PTransform3fViewerWidget::PTransform3fViewerWidget(FBase* field, QWidget* pParent) :
		PDataViewerWidget(field, pParent)
	{

		mfield = field;

		rowsHeight = 80;


		
		updateDataTable();

	}
	
	void PTransform3fViewerWidget::updateDataTable()
	{
		PDataViewerWidget::updateDataTable();

	}


	void PTransform3fViewerWidget::buildArrayDataTable(int first, int last)
	{
		//std::string template_name = mfield->getTemplateName();

		CArray<Transform3f>* dataPtr = NULL;
		CArray<Transform3f> cData;
		// ************************	    GPU	   **************************

			// ************************	    GPU	   **************************
		FArray<Transform3f, DeviceType::GPU>* f_GPU = TypeInfo::cast<FArray<Transform3f, DeviceType::GPU>>(mfield);
		if (f_GPU != nullptr)
		{

			std::shared_ptr<Array<Transform3f, DeviceType::GPU>>& data = f_GPU->getDataPtr();

			cData.assign(*data);
			dataPtr = &cData;

		}

		// ************************	    CPU	   **************************
		FArray<Transform3f, DeviceType::CPU>* f_CPU = TypeInfo::cast<FArray<Transform3f, DeviceType::CPU>>(mfield);
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
			auto trans = (*dataPtr)[i];

			QString value;
			for (size_t i = 0; i < 3; i++)
			{
				value.append(QString::number(trans.translation()[i]) + ", ");
				if (i < 2)
					value.append("\n");
			}
			addItemToPosition(value, rowTarget, 0, QString::number(rowId), QString("[") + QString::number(0) + QString("]") + QString(" Translation"), true, rowsHeight);

			value.clear();
			for (size_t i = 0; i < 3; i++)
			{
				value.append(QString::number(trans.rotation()(i, 0)) + ", ");
				value.append(QString::number(trans.rotation()(i, 1)) + ", ");
				value.append(QString::number(trans.rotation()(i, 2)) + ", ");
				if (i < 2)
					value.append("\n");
			}
			addItemToPosition(value, rowTarget, 1, QString::number(rowId), QString("[") + QString::number(1) + QString("]") + QString(" Rotation"), true, rowsHeight);

			value.clear();
			for (size_t i = 0; i < 3; i++)
			{
				value.append(QString::number(trans.scale()[i]) + ",");
				if (i < 2)
					value.append("\n");
			}
			addItemToPosition(value, rowTarget, 2, QString::number(rowId), QString("[") + QString::number(2) + QString("]") + QString(" Scale"), true, rowsHeight);
			
			rowId++;
			rowTarget++;

		}



	}

	void PTransform3fViewerWidget::buildArrayListDataTable(int first, int last)
	{
		//std::string template_name = mfield->getTemplateName();

		CArrayList<Transform3f>* dataPtr = NULL;
		CArrayList<Transform3f> cData;
			
		// ************************	    GPU	   **************************
		FArrayList<Transform3f, DeviceType::GPU>* f_GPU = TypeInfo::cast<FArrayList<Transform3f, DeviceType::GPU>>(mfield);
		if (f_GPU != nullptr)
		{
			std::shared_ptr<ArrayList<Transform3f, DeviceType::GPU>>& data = f_GPU->getDataPtr();

			cData.assign(*data);
			dataPtr = &cData;
		}

		// ************************	    CPU	   **************************
		FArrayList<Transform3f, DeviceType::CPU>* f_CPU = TypeInfo::cast<FArrayList<Transform3f, DeviceType::CPU>>(mfield);
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
					value.append(QString::number(trans.translation()[i]) + ", ");
					if (i < 2)
						value.append("\n");
				}
				addItemToPosition(value, rowTarget, 3 * j, QString::number(rowId), QString("[") + QString::number(j) + QString("]") + QString(" Translation"), true, rowsHeight);

				value.clear();
				for (size_t i = 0; i < 3; i++)
				{
					value.append(QString::number(trans.rotation()(i, 0)) + ", ");
					value.append(QString::number(trans.rotation()(i, 1)) + ", ");
					value.append(QString::number(trans.rotation()(i, 2)) + ", ");
					if (i < 2)
						value.append("\n");
				}
				addItemToPosition(value, rowTarget, 3 * j + 1, QString::number(rowId), QString("[") + QString::number(j) + QString("]") + QString(" Rotation"), true, rowsHeight);

				value.clear();
				for (size_t i = 0; i < 3; i++)
				{
					value.append(QString::number(trans.scale()[i]) + ",");
					if (i < 2)
						value.append("\n");
				}
				addItemToPosition(value, rowTarget, 3 * j + 2, QString::number(rowId), QString("[") + QString::number(j) + QString("]") + QString(" Scale"), true, rowsHeight);

			}


			rowId ++;
			rowTarget ++;
		}
		

		
					

	}



	void PTransform3fViewerWidget::buildVarDataTable()
	{
		//std::string template_name = mfield->getTemplateName();

		// ************************		FVar<Transform3f>    **************************
		{
			FVar<Transform3f>* f = TypeInfo::cast<FVar<Transform3f>>(mfield);
			if (f != nullptr)
			{
				const auto& trans = f->getValue();

				//BuildDataTable
				QString value;
				for (size_t i = 0; i < 3; i++)
				{
					value.append(QString::number(trans.translation()[i]) + ", ");
					if (i < 2)
						value.append("\n");
				}
				addItemToPosition(value, 0, 0, QString("0"), QString(" Translation"), true, rowsHeight);

				value.clear();
				for (size_t i = 0; i < 3; i++)
				{
					value.append(QString::number(trans.rotation()(i, 0)) + ", ");
					value.append(QString::number(trans.rotation()(i, 1)) + ", ");
					value.append(QString::number(trans.rotation()(i, 2)) + ", ");
					if (i < 2)
						value.append("\n");
				}
				addItemToPosition(value, 0, 1, QString("0"), QString(" Rotation"), true, rowsHeight);

				value.clear();
				for (size_t i = 0; i < 3; i++)
				{
					value.append(QString::number(trans.scale()[i]) + ",");
					if (i < 2)
						value.append("\n");
				}
				addItemToPosition(value, 0, 2, QString("0"), QString(" Scale"), true, rowsHeight);

			}
		}
	}

}