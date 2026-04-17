#pragma once
#include "PInstanceViewerWidget.h"
#include "Topology/TriangleSet.h"
#include "Topology/DiscreteElements.h"
#include "Mapping/DiscreteElementsToTriangleSet.h"

namespace dyno
{
	class PDiscreteElementViewerWidget : public PInstanceViewerWidget
	{
		Q_OBJECT

	public:

		PDiscreteElementViewerWidget(FBase* field, QWidget* pParent = NULL);

	public slots:

		void updateWidget()override;


	protected:

		FInstance<DiscreteElements<DataType3f>>* f_discreteElement = NULL;
		std::shared_ptr<DiscreteElementsToTriangleSet<DataType3f>> dis2Tri = NULL;
	};


	//class PBasicShapeWidget : public QWidget
	//{
	//	Q_OBJECT

	//public:

	//	PBasicShapeWidget(QWidget* pParent = NULL) 
	//	{
	//		mTypeCombox = new QComboBox;
	//		for (ConfigShapeType it : mAllShapeType)
	//		{
	//			switch (it)
	//			{
	//			case dyno::ConfigShapeType::CONFIG_BOX:
	//				mTypeCombox->addItem("Box");
	//				break;
	//			case dyno::ConfigShapeType::CONFIG_TET:
	//				mTypeCombox->addItem("Tet");
	//				break;
	//			case dyno::ConfigShapeType::CONFIG_CAPSULE:
	//				mTypeCombox->addItem("Capsule");
	//				break;
	//			case dyno::ConfigShapeType::CONFIG_SPHERE:
	//				mTypeCombox->addItem("Sphere");
	//				break;
	//			case dyno::ConfigShapeType::CONFIG_TRI:
	//				mTypeCombox->addItem("Tri");
	//				break;
	//			case dyno::ConfigShapeType::CONFIG_COMPOUND:
	//				mTypeCombox->addItem("Compound");
	//				break;
	//			case dyno::ConfigShapeType::CONFIG_Other:
	//				mTypeCombox->addItem("Other");
	//				break;
	//			default:
	//				break;
	//			}
	//		}
	//	}

	//public slots:

	//	void updateWidget();


	//protected:

	//	const std::vector<ConfigShapeType> mAllShapeType = {
	//		CONFIG_BOX,
	//		CONFIG_TET,
	//		CONFIG_CAPSULE,
	//		CONFIG_SPHERE,
	//		CONFIG_TRI,
	//		CONFIG_Other
	//	};

	//	QComboBox* mTypeCombox = NULL;
	//};



}