#include "PDiscreteElementViewerWidget.h"
#include <QHeaderView>
#include <QLabel>
#include <QVBoxLayout>
#include <QTabWidget>
#include "Framework/FInstance.h"
#include "Topology/TriangleSet.h"
#include "Topology/Cuda/Topology/TextureMesh.h"
#include "PVec3FieldViewerWidget.h"
#include "PVec2FieldViewerWidget.h"
#include "PSimulationThread.h"
#include <QSpinBox>

// Include the new GLMeshRenderWidget
#include "GLMeshRenderWidget.h"
#include "MaterialManager.h"
#include <QLineEdit>
#include <QCheckBox>
#include <QMouseEvent>
#include "PropertyItem/QColorWidget.h"
#include "PropertyItem/QPiecewiseDoubleSpinBox.h"
#include "PropertyItem/QVehicleInfoWidget.h"

namespace dyno
{
	Quat<Real> quatFromMatrix(const Mat3f& m)
	{
		Quat<Real> q;
		Real trace = m(0, 0) + m(1, 1) + m(2, 2);
		if (trace > 0) {
			Real s = std::sqrt(trace + 1.0) * 2; // s = 4*w
			q.w = 0.25 * s;
			q.x = (m(2, 1) - m(1, 2)) / s;
			q.y = (m(0, 2) - m(2, 0)) / s;
			q.z = (m(1, 0) - m(0, 1)) / s;
		}
		else if ((m(0, 0) > m(1, 1)) & (m(0, 0) > m(2, 2))) {
			Real s = std::sqrt(1.0 + m(0, 0) - m(1, 1) - m(2, 2)) * 2; // s = 4*x
			q.w = (m(2, 1) - m(1, 2)) / s;
			q.x = 0.25 * s;
			q.y = (m(0, 1) + m(1, 0)) / s;
			q.z = (m(0, 2) + m(2, 0)) / s;
		}
		else if (m(1, 1) > m(2, 2)) {
			Real s = std::sqrt(1.0 + m(1, 1) - m(0, 0) - m(2, 2)) * 2; // s = 4*y
			q.w = (m(0, 2) - m(2, 0)) / s;
			q.x = (m(0, 1) + m(1, 0)) / s;
			q.y = 0.25 * s;
			q.z = (m(1, 2) + m(2, 1)) / s;
		}
		else {
			Real s = std::sqrt(1.0 + m(2, 2) - m(0, 0) - m(1, 1)) * 2; // s = 4*z
			q.w = (m(1, 0) - m(0, 1)) / s;
			q.x = (m(0, 2) + m(2, 0)) / s;
			q.y = (m(1, 2) + m(2, 1)) / s;
			q.z = 0.25 * s;
		}
		return q.normalize(); 
	}

	PDiscreteElementViewerWidget::PDiscreteElementViewerWidget(FBase* field, QWidget* pParent) :
		PInstanceViewerWidget(field, pParent)
	{
		mfield = field;

		auto layout = new QVBoxLayout();
		this->setLayout(layout);

		f_discreteElement = TypeInfo::cast<FInstance<DiscreteElements<DataType3f>>>(field);

		auto spheresLabel = new QLabel("Spheres:");
		auto boxesLabel = new QLabel("Boxes:");
		auto tetsLabel = new QLabel("Tets:");
		auto capsulesLabel = new QLabel("Capsules:");
		auto trianglesLabel = new QLabel("Triangles:");

		auto ballAndSocketJointsLabel = new QLabel();
		auto sliderJointsLabel = new QLabel();
		auto hingeJointsLabel = new QLabel();
		auto fixedJointsLabel = new QLabel();
		auto pointJointsLabel = new QLabel();
		auto distanceJointsLabel = new QLabel();

		if (f_discreteElement)
		{	
			spheresLabel->setText((std::string("Spheres: ") + std::to_string(f_discreteElement->constDataPtr()->spheresInLocal().size())).c_str());
			boxesLabel->setText((std::string("Boxes: ") + std::to_string(f_discreteElement->constDataPtr()->boxesInLocal().size())).c_str());
			tetsLabel->setText((std::string("Tets: ") + std::to_string(f_discreteElement->constDataPtr()->tetsInLocal().size())).c_str());
			capsulesLabel->setText((std::string("Capsules: ") + std::to_string(f_discreteElement->constDataPtr()->capsulesInLocal().size())).c_str());
			trianglesLabel->setText((std::string("Triangles: ") + std::to_string(f_discreteElement->constDataPtr()->trianglesInLocal().size())).c_str());
			
			ballAndSocketJointsLabel->setText((std::string("BallAndSocketJoints: ") + std::to_string(f_discreteElement->constDataPtr()->ballAndSocketJoints().size())).c_str());
			sliderJointsLabel->setText((std::string("SliderJoints: ") + std::to_string(f_discreteElement->constDataPtr()->sliderJoints().size())).c_str());
			hingeJointsLabel->setText((std::string("HingeJoints: ") + std::to_string(f_discreteElement->constDataPtr()->hingeJoints().size())).c_str());
			fixedJointsLabel->setText((std::string("FixedJoints: ") + std::to_string(f_discreteElement->constDataPtr()->fixedJoints().size())).c_str());
			pointJointsLabel->setText((std::string("PointJoints: ") + std::to_string(f_discreteElement->constDataPtr()->pointJoints().size())).c_str());
			distanceJointsLabel->setText((std::string("DistanceJoints: ") + std::to_string(f_discreteElement->constDataPtr()->distanceJoints().size())).c_str());
		}

		layout->addWidget(spheresLabel);
		layout->addWidget(boxesLabel);
		layout->addWidget(tetsLabel);
		layout->addWidget(capsulesLabel);
		layout->addWidget(trianglesLabel);
		layout->addWidget(ballAndSocketJointsLabel);
		layout->addWidget(sliderJointsLabel);
		layout->addWidget(hingeJointsLabel);
		layout->addWidget(fixedJointsLabel);
		layout->addWidget(pointJointsLabel);
		layout->addWidget(distanceJointsLabel);

		QTabWidget* tabWidget = new QTabWidget(this);
		tabWidget->setTabPosition(QTabWidget::North);

		QWidget* discreateElementWidget = new QWidget();
		QHBoxLayout* discreteElementsLayout = new QHBoxLayout();
		QVBoxLayout* shapesLayout = new QVBoxLayout();

		QWidget* shapesListWidget = new QWidget;
		shapesListWidget->setFixedWidth(300);
		discreteElementsLayout->addWidget(shapesListWidget);
		shapesListWidget->setLayout(shapesLayout);

		discreateElementWidget->setLayout(discreteElementsLayout);

		tabWidget->addTab(discreateElementWidget, "Widgets");
		layout->addWidget(tabWidget);

		//if (f_discreteElement)
		//{
		//	// Shapes tab - display shape information
		//	{
		//		// Shape ID selector
		//		QHBoxLayout* elementLayout = new QHBoxLayout();
		//		QLabel* elementLabel = new QLabel("Elements:");
		//		elementLayout->addLayout(elementLayout);

		//		QHBoxLayout* transpracyLabelLayout = new QHBoxLayout();
		//		QLabel* transpracyLabel = new QLabel("transpracy:");
		//		QCheckBox* transpracyCheckBox = new QCheckBox();
		//		transpracyCheckBox->setChecked(false);

		//		transpracyLabelLayout->addWidget(transpracyLabel);
		//		transpracyLabelLayout->addWidget(transpracyCheckBox);

		//		shapesLayout->addLayout(elementLayout);
		//		shapesLayout->addLayout(transpracyLabelLayout);

		//		CArray<TOrientedBox3D<Real>> boxes;
		//		boxes.assign(f_discreteElement->constDataPtr()->boxesInLocal());

		//		for (size_t i = 0; i < f_discreteElement->constDataPtr()->boxesInLocal().size(); i++)
		//		{
		//			ShapeConfig config;
		//			config.shapeType = ConfigShapeType::CONFIG_BOX;
		//			config.center = boxes[i].center;

		//			Mat3f R;
		//			R.setCol(0, boxes[i].u);
		//			R.setCol(1, boxes[i].v);
		//			R.setCol(2, boxes[i].w);

		//			config.rot = quatFromMatrix(R);
		//			
		//			config.halfLength = boxes[i].extent;
		//			config.center = boxes[i].center;
		//			config.center = boxes[i].center;

		//			//QShapeDetail shape = new QShapeDetail(config);

		//		}


		//		shapesLayout->addStretch();

		//		auto updateWidgetInfo = [=]() {
		//			
		//		};

		//		auto updateField = [=]() {

		//		};

		//		//connect(MetallicWidget, QOverload<>::of(&mPiecewiseDoubleSpinBox::valueChange), updateField);

		//		// Initial update
		//		updateWidgetInfo();

		//		// Render tab - display TextureMesh
		//		{
		//			GLMeshRenderWidget* renderWidget = new GLMeshRenderWidget();
		//			renderWidget->setMinimumSize(400, 300);
		//			discreteElementsLayout->addWidget(renderWidget,1);

		//			auto updateRenderData = [=]() {
		//				if (dis2Tri == NULL)
		//					dis2Tri = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();

		//				if (f_discreteElement) 
		//				{
		//					f_discreteElement->connect(dis2Tri->inDiscreteElements());
		//					dis2Tri->update();
		//				}
		//			};

		//			auto updateTransparcy = [=]() {
		//				renderWidget->setTransparency(transpracyCheckBox->isChecked());
		//				renderWidget->update();
		//				};

		//			auto updateGL_Repaint = [=]() {
		//				renderWidget->updateModuleGL();
		//				renderWidget->update();
		//			};

		//			
		//			updateRenderData();
		//			if (f_discreteElement && dis2Tri)
		//			{
		//				if (!dis2Tri->outTriangleSet()->isEmpty()) 
		//				{
		//					auto t = TypeInfo::cast<FInstance<TriangleSet<DataType3f>>>(dis2Tri->outTriangleSet());
		//					if(t)
		//						renderWidget->setTriangleSet(std::vector<FInstance<TriangleSet<DataType3f>>*>{t});
		//				}
		//			}
		//		}
		//	}
		//}

		connect(PSimulationThread::instance(), &PSimulationThread::sceneGraphChanged, this, &QWidget::close);
	}

	void PDiscreteElementViewerWidget::updateWidget()
	{

	}

}