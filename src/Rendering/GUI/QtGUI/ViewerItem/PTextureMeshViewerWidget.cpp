#include "PTextureMeshViewerWidget.h"
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

namespace dyno
{

	PTextureMeshViewerWidget::PTextureMeshViewerWidget(FBase* field, QWidget* pParent) :
		PInstanceViewerWidget(field, pParent)
	{
		mfield = field;

		auto layout = new QVBoxLayout();
		this->setLayout(layout);

		f_textureMesh = TypeInfo::cast<FInstance<TextureMesh>>(field);

		auto vertexCount = new QLabel("Vertices:");
		auto normalCount = new QLabel("Normals:");
		auto texCoordCount = new QLabel("Texture Coords:");
		auto shapeCount = new QLabel("Shapes:");

		if (f_textureMesh)
		{
			auto meshData = f_textureMesh->constDataPtr()->geometry();
			if (meshData)
			{
				vertexCount->setText((std::string("Vertices: ") + std::to_string(meshData->vertices().size())).c_str());
				normalCount->setText((std::string("Normals: ") + std::to_string(meshData->normals().size())).c_str());
				texCoordCount->setText((std::string("Texture Coords: ") + std::to_string(meshData->texCoords().size())).c_str());
			}
			shapeCount->setText((std::string("Shapes: ") + std::to_string(f_textureMesh->constDataPtr()->shapes().size())).c_str());
		}

		layout->addWidget(vertexCount);
		layout->addWidget(normalCount);
		layout->addWidget(texCoordCount);
		layout->addWidget(shapeCount);

		QTabWidget* tabWidget = new QTabWidget(this);
		tabWidget->setTabPosition(QTabWidget::North);

		QWidget* tab1 = new QWidget();
		QWidget* tab2 = new QWidget();

		QHBoxLayout* tab1Layout = new QHBoxLayout(tab1);
		tab1->setLayout(tab1Layout);

		QHBoxLayout* tab2Layout = new QHBoxLayout();
		QVBoxLayout* shapesLayout = new QVBoxLayout();

		QWidget* shapesListWidget = new QWidget;
		shapesListWidget->setFixedWidth(500);
		tab2Layout->addWidget(shapesListWidget);
		shapesListWidget->setLayout(shapesLayout);

		tab2->setLayout(tab2Layout);

		tabWidget->addTab(tab2, "Shapes");
		tabWidget->addTab(tab1, "Geometry");

		layout->addWidget(tabWidget);

		if (f_textureMesh)
		{
			{
				auto meshData = f_textureMesh->constDataPtr()->geometry();
				if (meshData)
				{
					f_points = std::make_shared<FArray<Vec3f, DeviceType::GPU>>();
					auto vertices = meshData->vertices();
					f_points->assign(vertices);

					pointViewer = new PVec3FieldViewerWidget(f_points.get());

					QDataViewScrollBar* verticalScrollBar = new QDataViewScrollBar(Qt::Orientation::Vertical);

					connect(verticalScrollBar, &QScrollBar::valueChanged, pointViewer, &PDataViewerWidget::updateDataTableTo);
					connect(pointViewer, &PDataViewerWidget::wheelDeltaAngleChange, verticalScrollBar, &QDataViewScrollBar::updateScrollValue);
					connect(PSimulationThread::instance(), &PSimulationThread::oneFrameFinished, this, &PInstanceViewerWidget::updateWidget);

					pointViewer->setTableScrollBar(verticalScrollBar);
					verticalScrollBar->resize(verticalScrollBar->width(), this->height());
					tab1Layout->addWidget(pointViewer, 1);
					tab1Layout->addWidget(verticalScrollBar);
				}
			}

			// Shapes tab - display shape information
			{


				// Shape ID selector
				QHBoxLayout* shapeIdLayout = new QHBoxLayout();
				QLabel* shapeIdLabel = new QLabel("Shape ID:");
				QSpinBox* shapeIdSpinBox = new QSpinBox();

				shapeIdLayout->addWidget(shapeIdLabel);
				shapeIdLayout->addWidget(shapeIdSpinBox);
				shapesLayout->addLayout(shapeIdLayout);

				QHBoxLayout* isolatedViewLayout = new QHBoxLayout();
				QLabel* isolatedLabel = new QLabel("Isolated View:");
				QCheckBox* isolatedCheckBox = new QCheckBox();
				isolatedCheckBox->setChecked(false);

				isolatedViewLayout->addWidget(isolatedLabel);
				isolatedViewLayout->addWidget(isolatedCheckBox);
				shapesLayout->addLayout(isolatedViewLayout);

				// Shape information labels
				QLabel* triangleCountLabel = new QLabel("Triangle Indices: 0");
				QLabel* normalCountLabel = new QLabel("Normal Indices: 0");
				QLabel* texCoordCountLabel = new QLabel("Texcoords Indices: 0");
				QLabel* boundingBoxLabel = new QLabel("Bounding Box: ");
				boundingBoxLabel->setFixedWidth(110);
				QLabel* boundingTransformLabel = new QLabel("Bounding Transform: ");
				QHBoxLayout* boundingBoxLayout = new QHBoxLayout();
				boundingBoxLayout->addWidget(boundingBoxLabel);
				QVBoxLayout* boundingValueLayout = new QVBoxLayout();
				boundingBoxLayout->addLayout(boundingValueLayout);
				QLabel* minLabel = new QLabel();
				QLabel* maxLabel = new QLabel();
				boundingValueLayout->addWidget(minLabel);
				boundingValueLayout->addWidget(maxLabel);

				QLabel* materialNameLabel = new QLabel("Material Name: ");
				QLineEdit* materialEdit = new QLineEdit();

				QHBoxLayout* materialLayout = new QHBoxLayout();

				mPiecewiseDoubleSpinBox* MetallicWidget = new mPiecewiseDoubleSpinBox(0, "Metallic");
				MetallicWidget->setRange(0, 1);
				MetallicWidget->getDoubleSpinBox()->setMinimumWidth(150);
				MetallicWidget->setContentsMargins(0, 0, 0, 0);
				MetallicWidget->getLayout()->setSpacing(0);
				mPiecewiseDoubleSpinBox* RoughnessWidget = new mPiecewiseDoubleSpinBox(0, "Roughness");
				RoughnessWidget->setRange(0, 1);
				RoughnessWidget->getDoubleSpinBox()->setMinimumWidth(150);
				RoughnessWidget->setContentsMargins(0, 0, 0, 0);
				RoughnessWidget->getLayout()->setSpacing(0);
				mPiecewiseDoubleSpinBox* AlphaWidget = new mPiecewiseDoubleSpinBox(0, "Alpha");
				AlphaWidget->setRange(0, 1);
				AlphaWidget->getDoubleSpinBox()->setMinimumWidth(150);
				AlphaWidget->setContentsMargins(0, 0, 0, 0);
				AlphaWidget->getLayout()->setSpacing(0);
				

				QColorButton* ColorWidget = new QColorButton();

				//MaterialEdit
				materialLayout->addWidget(materialNameLabel);
				materialLayout->addWidget(materialEdit);


				shapesLayout->addWidget(ColorWidget);
				QHBoxLayout* ColorLayout = new QHBoxLayout;
				QLabel* colorLabel = new QLabel("Color");
				ColorLayout->addWidget(colorLabel);
				ColorLayout->addWidget(ColorWidget);
				shapesLayout->addLayout(ColorLayout);
				shapesLayout->addWidget(MetallicWidget);
				shapesLayout->addWidget(RoughnessWidget);
				shapesLayout->addWidget(AlphaWidget);

				//Geometry
				shapesLayout->addWidget(triangleCountLabel);
				shapesLayout->addWidget(normalCountLabel);
				shapesLayout->addWidget(texCoordCountLabel);
				shapesLayout->addLayout(boundingBoxLayout);
				shapesLayout->addWidget(boundingTransformLabel);
				shapesLayout->addLayout(materialLayout);
				shapesLayout->addStretch();



				// Update function for shape information
				auto updateShapeInfo = [=]() {
					int shapeId = shapeIdSpinBox->value();
					if (shapeId >= 0 && shapeId < f_textureMesh->constDataPtr()->shapes().size()) {
						auto shape = f_textureMesh->constDataPtr()->shapes()[shapeId];
						triangleCountLabel->setText((std::string("Triangle Indices: ") + std::to_string(shape->vertexIndex.size())).c_str());
						normalCountLabel->setText((std::string("Normal Indices: ") + std::to_string(shape->normalIndex.size())).c_str());
						texCoordCountLabel->setText((std::string("Texcoords Indices: ") + std::to_string(shape->texCoordIndex.size())).c_str());

						// Bounding Box
						auto bbMin = shape->boundingBox.v0;
						auto bbMax = shape->boundingBox.v1;
						boundingBoxLabel->setText("Bounding: ");
						minLabel->setText(("Min " + std::to_string(bbMin[0]) + ", " + std::to_string(bbMin[1]) + ", " + std::to_string(bbMin[2])).c_str());
						maxLabel->setText(("Max " + std::to_string(bbMax[0]) + ", " + std::to_string(bbMax[1]) + ", " + std::to_string(bbMax[2])).c_str());

						// Bounding Transform
						auto transform = shape->boundingTransform;
						auto translation = transform.translation();
						auto rotation = transform.rotation();
						std::string transformStr = "Bounding Transform: " +
							std::to_string(translation[0]) + ", " + std::to_string(translation[1]) + ", " + std::to_string(translation[2]);
						boundingTransformLabel->setText(transformStr.c_str());

						// Material Name
						std::string materialName = "";
						if (shape->material) {

							materialName = "DefaultMaterial";
							materialEdit->blockSignals(true);
							MetallicWidget->blockSignals(true);
							RoughnessWidget->blockSignals(true);
							AlphaWidget->blockSignals(true);
							materialEdit->setText(materialName.c_str());
							MetallicWidget->setValue(shape->material->metallic);
							RoughnessWidget->setValue(shape->material->roughness);
							AlphaWidget->setValue(shape->material->alpha);
							materialEdit->blockSignals(false);
							MetallicWidget->blockSignals(false);
							RoughnessWidget->blockSignals(false);
							AlphaWidget->blockSignals(false);

							ColorWidget->blockSignals(true);
							ColorWidget->setColor(QColor(shape->material->baseColor.r * 255, shape->material->baseColor.g * 255, shape->material->baseColor.b * 255),true);
							ColorWidget->blockSignals(false);
						}
						else 
						{
							materialEdit->blockSignals(true);
							MetallicWidget->blockSignals(true);
							RoughnessWidget->blockSignals(true);
							AlphaWidget->blockSignals(true);
							materialEdit->setText("");
							MetallicWidget->setValue(0);
							RoughnessWidget->setValue(1);
							AlphaWidget->setValue(1);
							materialEdit->blockSignals(false);
							MetallicWidget->blockSignals(false);
							RoughnessWidget->blockSignals(false);
							AlphaWidget->blockSignals(false);

							ColorWidget->blockSignals(true);
							ColorWidget->setColor(QColor(0,0,0), true);
							ColorWidget->blockSignals(false);
						}		
					}
				};

				connect(shapeIdSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), updateShapeInfo);
				connect(materialEdit, &QLineEdit::editingFinished, [=]() {
					int shapeId = shapeIdSpinBox->value();
					if (shapeId >= 0 && shapeId < f_textureMesh->constDataPtr()->shapes().size()) {
						auto shape = f_textureMesh->constDataPtr()->shapes()[shapeId];
						std::string materialName = materialEdit->text().toStdString();
						if (!materialName.empty()) {
							// Try to get material by name
							auto material = dyno::MaterialManager::getMaterialPtr(materialName);
							if (material) {
								// Set the material for the shape
								shape->material = material;
								f_textureMesh->getDataPtr();
							}
						}
					}
				});

				auto changeShapeMaterial = [=]() {
					int shapeId = shapeIdSpinBox->value();
					if (shapeId >= 0 && shapeId < f_textureMesh->constDataPtr()->shapes().size()) {
						f_textureMesh->constDataPtr()->shapes()[shapeId]->material->metallic = MetallicWidget->getValue();
						f_textureMesh->constDataPtr()->shapes()[shapeId]->material->roughness = RoughnessWidget->getValue();
						f_textureMesh->constDataPtr()->shapes()[shapeId]->material->alpha = AlphaWidget->getValue();
						f_textureMesh->constDataPtr()->shapes()[shapeId]->material->baseColor = 
							Color(
								(float)ColorWidget->getColor().red()/255.0f,
								(float)ColorWidget->getColor().green()/255.0f,
								(float)ColorWidget->getColor().blue()/255
							);
					}
				};

				connect(MetallicWidget, QOverload<>::of(&mPiecewiseDoubleSpinBox::valueChange), changeShapeMaterial);
				connect(RoughnessWidget, QOverload<>::of(&mPiecewiseDoubleSpinBox::valueChange), changeShapeMaterial);
				connect(AlphaWidget, QOverload<>::of(&mPiecewiseDoubleSpinBox::valueChange), changeShapeMaterial);
				connect(ColorWidget, QOverload<const QColor&>::of(&QColorButton::colorChanged), changeShapeMaterial);


				// Initial update
				updateShapeInfo();

				// Render tab - display TextureMesh
				{
					GLMeshRenderWidget* renderWidget = new GLMeshRenderWidget();
					renderWidget->setMinimumSize(400, 300);
					tab2Layout->addWidget(renderWidget,1);

					auto updateRenderShape = [=]() {
						int id = shapeIdSpinBox->value();
						std::vector<uint> shapeIds;

						bool useHighlight = isolatedCheckBox->isChecked();
						if (useHighlight)
							shapeIds.push_back(id);
						else
						{
							for (size_t i = 0; i < f_textureMesh->constDataPtr()->shapes().size(); i++)
								shapeIds.push_back(i);
						}

						renderWidget->setTexMeshShapesID(shapeIds);
						renderWidget->update();

					};

					auto updateTransparcy = [=]() {
						renderWidget->setTransparency(isolatedCheckBox->isChecked());
						renderWidget->update();
						};
					connect(shapeIdSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), updateRenderShape);
					connect(isolatedCheckBox, QOverload<int>::of(&QCheckBox::stateChanged), updateRenderShape);
					connect(isolatedCheckBox, QOverload<int>::of(&QCheckBox::stateChanged), updateTransparcy);

					auto updateGL_Repaint = [=]() {
						renderWidget->updateModuleGL();
						renderWidget->update();
					};

					connect(MetallicWidget, QOverload<>::of(&mPiecewiseDoubleSpinBox::valueChange), updateGL_Repaint);
					connect(RoughnessWidget, QOverload<>::of(&mPiecewiseDoubleSpinBox::valueChange), updateGL_Repaint);
					connect(AlphaWidget, QOverload<>::of(&mPiecewiseDoubleSpinBox::valueChange), updateGL_Repaint);
					connect(ColorWidget, QOverload<const QColor&>::of(&QColorButton::colorChanged), updateGL_Repaint);

					updateRenderShape();
					if (f_textureMesh)
					{
						renderWidget->setTextureMesh(std::vector<FInstance<TextureMesh>*>{f_textureMesh});
					}
				}
			}
		}

		connect(PSimulationThread::instance(), &PSimulationThread::sceneGraphChanged, this, &QWidget::close);


	}

	void PTextureMeshViewerWidget::updateWidget()
	{
		if (f_textureMesh)
		{
			auto meshData = f_textureMesh->constDataPtr()->geometry();
			if (meshData)
			{
				auto vertices = meshData->vertices();
				if (f_points)
					f_points->assign(vertices);
			}
		}
		else
			return;

		if (pointViewer)
			pointViewer->updateDataTable();

	}
}