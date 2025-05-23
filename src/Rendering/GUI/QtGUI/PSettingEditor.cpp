#include "PSettingEditor.h"

#include <QHBoxLayout>
#include <QDebug>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPainter>
#include <QPushButton>
#include <QToolButton>
#include <QSvgRenderer>
#include <QGridLayout>
#include <QFormLayout>
#include <QComboBox>
#include <QCheckBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QSlider>

#include "PDockWidget.h"
#include "PModuleEditorToolBar.h"

#include "ToolBar/Group.h"
#include "ToolBar/ToolButtonStyle.h"
#include "ToolBar/CompactToolButton.h"

#include "qscrollarea.h"
#include "PropertyItem/QVector3FieldWidget.h"
#include "SceneGraphFactory.h"
#include "SceneGraph.h"

#include <GLRenderEngine.h>



namespace dyno
{
	PSettingEditor::PSettingEditor(QWidget* widget)
		: QMainWindow(nullptr)
		
	{

		this->setWindowTitle(QString("Setting"));

		QDockWidget* Docker = new QDockWidget();
		this->addDockWidget(Qt::LeftDockWidgetArea, Docker);
		Docker->setMinimumSize(QSize(200, 600));
		Docker->setMaximumWidth(200);
		auto titleBar1 = Docker->titleBarWidget();
		Docker->setTitleBarWidget(new QWidget());
		delete titleBar1;

		QLabel* SelectLabel = new QLabel("Cateory");
		SelectLabel->setAlignment(Qt::AlignCenter);
		PPushButton* SceneSettingbtr = new PPushButton("SceneSetting");
		PPushButton* RenderSettingbtr = new PPushButton("RenderSetting");
		PPushButton* OtherSettingbtr = new PPushButton("Other");
		QWidget* selectWidget = new QWidget();
		QVBoxLayout* selectVLayout = new QVBoxLayout();

		settingWidget = new PSceneSetting(this, "Scene Setting");
		settingWidget->updateData();

		renderSettingWidget = new PRenderSetting(this, "Render Setting");

		this->connect(SceneSettingbtr, SIGNAL(active(QString)), this, SLOT(buildSceneSettingWidget()));
		this->connect(OtherSettingbtr, SIGNAL(active(QString)), this, SLOT(buildOtherSettingWidget()));
		this->connect(RenderSettingbtr, SIGNAL(active(QString)), this, SLOT(showRenderSettingWidget()));

		selectVLayout->addWidget(SelectLabel);
		selectVLayout->addWidget(SceneSettingbtr);
		selectVLayout->addWidget(RenderSettingbtr);
		selectVLayout->addWidget(OtherSettingbtr);
		selectVLayout->addStretch();

		selectWidget->setLayout(selectVLayout);
		Docker->setWidget(selectWidget);

		DockerRight = new QDockWidget();
		this->addDockWidget(Qt::RightDockWidgetArea, DockerRight);
		auto titleBar2 = DockerRight->titleBarWidget();
		DockerRight->setMinimumSize(QSize(500,600));
		DockerRight->setTitleBarWidget(new QWidget());
		delete titleBar2;
		DockerRight->setWidget(settingWidget);		

	}

	void PSettingEditor::setRenderEngine(std::shared_ptr<RenderEngine> engine)
	{
		((PRenderSetting*)this->renderSettingWidget)->setRenderEngine(engine);
	}

	void PSettingEditor::buildSceneSettingWidget()
	{	
		if (settingWidget != nullptr)
		{
			delete settingWidget;
			settingWidget = nullptr;
		}

		settingWidget = new PSceneSetting(this,"Scene Setting");
		settingWidget->updateData();
		DockerRight->setWidget(settingWidget);	
	}

	

	void PSettingEditor::buildOtherSettingWidget()
	{
		if (settingWidget != nullptr) 
		{
			delete settingWidget;
			settingWidget = nullptr;
		}
			
		settingWidget = new POtherSetting(this,"Other");
		settingWidget->updateData();
		DockerRight->setWidget(settingWidget);
	}

	void PSettingEditor::showRenderSettingWidget()
	{

		renderSettingWidget->updateData();
		DockerRight->setWidget(renderSettingWidget);
	}


	PSettingWidget::PSettingWidget(PSettingEditor* editor,std::string title)
	{
		Editor = editor;

		layoutRight = new QGridLayout;

		mMainLayout = new QVBoxLayout;
		mScrollArea = new QScrollArea;

		mMainLayout->setContentsMargins(0, 0, 0, 0);
		mMainLayout->setSpacing(0);

		mTitle = new QLabel(title.c_str());
		int detailLabelHeight = 40;
		mTitle->setMinimumHeight(detailLabelHeight);
		mTitle->setMaximumHeight(detailLabelHeight);
		mTitle->setAlignment(Qt::AlignCenter);
		mMainLayout->addWidget(mTitle);

		mMainLayout->addWidget(mScrollArea);

		mScrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
		mScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
		mScrollArea->setWidgetResizable(true);

		mScrollLayout = new QGridLayout;
		mScrollLayout->setAlignment(Qt::AlignTop);
		mScrollLayout->setContentsMargins(0, 0, 0, 0);

		QWidget* m_scroll_widget = new QWidget;
		m_scroll_widget->setLayout(mScrollLayout);
		mScrollArea->setWidget(m_scroll_widget);
		mScrollArea->setContentsMargins(0, 0, 0, 0);

		setLayout(mMainLayout);
	};


	void PSceneSetting::updateData()
	{

		auto scn = SceneGraphFactory::instance()->active();
		Vec3f gValue = scn->getGravity();
		Vec3f lowerBValue = scn->getLowerBound();
		Vec3f upperBValue = scn->getUpperBound();

		gravityWidget = new QVector3FieldWidget(QString("Gravity"), gValue);
		lowerBoundWidget= new QVector3FieldWidget(QString("Lower Bound"), lowerBValue);
		upperBoundWidget = new QVector3FieldWidget(QString("Upper Bound"), upperBValue);

		QObject::connect(gravityWidget, SIGNAL(vec3fChange(double, double, double)), this, SLOT(setGravity(double, double, double)));
		QObject::connect(lowerBoundWidget, SIGNAL(vec3fChange(double, double, double)), this, SLOT(setLowerBound(double, double, double)));
		QObject::connect(upperBoundWidget, SIGNAL(vec3fChange(double, double, double)), this, SLOT(setUpperBound(double, double, double)));

		getScrollLayout()->addWidget(gravityWidget, 0, 0);
		getScrollLayout()->addWidget(lowerBoundWidget, 1, 0);
		getScrollLayout()->addWidget(upperBoundWidget, 2, 0);

	}

	void PSceneSetting::setGravity(double v0, double v1, double v2)
	{
		auto scn = SceneGraphFactory::instance()->active();
		scn->setGravity(Vec3f(float(v0), float(v1), float(v2)));
	}
	void PSceneSetting::setLowerBound(double v0, double v1, double v2)
	{
		auto scn = SceneGraphFactory::instance()->active();
		scn->setLowerBound(Vec3f(float(v0), float(v1), float(v2)));
	}
	void PSceneSetting::setUpperBound(double v0, double v1, double v2)
	{
		auto scn = SceneGraphFactory::instance()->active();
		scn->setUpperBound(Vec3f(float(v0), float(v1), float(v2)));
	}

	PRenderSetting::PRenderSetting(PSettingEditor* editor, std::string title)
		: PSettingWidget(editor, title)
	{
		QFormLayout* layout = new QFormLayout();
		this->getScrollLayout()->addLayout(layout, 0, 0);

		fxaaEnabled = new QCheckBox(this);
		
		msaaSamples = new QComboBox(this);
		msaaSamples->addItems({"Disable", "2", "4", "8"});

		shadowMapSize = new QComboBox(this);
		shadowMapSize->addItems({ "256", "512", "1024", "2048" });

		shadowBlurIters = new QSpinBox(this);
		shadowBlurIters->setRange(0, 10);

		layout->addRow(tr("Enable FXAA"), fxaaEnabled);
		layout->addRow(tr("MSAA Samples"), msaaSamples);
		layout->addRow(tr("ShadowMap Size"), shadowMapSize);
		layout->addRow(tr("ShadowMap Blur"), shadowBlurIters);


		QHBoxLayout* hLayout = new QHBoxLayout;
		hLayout->setContentsMargins(0, 0, 0, 0);
		hLayout->setSpacing(0);

		mLineEdit = new QLineEdit;
		mLineEdit->setText(QString::fromStdString(""));

		QPushButton* open = new QPushButton("Open");
		// 		open->setStyleSheet("QPushButton{color: black;   border-radius: 10px;  border: 1px groove black;background-color:white; }"
		// 							"QPushButton:hover{background-color:white; color: black;}"  
		// 							"QPushButton:pressed{background-color:rgb(85, 170, 255); border-style: inset; }" );
		open->setFixedSize(60, 24);

		hLayout->addWidget(mLineEdit, 0);
		hLayout->addWidget(open, 1);
		hLayout->setSpacing(3);

		layout->addRow("Env Map", hLayout);

		connect(open, &QPushButton::clicked, this, [=]() {
			QString path = QFileDialog::getOpenFileName(this, tr("Open File"), QString::fromStdString(getAssetPath()), tr("Text Files(*.hdr)"));
			if (!path.isEmpty()) {
				//Windows: "\\"; Linux: "/"
				path = QDir::toNativeSeparators(path);
				QFile file(path);
				if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
					QMessageBox::warning(this, tr("Read File"),
						tr("Cannot open file:\n%1").arg(path));
					return;
				}
				mLineEdit->setText(path);
				if (mRenderEngine != nullptr)
				{
					mRenderEngine->setEnvmap(path.toStdString());
				}
				file.close();
			}
			else {
				QMessageBox::warning(this, tr("Path"), tr("You do not select any file."));
			}
			});
	}

	PRenderSetting::~PRenderSetting()
	{

	}

	void PRenderSetting::setRenderEngine(std::shared_ptr<RenderEngine> engine) {
		mRenderEngine = std::dynamic_pointer_cast<GLRenderEngine>(engine);
		if (mRenderEngine)
		{
			// update values
			fxaaEnabled->setChecked(mRenderEngine->getFXAA());
			msaaSamples->setCurrentText(QString::number(mRenderEngine->getMSAA()));
			shadowMapSize->setCurrentText(QString::number(mRenderEngine->getShadowMapSize()));
			shadowBlurIters->setValue(mRenderEngine->getShadowBlurIters());
			mLineEdit->setText(QString::fromStdString(mRenderEngine->getEnvmapFilePath()));

			// connection
			connect(fxaaEnabled, &QCheckBox::toggled, [=]() {
				mRenderEngine->setFXAA(fxaaEnabled->isChecked());
				});

			connect(msaaSamples, SIGNAL(currentIndexChanged(int)), this, SLOT(setMSAA(int)));


			connect(shadowMapSize, SIGNAL(currentIndexChanged(int)), this, SLOT(setShadowMapSize(int)));


			connect(shadowBlurIters, SIGNAL(valueChanged(int)), this, SLOT(setShadowBlurIters(int)));
		}
	}

	void PRenderSetting::updateData()
	{
		if (mRenderEngine)
		{
			// ??
		}
	}

	void PRenderSetting::setMSAA(int idx)
	{
		mRenderEngine->setFXAA(fxaaEnabled->isChecked());
	}

	void PRenderSetting::setShadowMapSize(int idx)
	{
		mRenderEngine->setShadowMapSize((256 << idx));
	}

	void PRenderSetting::setShadowBlurIters(int iters)
	{
		mRenderEngine->setShadowBlurIters(iters);
	}
}
