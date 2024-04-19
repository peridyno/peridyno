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
		QFormLayout* layout = new QFormLayout(this);
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

			// connection
			connect(fxaaEnabled, &QCheckBox::toggled, [=]() {
				mRenderEngine->setFXAA(fxaaEnabled->isChecked());
				});

			connect(msaaSamples, &QComboBox::currentIndexChanged, [=](int idx) {
				mRenderEngine->setMSAA((1 << idx));
				});

			connect(shadowMapSize, &QComboBox::currentIndexChanged, [=](int idx) {
				mRenderEngine->setShadowMapSize((256 << idx));
				});

			connect(shadowBlurIters, &QSpinBox::valueChanged, [=](int iters) {
				mRenderEngine->setShadowBlurIters(iters);
				});
		}
	}

	void PRenderSetting::updateData()
	{
		if (mRenderEngine)
		{
			// ??
		}
	}

}
