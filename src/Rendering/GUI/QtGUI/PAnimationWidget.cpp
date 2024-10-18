#include "PAnimationWidget.h"

#include "SceneGraph.h"
#include "SceneGraphFactory.h"
#include "PSimulationThread.h"
#include "Platform.h"

#include <QString>
#include <QGridLayout>
#include <QPushButton>
#include <QSpinBox>
#include <QLineEdit>
#include <QIntValidator>
#include <QDebug>
namespace dyno
{
	PAnimationWidget::PAnimationWidget(QWidget *parent) : 
		QWidget(parent),
		mStartSim(nullptr),
		mResetSim(nullptr),
		mStartLabel(nullptr),
		mResetLabel(nullptr),
		mStartIcon(nullptr),
		mPauseIcon(nullptr),
		mResetIcon(nullptr),
		mFinishIcon(nullptr)
	{
		mTotalFrame = PSimulationThread::instance()->getTotalFrames();

		QHBoxLayout* layout = new QHBoxLayout();
		setLayout(layout);

		QGridLayout* frameLayout	= new QGridLayout();

		mTotalFrameSpinbox = new QSpinBox();
		mTotalFrameSpinbox->setFixedSize(96, 29);
		mTotalFrameSpinbox->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

		mTotalFrameSpinbox->setMaximum(1000000);
		mTotalFrameSpinbox->setValue(mTotalFrame);

		mFrameSlider = new PAnimationQSlider(0, mTotalFrame, this);
		mFrameSlider->setObjectName("AnimationSlider");
		mFrameSlider->setStyleSheet("border-top-right-radius: 0px; border-bottom-right-radius: 0px;");
		mFrameSlider->setFixedHeight(29);

		frameLayout->addWidget(mFrameSlider, 0, 0, 0 , (labelSize - 1) * 2);

		QHBoxLayout* operationLayout = new QHBoxLayout();

		mStartSim = new QPushButton();								//创建按钮
		mResetSim = new QPushButton();
		mStartSim->setStyleSheet("padding: 6px;");	
		mResetSim->setStyleSheet("padding: 6px;");

		mStartSim->setShortcut(QKeySequence(Qt::Key_Space));		//设置播放快捷键

		mStartIcon = new QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/AnimationSlider/Start.png"));//设置按钮icon
		mPauseIcon = new QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/AnimationSlider/Pause.png"));
		mResetIcon = new QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/AnimationSlider/Reset.png"));
		mFinishIcon = new QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/AnimationSlider/Finish.png"));

		mStartLabel = new QLabel;													//创建QLabel以承载icon
		PAnimationWidget::buildIconLabel(mStartLabel,mStartIcon, mStartSim, 30);		//构建PushButton上的Label样式
		mResetLabel = new QLabel;
		PAnimationWidget::buildIconLabel(mResetLabel, mResetIcon, mResetSim, 30);

		mResetSim->setCheckable(false);

		mPersistent = new QCheckBox();
		mPersistent->setObjectName("SimulationCheckBox");
		mPersistent->setChecked(false);
		mPersistent->setFixedSize(29, 29);
		mPersistent->setContentsMargins(0, 0, 0, 0);
		mPersistent->setToolTip("Run the simulation permanently");
		
		operationLayout->addWidget(mTotalFrameSpinbox, 0);
		operationLayout->addWidget(mStartSim, 0);
		operationLayout->addWidget(mResetSim, 0);
		operationLayout->addWidget(mPersistent, 0);
		operationLayout->setSpacing(0);

		mStartSim->setCheckable(true);

		layout->addLayout(frameLayout, 10);
		layout->addStretch();
		layout->addLayout(operationLayout, 1);
		layout->setSpacing(0);
		
		connect(mStartSim, SIGNAL(released()), this, SLOT(toggleSimulation()));
		connect(mResetSim, SIGNAL(released()), this, SLOT(resetSimulation()));
		connect(PSimulationThread::instance(), SIGNAL(simulationFinished()), this, SLOT(simulationFinished()));

		connect(PSimulationThread::instance(), SIGNAL(oneFrameFinished(int)), this, SLOT(updateSlider(int)));

		// 动态改变 Slider
		connect(mTotalFrameSpinbox, SIGNAL(valueChanged(int)), mFrameSlider, SLOT(maximumChanged(int)));
		connect(mTotalFrameSpinbox, SIGNAL(valueChanged(int)), this, SLOT(totalFrameChanged(int)));

		connect(mPersistent, SIGNAL(stateChanged(int)), this, SLOT(runForever(int)));

		PSimulationThread::instance()->start();


	}

	PAnimationWidget::~PAnimationWidget()
	{
		PSimulationThread::instance()->stop();
		PSimulationThread::instance()->deleteLater();
		PSimulationThread::instance()->wait();  //必须等待线程结束
	}
	
	void PAnimationWidget::toggleSimulation()
	{
		if (mStartSim->isChecked())
		{
			PSimulationThread::instance()->resume();
			mStartSim->setText("");
			//m_startSim->setIcon(*PauseIcon);//更新icon状态
			mStartLabel->setPixmap(*mPauseIcon);//更新Label上的icon为Pauseicon

			mResetSim->setDisabled(true);
			mTotalFrameSpinbox->setEnabled(false);
			mFrameSlider->setEnabled(false);

			emit simulationStarted();
		}
		else
		{
			PSimulationThread::instance()->pause();
			mStartSim->setText("");
			mResetSim->setDisabled(false);
			mStartLabel->setPixmap(*mStartIcon);		//更新Label上的icon为Starticon


			mTotalFrameSpinbox->setEnabled(true);
			mFrameSlider->setEnabled(true);

			emit simulationStopped();
		}
	}

	void PAnimationWidget::resetSimulation()
	{
		PSimulationThread::instance()->reset(mTotalFrameSpinbox->value());

		mStartSim->setText("");
		mStartSim->setEnabled(true);
		mStartSim->setChecked(false);
		mStartLabel->setPixmap(*mStartIcon);		//更新Label上的icon为Starticon

		mTotalFrameSpinbox->setEnabled(true);
		mFrameSlider->setEnabled(true);
		mFrameSlider->setValue(0);
	}

	void PAnimationWidget::simulationFinished()
	{
		mStartLabel->setPixmap(*mFinishIcon);		//更新Label上的icon为Finishicon

		mStartSim->setText("");
		mStartSim->setDisabled(true);
		mStartSim->setChecked(false);

		mResetSim->setDisabled(false);

		mTotalFrameSpinbox->setEnabled(true);
	}
	
	void PAnimationWidget::updateSlider(int frame)
	{
		mFrameSlider->setValue(frame);
	}

	void PAnimationWidget::buildIconLabel(QLabel* Label, QPixmap* Icon, QPushButton* btn, int size) {

		Label->setScaledContents(true);							//允许图标缩放
		Label->setStyleSheet("background: transparent;");
		Label->setPixmap(*Icon);								//指定icon，设置背景透明，设置大小
		Label->setFixedSize(size,size);

		QHBoxLayout* iconLayout = new QHBoxLayout();			//创建HBoxLayout承载Label
		iconLayout->addWidget(Label);				
		iconLayout->setSizeConstraint(QLayout::SetFixedSize);
		iconLayout->setContentsMargins(0, 0, 0, 0);
		btn->setLayout(iconLayout);								//将Layout指定给目标Button
	}

	void PAnimationWidget::totalFrameChanged(int num)
	{
		auto scn = SceneGraphFactory::instance()->active();

		if (scn->getFrameNumber() < num)
		{
			mResetSim->setEnabled(true);
			mStartSim->setEnabled(true);
			
			mStartLabel->setPixmap(*mStartIcon);

			PSimulationThread::instance()->proceed(num);
		}
	}

	void PAnimationWidget::runForever(int state)
	{
		switch (state)
		{
		case Qt::CheckState::Checked:
			mStartSim->setDisabled(true);
			mResetSim->setDisabled(true);
			mFrameSlider->setDisabled(true);
			mTotalFrameSpinbox->setDisabled(true);

			disconnect(PSimulationThread::instance(), SIGNAL(oneFrameFinished(int)), this, SLOT(updateSlider(int)));

			PSimulationThread::instance()->setTotalFrames(std::numeric_limits<int>::max());
			PSimulationThread::instance()->resume();

			emit simulationStarted();

			break;

		case Qt::CheckState::Unchecked:
			mStartSim->setDisabled(false);
			mResetSim->setDisabled(false);
			mStartLabel->setPixmap(*mStartIcon);
			mFrameSlider->setDisabled(false);
			mTotalFrameSpinbox->setDisabled(false);

			connect(PSimulationThread::instance(), SIGNAL(oneFrameFinished(int)), this, SLOT(updateSlider(int)));

			PSimulationThread::instance()->setTotalFrames(mTotalFrameSpinbox->value());
			PSimulationThread::instance()->pause();

			emit simulationStopped();

			break;

		default:
			break;
		}
	}

}
