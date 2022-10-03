#include "PAnimationWidget.h"

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
		m_startSim(nullptr),
		m_resetSim(nullptr),
		StartLabel(nullptr),
		ResetLabel(nullptr),
		Starticon(nullptr),
		Pauseicon(nullptr),
		Reseticon(nullptr),
		Finishicon(nullptr)
	{
		mTotalFrame = 800;

		QHBoxLayout* layout = new QHBoxLayout();
		setLayout(layout);

		QGridLayout* frameLayout	= new QGridLayout();

		mTotalFrameSpinbox = new QSpinBox();
		mTotalFrameSpinbox->setFixedSize(60, 29);
		mTotalFrameSpinbox->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

		mTotalFrameSpinbox->setMaximum(1000000);
		mTotalFrameSpinbox->setValue(mTotalFrame);

		QGridLayout* GLayout = new QGridLayout;
		
		mFrameSlider = new PAnimationQSlider(0, mTotalFrame, this);
		mFrameSlider->setObjectName("AnimationSlider");
		mFrameSlider->setStyleSheet("border-top-right-radius: 0px; border-bottom-right-radius: 0px;");
		mFrameSlider->setFixedHeight(29);

		frameLayout->addWidget(mFrameSlider, 0, 0, 0 , (labelSize - 1) * 2);

		QHBoxLayout* operationLayout = new QHBoxLayout();

		m_startSim = new QPushButton();								//创建按钮
		m_resetSim = new QPushButton();
		m_startSim->setStyleSheet("padding: 6px;");	
		m_resetSim->setStyleSheet("padding: 6px;");

		m_startSim->setShortcut(QKeySequence(Qt::Key_Down));		//设置播放快捷键

		Starticon = new QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/AnimationSlider/Start.png"));//设置按钮icon
		Pauseicon = new QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/AnimationSlider/Pause.png"));
		Reseticon = new QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/AnimationSlider/Reset.png"));
		Finishicon = new QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/AnimationSlider/Finish.png"));

		StartLabel = new QLabel;													//创建QLabel以承载icon
		PAnimationWidget::buildIconLabel(StartLabel,Starticon, m_startSim, 30);		//构建PushButton上的Label样式
		ResetLabel = new QLabel;
		PAnimationWidget::buildIconLabel(ResetLabel, Reseticon, m_resetSim, 30);

		m_resetSim->setCheckable(false);

		operationLayout->addWidget(mTotalFrameSpinbox, 0);
		operationLayout->addWidget(m_startSim, 0);
		operationLayout->addWidget(m_resetSim, 0);
		operationLayout->setSpacing(0);

		m_startSim->setCheckable(true);

		layout->addLayout(frameLayout, 10);
		layout->addStretch();
		layout->addLayout(operationLayout, 1);
		layout->setSpacing(0);
		
		connect(m_startSim, SIGNAL(released()), this, SLOT(toggleSimulation()));
		connect(m_resetSim, SIGNAL(released()), this, SLOT(resetSimulation()));
		connect(PSimulationThread::instance(), SIGNAL(simulationFinished()), this, SLOT(simulationFinished()));

		connect(PSimulationThread::instance(), SIGNAL(oneFrameFinished()), this, SLOT(updateSlider()));

		// 动态改变 Slider
		connect(mTotalFrameSpinbox, SIGNAL(valueChanged(int)), mFrameSlider, SLOT(maximumChanged(int)));

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
		if (m_startSim->isChecked())
		{
			PSimulationThread::instance()->resume();
			m_startSim->setText("");
			//m_startSim->setIcon(*PauseIcon);//更新icon状态
			StartLabel->setPixmap(*Pauseicon);//更新Label上的icon为Pauseicon

			m_resetSim->setDisabled(true);
			mTotalFrameSpinbox->setEnabled(false);
			mFrameSlider->setEnabled(false);
		}
		else
		{
			PSimulationThread::instance()->pause();
			m_startSim->setText("");
			m_resetSim->setDisabled(false);
			StartLabel->setPixmap(*Starticon);		//更新Label上的icon为Starticon


			mTotalFrameSpinbox->setEnabled(true);
			mFrameSlider->setEnabled(true);
		}
	}

	void PAnimationWidget::resetSimulation()
	{
		PSimulationThread::instance()->reset(mTotalFrameSpinbox->value());

		m_startSim->setText("");
		m_startSim->setEnabled(true);
		m_startSim->setChecked(false);
		StartLabel->setPixmap(*Starticon);		//更新Label上的icon为Starticon

		mTotalFrameSpinbox->setEnabled(true);
		mFrameSlider->setEnabled(true);
		mFrameSlider->setValue(0);
	}

	void PAnimationWidget::simulationFinished()
	{
		StartLabel->setPixmap(*Finishicon);		//更新Label上的icon为Finishicon

		m_startSim->setText("");
		m_startSim->setDisabled(true);
		m_startSim->setChecked(false);

		m_resetSim->setDisabled(false);

		mTotalFrameSpinbox->setEnabled(true);
	}
	
	void PAnimationWidget::updateSlider()
	{
		mFrameSlider->setValue(PSimulationThread::instance()->getCurrentFrameNum());
	}

	void PAnimationWidget::buildIconLabel(QLabel* Label, QPixmap* Icon, QPushButton* btn, int size) {

		Label->setScaledContents(true);							//允许图标缩放
		Label->setStyleSheet("background: transparent;");
		Label->setPixmap(*Icon);								//指定icon，设置背景透明，设置大小
		Label->setFixedSize(size,size);

		QHBoxLayout* iconLayout = new QHBoxLayout();			//创建HBoxLayout承载Label
		iconLayout->addWidget(Label);				
		iconLayout->setSizeConstraint(QLayout::SetFixedSize);
		iconLayout->setMargin(0);
		btn->setLayout(iconLayout);								//将Layout指定给目标Button
	}
}
