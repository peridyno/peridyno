#include "QVehicleInfoWidget.h"

#include <QLabel>
#include <QScrollBar>
#include <QStyle>
#include <QGridLayout>

#include "Field.h"
#include "QPiecewiseSpinBox.h"

#include <QPushButton>
#include <QWheelEvent>

void QComboBox::wheelEvent(QWheelEvent* e){}

void QAbstractSpinBox::wheelEvent(QWheelEvent* e){}

namespace dyno
{

	// ===================== ShapeDetailListWidget 实现 =====================
	ShapeDetailListWidget::ShapeDetailListWidget(std::vector<ShapeConfig>* shapes, QWidget* parent)
		: QWidget(parent)
		, m_bgColor(Qt::white)          // 默认背景色
		, m_borderColor(Qt::gray)       // 默认边框色
		, m_borderWidth(1)              // 默认边框宽度
	{
		this->mShapes = shapes;
		initUI();

	}

	void ShapeDetailListWidget::initUI()
	{
		// 主布局（按钮 + 滚动区域）

		auto shapeDetailLayout = new QHBoxLayout(this);
		//QLabel* name = new QLabel("Shapes");
		//shapeDetailLayout->addWidget(name);

		m_mainLayout = new QVBoxLayout();
		m_mainLayout->setContentsMargins(0, 0, 0, 0);
		m_mainLayout->setSpacing(8);

		shapeDetailLayout->addLayout(m_mainLayout);
		m_addBtn = new QPushButton(" + ", this);
		shapeDetailLayout->addWidget(m_addBtn);

		// 按钮布局
		QHBoxLayout* btnLayout = new QHBoxLayout();
		btnLayout->setContentsMargins(8, 8, 8, 0);
		btnLayout->setSpacing(8);

		// 添加按钮


		m_mainLayout->addLayout(btnLayout);

		// 滚动区域
		m_scrollArea = new QScrollArea(this);
		m_scrollArea->setWidgetResizable(true); // 自适应内容大小
		m_scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff); // 隐藏水平滚动条
		m_scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);    // 按需显示垂直滚动条

		// 内容容器（存放所有QShapeDetail）
		m_contentWidget = new QWidget(this);
		m_contentLayout = new QVBoxLayout(m_contentWidget);
		m_contentLayout->setContentsMargins(8, 8, 8, 8);
		m_contentLayout->setSpacing(4);
		m_contentLayout->addStretch(1); // 底部拉伸，让元素靠上排列

		m_scrollArea->setWidget(m_contentWidget);
		m_mainLayout->addWidget(m_scrollArea);

		for (size_t i = 0; i < mShapes->size(); i++)
		{
			const auto& shape = (*mShapes)[i];
			this->buildShapeDetail(shape,i);

		}

		// 连接按钮信号
		connect(m_addBtn, &QPushButton::clicked, this, &ShapeDetailListWidget::addShapeDetail);

		// 初始化样式
		updateStyleSheet();
	}

	void ShapeDetailListWidget::buildShapeDetail(ShapeConfig shapeData, int id)
	{
		QShapeDetail* detail = new QShapeDetail(shapeData, id);
		//mShapes->push_back(detail->GetShapeConfig());
		m_contentLayout->insertWidget(m_contentLayout->count() - 1, detail);
		m_detailList.append(detail);

		QObject::connect(detail, QOverload<>::of(&QShapeDetail::shapeChange), [=]() {updateShapeListData(); });
		QObject::connect(detail, SIGNAL(removeShapeItem(int)), this, SLOT(removeShapeDetail(int)));
	}

	void ShapeDetailListWidget::addShapeDetail(bool t)
	{	
		this->mShapes->push_back(ShapeConfig());
		
		QShapeDetail* detail = new QShapeDetail((*this->mShapes)[this->mShapes->size() - 1], this->mShapes->size() - 1);
		//mShapes->push_back(detail->GetShapeConfig());
		m_contentLayout->insertWidget(m_contentLayout->count() - 1, detail);
		m_detailList.append(detail);
		
		QObject::connect(detail, QOverload<>::of(&QShapeDetail::shapeChange), [=]() {updateShapeListData(); });
		QObject::connect(detail, SIGNAL(removeShapeItem(int)), this, SLOT(removeShapeDetail(int)));

		updateShapeListData();
	}

	void ShapeDetailListWidget::updateShapeListData() 
	{
		this->mShapes->resize(m_detailList.size());
		for (size_t i = 0; i < m_detailList.size(); i++)
		{
			(*this->mShapes)[i] = m_detailList[i]->GetShapeConfig();
		}
		emit shapesChange();
	}



	void ShapeDetailListWidget::removeLastShapeDetail()
	{
		if (m_detailList.isEmpty()) {
			return;
		}

		removeShapeDetail(m_detailList.count() - 1);
		updateShapeListData();
	}

	bool ShapeDetailListWidget::removeShapeDetail(int index)
	{
		if (index < 0 || index >= m_detailList.count()) {
			return false; 
		}

		QShapeDetail* detail = m_detailList.takeAt(index);
		m_contentLayout->removeWidget(detail);
		detail->deleteLater();
		
		updateShapeListData();

		return true;
	}

	void ShapeDetailListWidget::clearAll()
	{
		// 清空所有元素
		qDeleteAll(m_detailList);
		m_detailList.clear();

		// 重置布局（保留拉伸项）
		while (m_contentLayout->count() > 1) {
			QLayoutItem* item = m_contentLayout->takeAt(0);
			if (item->widget()) {
				item->widget()->deleteLater();
			}
			delete item;
		}
	}

	void ShapeDetailListWidget::setContainerBgColor(const QColor& color)
	{
		m_bgColor = color;
		updateStyleSheet();
	}

	void ShapeDetailListWidget::setContainerBorderColor(const QColor& color)
	{
		m_borderColor = color;
		updateStyleSheet();
	}

	void ShapeDetailListWidget::setContainerBorderWidth(int width)
	{
		m_borderWidth = width;
		updateStyleSheet();
	}

	void ShapeDetailListWidget::updateStyleSheet()
	{
		// 构建样式表，自定义背景色和边框
		QString styleSheet = QString(
			"QScrollArea {"
			"   background-color: #000000;"
			"   border: %2px solid %3;"
			"   border-radius: 6px;"
			"}"
			"QScrollArea QWidget#m_contentWidget {"
			"   background-color: %1;"
			"}"
			"QPushButton {"
			"   background-color: #454545;"
			"   border: 1px solid %3;"
			"   border-radius: 4px;"
			"   padding: 4px 8px;"
			"}"
			"QPushButton:hover {"
			"   background-color: #9e9e9e;"
			"}"
		).arg(m_bgColor.name())
			.arg(m_borderWidth)
			.arg(m_borderColor.name());

		m_scrollArea->setStyleSheet(styleSheet);
		m_addBtn->setStyleSheet(styleSheet);
	}

	//**************************************** ShapeDetail *****************************************//

	QShapeDetail::QShapeDetail(ShapeConfig shapeData,int id)
	{
		mShapeData = shapeData;
		this->id = id;
		
		this->setWindowFlags(Qt::WindowStaysOnTopHint);

		this->setContentsMargins(0, 0, 0, 0);
		mDeleteButton = new QPushButton(" - ");

		mMainLayout = new QHBoxLayout;
		mMainLayout->setContentsMargins(0, 0, 0, 0);
		mMainLayout->setAlignment(Qt::AlignLeft);
		mMainLayout->setSpacing(2);

		this->setLayout(mMainLayout);

		mTypeCombox = new QComboBox;
		for (ConfigShapeType it : mAllShapeType)
		{
			switch (it)
			{
			case dyno::ConfigShapeType::CONFIG_BOX:
				mTypeCombox->addItem("Box");
				break;
			case dyno::ConfigShapeType::CONFIG_TET:
				mTypeCombox->addItem("Tet");
				break;
			case dyno::ConfigShapeType::CONFIG_CAPSULE:
				mTypeCombox->addItem("Capsule");
				break;
			case dyno::ConfigShapeType::CONFIG_SPHERE:
				mTypeCombox->addItem("Sphere");
				break;
			case dyno::ConfigShapeType::CONFIG_TRI:
				mTypeCombox->addItem("Tri");
				break;
			case dyno::ConfigShapeType::CONFIG_COMPOUND:
				mTypeCombox->addItem("Compound");
				break;
			case dyno::ConfigShapeType::CONFIG_Other:
				mTypeCombox->addItem("Other");
				break;
			default:
				break;
			}
		}
		mMainLayout->addWidget(mDeleteButton);
		mMainLayout->addWidget(mTypeCombox, 0);
		mTypeCombox->setFixedWidth(100);

		{
			//case ConfigShapeType::CONFIG_BOX:
			mHalfLengthWidget = new mVec3fWidget(mShapeData.halfLength, std::string("Half Length"));
			mMainLayout->addWidget(mHalfLengthWidget);
			//case ConfigShapeType::CONFIG_TET:
			mTetWidget_0 = new mVec3fWidget(mShapeData.tet[0], std::string("Tet[0]"), this);
			mTetWidget_1 = new mVec3fWidget(mShapeData.tet[1], std::string("Tet[1]"), this);
			mTetWidget_2 = new mVec3fWidget(mShapeData.tet[2], std::string("Tet[2]"), this);
			mTetWidget_3 = new mVec3fWidget(mShapeData.tet[3], std::string("Tet[3]"), this);

			mMainLayout->addWidget(mTetWidget_0);
			mMainLayout->addWidget(mTetWidget_1);
			mMainLayout->addWidget(mTetWidget_2);
			mMainLayout->addWidget(mTetWidget_3);

			//case ConfigShapeType::CONFIG_CAPSULE:
			mRadiusWidget = new mPiecewiseDoubleSpinBox(mShapeData.radius, "Radius", this);
			mCapsuleLengthWidget = new mPiecewiseDoubleSpinBox(mShapeData.capsuleLength, "Capsule Length", this);
			mMainLayout->addWidget(mRadiusWidget);
			mMainLayout->addWidget(mCapsuleLengthWidget);

		}
		hideAllWidget();
		mDensity = new mPiecewiseDoubleSpinBox(mShapeData.density, "Density");
		mCenterWidget = new mVec3fWidget(mShapeData.center, "Center");
		Vec3f R;
		mShapeData.rot.toEulerAngle(R.y, R.x, R.z);
		mAngleWidget = new mVec3fWidget(R, "Rot");

		mMainLayout->addWidget(mDensity);
		mMainLayout->addWidget(mCenterWidget);
		mMainLayout->addWidget(mAngleWidget);
		mMainLayout->addStretch(1);

		switch (shapeData.shapeType)
		{
		case CONFIG_BOX:
			this->mTypeCombox->setCurrentIndex(0);
			break;
		case CONFIG_TET:
			this->mTypeCombox->setCurrentIndex(1);
			break;
		case CONFIG_CAPSULE:
			this->mTypeCombox->setCurrentIndex(2);
			break;
		case CONFIG_SPHERE:
			this->mTypeCombox->setCurrentIndex(3);
			break;
		case CONFIG_TRI:
			this->mTypeCombox->setCurrentIndex(4);
			break;
		case CONFIG_COMPOUND:
			this->mTypeCombox->setCurrentIndex(5);
			break;
		case CONFIG_Other:
			this->mTypeCombox->setCurrentIndex(6);
			break;
		default:
			break;
		}
		

		updateElement(mTypeCombox->currentIndex());

		
		//if (mShapeData)

		QObject::connect(mTypeCombox, SIGNAL(currentIndexChanged(int)), this, SLOT(updateElement(int)));
		QObject::connect(mDeleteButton, QOverload<>::of(&QPushButton::released), [=]() {removeItemSlot(); });

		QObject::connect(mDensity, QOverload<>::of(&mPiecewiseDoubleSpinBox::valueChange), [=]() {updateData(); });
		QObject::connect(mCenterWidget, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
		QObject::connect(mAngleWidget, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
		QObject::connect(mRadiusWidget, QOverload<>::of(&mPiecewiseDoubleSpinBox::valueChange), [=]() {updateData(); });
		QObject::connect(mCapsuleLengthWidget, QOverload<>::of(&mPiecewiseDoubleSpinBox::valueChange), [=]() {updateData(); });
		QObject::connect(mHalfLengthWidget, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });

		QObject::connect(mTetWidget_0, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
		QObject::connect(mTetWidget_1, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
		QObject::connect(mTetWidget_2, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
		QObject::connect(mTetWidget_3, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });

		
	}


	void QShapeDetail::removeItemSlot() 
	{
		emit removeShapeItem(id);
	}

	void QShapeDetail::updateElement(int type)
	{
		mShapeData.shapeType = mAllShapeType[type];
		hideAllWidget();
		switch (mShapeData.shapeType)
		{
		case ConfigShapeType::CONFIG_BOX:
			mHalfLengthWidget->show();

			break;

		case ConfigShapeType::CONFIG_TET:
			mTetWidget_0->show();
			mTetWidget_1->show();
			mTetWidget_2->show();
			mTetWidget_3->show();

			break;

		case ConfigShapeType::CONFIG_CAPSULE:
			mRadiusWidget->show();
			mCapsuleLengthWidget->show();

			break;
		case ConfigShapeType::CONFIG_SPHERE:
			mRadiusWidget->show();

			break;
		case ConfigShapeType::CONFIG_TRI:
			//;
			break;
		case ConfigShapeType::CONFIG_Other:
			//;
			break;

		default:
			break;
		}
		updateData();
	}

	void QShapeDetail::updateData() 
	{

		mShapeData.density = mDensity->getValue();
		mShapeData.center = mCenterWidget->getValue();
		Quat<Real> q =
			Quat<Real>(Real(M_PI) * mAngleWidget->getValue()[2] / 180, Vec3f(0, 0, 1))
			* Quat<Real>(Real(M_PI) * mAngleWidget->getValue()[1] / 180, Vec3f(0, 1, 0))
			* Quat<Real>(Real(M_PI) * mAngleWidget->getValue()[0] / 180, Vec3f(1, 0, 0));
		q.normalize();
		mShapeData.rot = q;
		mShapeData.radius = mRadiusWidget->getValue();
		mShapeData.capsuleLength = mCapsuleLengthWidget->getValue();
		mShapeData.halfLength = mHalfLengthWidget->getValue();
		mShapeData.tet[0] = mTetWidget_0->getValue();
		mShapeData.tet[1] = mTetWidget_1->getValue();
		mShapeData.tet[2] = mTetWidget_2->getValue();
		mShapeData.tet[3] = mTetWidget_3->getValue();


		emit shapeChange();
	}

	//**************************************** RigidBody Detail *****************************************//

	QRigidBodyDetail::QRigidBodyDetail(RigidBodyConfig* rigidInfo)
	{
		mRigidBodyData = rigidInfo;
		this->setWindowFlags(Qt::WindowStaysOnTopHint);

		this->setContentsMargins(0, 0, 0, 0);

		auto mainLayout = new QVBoxLayout;
		mainLayout->setContentsMargins(0, 0, 0, 0);
		mainLayout->setAlignment(Qt::AlignLeft);
		mainLayout->setSpacing(0);

		this->setLayout(mainLayout);

		auto title = new QLabel(QString((std::string("<b>") + std::string("Rigid Body Name:  ") + rigidInfo->shapeName.name + std::string("</b>")).c_str()), this);

		title->setAlignment(Qt::AlignCenter);
		auto titleLayout = new QHBoxLayout;
		titleLayout->addWidget(title);
		titleLayout->setAlignment(Qt::AlignHCenter);
		titleLayout->setContentsMargins(0, 10, 0, 15);
		mainLayout->addItem(titleLayout);

		QHBoxLayout* rigidNameLayout = new QHBoxLayout;
		mNameInput = new QLineEdit(QString(rigidInfo->shapeName.name.c_str()), this);
		mRigidGroup = new QSpinBox(this);
		mRigidGroup->setValue(mRigidBodyData->ConfigGroup);
		mRigidGroup->setRange(0, 100);
		rigidNameLayout->addWidget(mNameInput);
		rigidNameLayout->addWidget(mRigidGroup);
		mainLayout->addItem(rigidNameLayout);

		//Transform
		Vec3f R;
		rigidInfo->angle.toEulerAngle(R.y,R.x,R.z);
		mPositionWidget = new mVec3fWidget(rigidInfo->position, std::string("Position"), this);
		mAngleWidget = new mVec3fWidget(R * 180 / M_PI, std::string("Angle"), this);
		mOffsetWidget = new mVec3fWidget(rigidInfo->offset, std::string("Offset"), this);
		
		mainLayout->addWidget(mPositionWidget);
		mainLayout->addWidget(mAngleWidget);
		mainLayout->addWidget(mOffsetWidget);

		QHBoxLayout* visualLayout = new QHBoxLayout;
		mVisualMeshID = new QSpinBox(this);
		int value = rigidInfo->visualShapeIds.size() ? rigidInfo->visualShapeIds[0] : -1;
		mVisualMeshID->setValue(value);
		visualLayout->addWidget(new QLabel("Visual ID", this));
		visualLayout->addWidget(mVisualMeshID);
		visualLayout->setContentsMargins(9, 0, 8, 0);
		mainLayout->addItem(visualLayout);
		mShapeConfigs = new ShapeDetailListWidget(&rigidInfo->shapeConfigs, this);



		mainLayout->addWidget(mShapeConfigs);

		mMotionWidget = new QComboBox(this);
		for (auto it : mAllConfigMotionTypes)
		{
			switch (it)
			{
			case dyno::ConfigMotionType::CONFIG_Static:
				mMotionWidget->addItem("Static");
				break;
			case dyno::ConfigMotionType::CONFIG_Kinematic:
				mMotionWidget->addItem("Kinematic");
				break;
			case dyno::ConfigMotionType::CONFIG_Dynamic:
				mMotionWidget->addItem("Dynamic");
				break;
			case dyno::ConfigMotionType::CONFIG_NonRotatable:
				mMotionWidget->addItem("NonRotatable");
				break;
			case dyno::ConfigMotionType::CONFIG_NonGravitative:
				mMotionWidget->addItem("NonGravitative");
				break;
			default:
				break;
			}
		}
		
		switch (rigidInfo->motionType)
		{
		case dyno::ConfigMotionType::CONFIG_Static:
			mMotionWidget->setCurrentIndex(0);
			break;
		case dyno::ConfigMotionType::CONFIG_Kinematic:
			mMotionWidget->setCurrentIndex(1);
			break;
		case dyno::ConfigMotionType::CONFIG_Dynamic:
			mMotionWidget->setCurrentIndex(2);
			break;
		case dyno::ConfigMotionType::CONFIG_NonRotatable:
			mMotionWidget->setCurrentIndex(3);
			break;
		case dyno::ConfigMotionType::CONFIG_NonGravitative:
			mMotionWidget->setCurrentIndex(4);
			break;
		default:
			break;
		}

		QHBoxLayout* motionLayout = new QHBoxLayout;
		motionLayout->addWidget(new QLabel("Motion Type", this));
		motionLayout->addWidget(mMotionWidget);
		mMotionWidget->setFixedWidth(100);
		motionLayout->setContentsMargins(9, 0, 8, 0);
		mainLayout->addItem(motionLayout);




		mLinearVelocity = new mVec3fWidget(rigidInfo->linearVelocity, std::string("Linear Velocity"), this);
		mAngularVelocity = new mVec3fWidget(rigidInfo->angularVelocity, std::string("Angular Velocity"), this);
		mainLayout->addWidget(mLinearVelocity);
		mainLayout->addWidget(mAngularVelocity);

		mFriction = new mPiecewiseDoubleSpinBox(rigidInfo->friction, "Friction", this);
		mRestitution = new mPiecewiseDoubleSpinBox(rigidInfo->restitution, "Friction", this);
		mainLayout->addWidget(mFriction);
		mainLayout->addWidget(mRestitution);

		mMask = new QComboBox(this);
		for (auto it : mAllConfigCollisionMasks)
		{
			switch (it)
			{
			case dyno::ConfigCollisionMask::CONFIG_AllObjects:
				mMask->addItem("AllObjects");
				break;
			case dyno::ConfigCollisionMask::CONFIG_BoxExcluded:
				mMask->addItem("BoxExcluded");
				break;
			case dyno::ConfigCollisionMask::CONFIG_TetExcluded:
				mMask->addItem("TetExcluded");
				break;
			case dyno::ConfigCollisionMask::CONFIG_CapsuleExcluded:
				mMask->addItem("CapsuleExcluded");
				break;
			case dyno::ConfigCollisionMask::CONFIG_SphereExcluded:
				mMask->addItem("SphereExcluded");
				break;
			case dyno::ConfigCollisionMask::CONFIG_BoxOnly:
				mMask->addItem("BoxOnly");
				break;
			case dyno::ConfigCollisionMask::CONFIG_TetOnly:
				mMask->addItem("TetOnly");
				break;
			case dyno::ConfigCollisionMask::CONFIG_CapsuleOnly:
				mMask->addItem("CapsuleOnly");
				break;
			case dyno::ConfigCollisionMask::CONFIG_SphereOnly:
				mMask->addItem("SphereOnly");
				break;
			case dyno::ConfigCollisionMask::CONFIG_Disabled:
				mMask->addItem("Disabled");
				break;
			default:
				break;
			}
		}

		QHBoxLayout* maskLayout = new QHBoxLayout;

		switch (rigidInfo->motionType)
		{
		case dyno::ConfigCollisionMask::CONFIG_AllObjects:
			mMask->setCurrentIndex(0);
			break;
		case dyno::ConfigCollisionMask::CONFIG_BoxExcluded:
			mMask->setCurrentIndex(1);
			break;
		case dyno::ConfigCollisionMask::CONFIG_TetExcluded:
			mMask->setCurrentIndex(2);
			break;
		case dyno::ConfigCollisionMask::CONFIG_CapsuleExcluded:
			mMask->setCurrentIndex(3);
			break;
		case dyno::ConfigCollisionMask::CONFIG_SphereExcluded:
			mMask->setCurrentIndex(4);
			break;
		case dyno::ConfigCollisionMask::CONFIG_BoxOnly:
			mMask->setCurrentIndex(5);
			break;
		case dyno::ConfigCollisionMask::CONFIG_TetOnly:
			mMask->setCurrentIndex(6);
			break;
		case dyno::ConfigCollisionMask::CONFIG_CapsuleOnly:
			mMask->setCurrentIndex(7);
			break;
		case dyno::ConfigCollisionMask::CONFIG_SphereOnly:
			mMask->setCurrentIndex(8);
			break;
		case dyno::ConfigCollisionMask::CONFIG_Disabled:
			mMask->setCurrentIndex(9);
			break;
		default:
			break;
		}

		mMask->setFixedWidth(100);
		maskLayout->addWidget(new QLabel("Mask Type", this));
		maskLayout->addWidget(mMask);
		maskLayout->setContentsMargins(9, 0, 8, 0);
		mainLayout->addItem(maskLayout);


		//mVec3fWidget* mInertia1 = nullptr;
		//mVec3fWidget* mInertia2 = nullptr;
		//mVec3fWidget* mInertia3 = nullptr;

		//std::vector<QShapeDetail*> mShapeConfigs;
		//QComboBox* mMask;

		QObject::connect(mOffsetWidget, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
		QObject::connect(mPositionWidget, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
		QObject::connect(mAngleWidget, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
		QObject::connect(mMotionWidget, QOverload<int>::of(&QComboBox::currentIndexChanged), [=]() {updateData(); });
		QObject::connect(mRigidGroup, QOverload<int>::of(&QSpinBox::valueChanged), [=]() {updateData(); });
		QObject::connect(mShapeConfigs, QOverload<>::of(&ShapeDetailListWidget::shapesChange), [=]() {updateData(); });


		mainLayout->addStretch();

	}


	void QRigidBodyDetail::updateData()
	{
		Vec3f R = mAngleWidget->getValue() * M_PI / 180;
		mRigidBodyData->shapeName.name = mNameInput->text().toStdString();
		mRigidBodyData->shapeName.name = mRigidGroup->value();

		mRigidBodyData->angle = Quat<Real>(R.y, R.x, R.z);
		mRigidBodyData->linearVelocity = mLinearVelocity->getValue();
		mRigidBodyData->angularVelocity = mAngularVelocity->getValue();
		mRigidBodyData->position = mPositionWidget->getValue();
		mRigidBodyData->offset = mOffsetWidget->getValue();

		
		mRigidBodyData->friction = mFriction->getValue();
		mRigidBodyData->restitution = mRestitution->getValue();
		mRigidBodyData->motionType = mAllConfigMotionTypes[mMotionWidget->currentIndex()];
		mRigidBodyData->collisionMask = mAllConfigCollisionMasks[mMask->currentIndex()];

		if (mRigidBodyData->visualShapeIds.size())
			mRigidBodyData->visualShapeIds[0] = mVisualMeshID->value();
		else
			mRigidBodyData->visualShapeIds.push_back(mVisualMeshID->value());

		
		//SquareMatrix<Real, 3> inertia = SquareMatrix<Real, 3>(0.0f);;
		//ConfigShapeType shapeType = ConfigShapeType::CONFIG_Other;
		//std::vector<ShapeConfig> shapeConfigs;



		emit rigidChange();
	}


	//Joint Detail
	QJointBodyDetail::QJointBodyDetail(MultiBodyJointConfig& jointInfo)
	{
		mJointData = &jointInfo;

		this->setMinimumWidth(600);
		this->setMinimumHeight(500);

		this->setWindowFlags(Qt::WindowStaysOnTopHint);

		this->setContentsMargins(0, 0, 0, 0);

		mCurrentType = jointInfo.mJointType;

		auto mainLayout = new QVBoxLayout;
		mainLayout->setContentsMargins(0, 0, 0, 0);
		mainLayout->setAlignment(Qt::AlignLeft);
		mainLayout->setSpacing(0);
		this->setLayout(mainLayout);



		auto title = new QLabel(QString((std::string("<b>") + std::string("Joint:  ")
			+ jointInfo.mRigidBodyName_1.name + std::string(" - ")
			+ jointInfo.mRigidBodyName_2.name + std::string("</b>")).c_str()), this);
		title->setAlignment(Qt::AlignCenter);
		auto titleLayout = new QHBoxLayout;
		titleLayout->addWidget(title);
		titleLayout->setAlignment(Qt::AlignHCenter);
		titleLayout->setContentsMargins(0, 10, 0, 15);
		mainLayout->addItem(titleLayout);

		mAnchorPointWidget = new mVec3fWidget(jointInfo.mAnchorPoint, std::string("AnchorPoint"), this);
		mAnchorPointWidget->getNameLabel()->setMinimumWidth(140);
		mAxisWidget = new mVec3fWidget(jointInfo.mAxis, std::string("Axis"), this);


		mNameLabel = new QToggleLabel("Range", this);
		mNameLabel->setMinimumWidth(90);
		mUseRangeWidget = new QCheckBox(this);
		mUseRangeWidget->setChecked(jointInfo.mUseRange);
		mMinWidget = new QPiecewiseDoubleSpinBox(jointInfo.mMin, this);
		mMaxWidget = new QPiecewiseDoubleSpinBox(jointInfo.mMax, this);
		mMinWidget->setRange(-9999999999, 99999999999);
		mMaxWidget->setRange(-9999999999, 99999999999);
		mMinWidget->setMinimumWidth(120);
		mMaxWidget->setMinimumWidth(120);

		QHBoxLayout* rangeLayout = new QHBoxLayout;
		rangeLayout->setContentsMargins(9, 0, 8, 10);
		rangeLayout->setAlignment(Qt::AlignLeft);
		rangeLayout->setSpacing(10);

		rangeLayout->addWidget(mNameLabel);
		rangeLayout->addStretch(1);
		rangeLayout->addWidget(mMinWidget);
		rangeLayout->addWidget(mMaxWidget);
		rangeLayout->addWidget(mUseRangeWidget);

		mUseMoter = new QCheckBox;
		mMoterInput = new mPiecewiseDoubleSpinBox(jointInfo.mMoter, "Moter", this);
		QHBoxLayout* moterLayout = mMoterInput->getLayout();
		moterLayout->addWidget(mUseMoter);
		

		QObject::connect(mUseMoter, SIGNAL(stateChanged(int)), this, SLOT(emitChange(int)));
		QObject::connect(mUseMoter, QOverload<int>::of(&QCheckBox::stateChanged), [=]() {updateData(); });
		QObject::connect(mMoterInput, QOverload<>::of(&mPiecewiseDoubleSpinBox::valueChange), [=]() {updateData(); });

		QObject::connect(mNameLabel, SIGNAL(toggle(bool)), mMinWidget, SLOT(toggleDecimals(bool)));
		QObject::connect(mNameLabel, SIGNAL(toggle(bool)), mMaxWidget, SLOT(toggleDecimals(bool)));

		QObject::connect(mAnchorPointWidget, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
		QObject::connect(mAxisWidget, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
		QObject::connect(mUseRangeWidget, QOverload<int>::of(&QCheckBox::stateChanged), [=]() {updateData(); });
		QObject::connect(mMinWidget, QOverload<double>::of(&QPiecewiseDoubleSpinBox::valueChanged), [=]() {updateData(); });
		QObject::connect(mMaxWidget, QOverload<double>::of(&QPiecewiseDoubleSpinBox::valueChanged), [=]() {updateData(); });

		mainLayout->addWidget(mAnchorPointWidget);
		mainLayout->addWidget(mAxisWidget);
		mainLayout->addLayout(rangeLayout);
		mainLayout->addWidget(mMoterInput);
		mainLayout->addStretch(1);
	}


	void QJointBodyDetail::updateData()
	{
		mJointData->mAnchorPoint = mAnchorPointWidget->getValue();
		mJointData->mAxis = mAxisWidget->getValue();
		mJointData->mUseRange = mUseRangeWidget->checkState();
		mJointData->mMin = mMinWidget->getRealValue();
		mJointData->mMax = mMaxWidget->getRealValue();
		mJointData->mUseMoter = mUseMoter->isChecked();
		mJointData->mMoter = mMoterInput->getValue();

		emit jointChange();
	}


	//mRigidBodyItemLayout	//RigidBody Configuration

	RigidBodyItemLayout::RigidBodyItemLayout(int id, const RigidBodyConfig& rigidInfo)
	{
		this->setContentsMargins(0, 0, 0, 0);
		this->mRigidInfo = rigidInfo;
		mElementIndex = id;
		mIndexLabel = new QLabel(std::to_string(id).c_str());
		mNameInput = new QLineEdit;

		mShapeIDSpin = new QSpinBox;
		mShapeIDSpin->setRange(-1, 2000);
		mShapeIDSpin->setValue(id);
		mRemoveButton = new QPushButton("Delete");
		mEditButton = new QPushButton("Edit");
	
		QHBoxLayout* properties = new QHBoxLayout();
		mShapeList = new ShapeDetailListWidget(&this->mRigidInfo.shapeConfigs);
		this->addLayout(properties);

		this->addWidget(mShapeList, 0);

		properties->addWidget(mIndexLabel, 0);
		properties->addWidget(mNameInput, 0);
		properties->addWidget(mShapeIDSpin, 0);
		properties->addStretch(1);
		properties->addWidget(mEditButton, 0);
		properties->addWidget(mRemoveButton, 0);


		mIndexLabel->setFixedWidth(25);
		mNameInput->setFixedWidth(100);

		mShapeIDSpin->setFixedWidth(76);
		mEditButton->setFixedWidth(76);
		mRemoveButton->setFixedWidth(76);

		QObject::connect(mNameInput, &QLineEdit::editingFinished, [=]() {emitChange(1); });
		QObject::connect(mNameInput, &QLineEdit::editingFinished, [=]() {emitNameChange(1); });
		QObject::connect(mShapeIDSpin, SIGNAL(valueChanged(int)), this, SLOT(emitChange(int)));
		QObject::connect(mRemoveButton, SIGNAL(pressed()), this, SLOT(emitRemove()));

		this->mNameInput->setText(std::string("Rigid").append(std::to_string(mElementIndex)).c_str());
		QObject::connect(mEditButton, SIGNAL(released()), this, SLOT(createRigidDetailWidget()));
		QObject::connect(mShapeList, QOverload<>::of(&ShapeDetailListWidget::shapesChange), [=]() {emitChange(1); });
	};


	RigidBodyItemLayout::~RigidBodyItemLayout()
	{
		delete mNameInput;
		delete mShapeIDSpin;
		delete mRemoveButton;
		delete mIndexLabel;
		delete mEditButton;
		delete mTypeCombox;

		for (auto it : mDetailWidgets)
		{
			if (it)
				it->close();
		}
		mDetailWidgets.clear();
	};


	RigidBodyConfig RigidBodyItemLayout::value()
	{
		mRigidInfo.shapeName.name = mNameInput->text().toStdString();
		mRigidInfo.shapeName.rigidBodyId = mElementIndex;
		if(mRigidInfo.visualShapeIds.size())
			mRigidInfo.visualShapeIds[0] = mShapeIDSpin->value();
		else
			mRigidInfo.visualShapeIds.push_back(mShapeIDSpin->value());

		return mRigidInfo;
	};

	void RigidBodyItemLayout::setValue(const RigidBodyConfig& v)
	{
		mNameInput->setText(QString(v.shapeName.name.c_str()));

		if(v.visualShapeIds.size())
			mShapeIDSpin->setValue(v.visualShapeIds[0]);
		else
			mShapeIDSpin->setValue(-1);

	}

	void RigidBodyItemLayout::createRigidDetailWidget()
	{
		auto detail = new QRigidBodyDetail(&mRigidInfo);
		detail->show();
		QObject::connect(detail, QOverload<>::of(&QRigidBodyDetail::rigidChange), [=]() {emitChange(1); });
		mDetailWidgets.push_back(detail);

	}



	//mJointItemLayout	// Joint Configuration;
	mJointItemLayout::mJointItemLayout(int id)
	{
		this->setContentsMargins(0, 0, 0, 0);

		mElementIndex = id;

		mIndex = new QLabel(std::to_string(id).c_str());
		mNameInput1 = new QComboBox;
		mNameInput2 = new QComboBox;
		mTypeInput = new QComboBox;

		mEditButton = new QPushButton("Edit");
		mRemoveButton = new QPushButton("Delete");

		//CONFIG_BallAndSocket,
		//	CONFIG_Slider,
		//	CONFIG_Hinge,
		//	CONFIG_Fixed,
		//	CONFIG_Point,
		//	CONFIG_DistanceJoint,
		//	CONFIG_OtherJoint
		for (ConfigJointType it : mAllJointType)
		{
			switch (it)
			{
			case ConfigJointType::CONFIG_BallAndSocket:
				mTypeInput->addItem("Ball");
				break;
			case ConfigJointType::CONFIG_Slider:
				mTypeInput->addItem("Slider");
				break;
			case ConfigJointType::CONFIG_Hinge:
				mTypeInput->addItem("Hinge");
				break;
			case ConfigJointType::CONFIG_Fixed:
				mTypeInput->addItem("Fixed");
				break;
			case ConfigJointType::CONFIG_Point:
				mTypeInput->addItem("Point");
				break;
			case ConfigJointType::CONFIG_DistanceJoint:
				mTypeInput->addItem("Distance");
				break;
			case ConfigJointType::CONFIG_OtherJoint:
				mTypeInput->addItem("Other");
				break;
			default:
				break;
			}
		}

		this->addWidget(mIndex, 0);
		this->addWidget(mNameInput1, 0);
		this->addWidget(mNameInput2, 0);
		this->addWidget(mTypeInput, 0);

		this->addStretch(1);
		this->addWidget(mEditButton, 0);
		this->addWidget(mRemoveButton, 0);

		mIndex->setFixedWidth(25);
		mNameInput1->setFixedWidth(120);
		mNameInput2->setFixedWidth(120);
		mTypeInput->setFixedWidth(100);

		mEditButton->setFixedWidth(50);
		mRemoveButton->setFixedWidth(75);

		QObject::connect(mNameInput1, SIGNAL(currentIndexChanged(int)), this, SLOT(emitChange(int)));
		QObject::connect(mNameInput2, SIGNAL(currentIndexChanged(int)), this, SLOT(emitChange(int)));
		QObject::connect(mNameInput2, SIGNAL(currentIndexChanged(int)), this, SLOT(emitChange(int)));

	
		QObject::connect(mTypeInput, SIGNAL(currentIndexChanged(int)), this, SLOT(emitChange(int)));
		QObject::connect(mRemoveButton, SIGNAL(pressed()), this, SLOT(emitRemove()));
		QObject::connect(mEditButton, SIGNAL(released()), this, SLOT(createJointDetailWidget()));

		this->mTypeInput->setCurrentIndex(2);
	};


	mJointItemLayout::~mJointItemLayout()
	{
		delete mNameInput1;
		delete mNameInput2;
		delete mTypeInput;
		delete mIndex;
		delete mEditButton;
		delete mRemoveButton;
		for (auto it : mDetailWidgets)
		{
			if (it)
				it->close();
		}
		mDetailWidgets.clear();
	}

	MultiBodyJointConfig mJointItemLayout::value()
	{
		//jointInfo.Joint_Actor = ActorId;
		mJointInfo.mRigidBodyName_1.name = mNameInput1->currentText().toStdString();
		mJointInfo.mRigidBodyName_2.name = mNameInput2->currentText().toStdString();

		mJointInfo.mJointType = mAllJointType[mTypeInput->currentIndex()];
		//mJointInfo.mUseMoter = mUseMoter->isChecked();
		//mJointInfo.mMoter = mMoterInput->value();


		return mJointInfo;
	}

	void mJointItemLayout::setValue(const MultiBodyJointConfig& v)
	{
		mName1_ObjID = (v.mRigidBodyName_1.rigidBodyId);
		mName2_ObjID = (v.mRigidBodyName_2.rigidBodyId);

		//Type
		//mUseMoter->setChecked(v.mUseMoter);
		mJointInfo.mUseRange = v.mUseRange;
		mJointInfo.mAnchorPoint = v.mAnchorPoint;
		mJointInfo.mMin = v.mMin;
		mJointInfo.mMax = v.mMax;
		//mMoterInput->setValue(v.mMoter);
		mJointInfo.mAxis = v.mAxis;
	}

	
	void  mJointItemLayout::createJointDetailWidget()
	{
		auto detail = new QJointBodyDetail(this->mJointInfo);
		detail->show();
		QObject::connect(detail, QOverload<>::of(&QJointBodyDetail::jointChange), [=]() {emitChange(1); });

		mDetailWidgets.push_back(detail);
	}


	IMPL_FIELD_WIDGET(MultiBodyBind, QVehicleInfoWidget)

		QVehicleInfoWidget::QVehicleInfoWidget(FBase* field)
		: QFieldWidget(field)
	{
		mMainLayout = new QVBoxLayout;
		mMainLayout->setContentsMargins(0, 0, 0, 0);
		mMainLayout->setAlignment(Qt::AlignLeft);

		this->setLayout(mMainLayout);

		//Label

		auto titleLayout = new QVBoxLayout;

		QLabel* name = new QLabel();
		name->setText(FormatFieldWidgetName(field->getObjectName()));
		titleLayout->addWidget(name);
		mMainLayout->addLayout(titleLayout);

		//RigidBody UI
		auto RigidBodyUI = new QVBoxLayout;
		RigidBodyUI->setContentsMargins(0, 0, 0, 0);

		QHBoxLayout* nameLayout = new QHBoxLayout;

		//QLabel* idLabel = new QLabel("<b>No.</b>", this);
		//QLabel* rigidNameLabel = new QLabel("<b>Name</b>", this);
		//QLabel* shapeIdLabel = new QLabel("<b>ShapeID</b>", this);
		//QLabel* typeLabel = new QLabel("<b>Type</b>", this);
		//QLabel* offsetLabel = new QLabel("<b>Edit</b>", this);
		QLabel* rigidLabel = new QLabel("<b>RigidBody Configs</b>", this);

		QPushButton* addItembutton = new QPushButton("Add Item", this);
		addItembutton->setFixedSize(80, 30);

		nameLayout->addStretch(1);
		//nameLayout->addWidget(idLabel);
		//nameLayout->addWidget(rigidNameLabel);
		//nameLayout->addWidget(shapeIdLabel);
		//nameLayout->addWidget(typeLabel);
		//nameLayout->addWidget(offsetLabel);
		nameLayout->addWidget(rigidLabel);
		nameLayout->addStretch(1);
		nameLayout->addWidget(addItembutton);
		

		RigidBodyUI->addLayout(nameLayout);
		mMainLayout->addLayout(RigidBodyUI);

		mRigidsLayout = new QVBoxLayout;
		mMainLayout->addLayout(mRigidsLayout);

		//idLabel->setFixedWidth(25);
		//rigidNameLabel->setFixedWidth(100);
		//typeLabel->setFixedWidth(76);
		//shapeIdLabel->setFixedWidth(76);
		//offsetLabel->setFixedWidth(76);
		rigidLabel->setFixedWidth(220);
		addItembutton->setFixedWidth(120);
		//idLabel->setAlignment(Qt::AlignCenter);
		//rigidNameLabel->setAlignment(Qt::AlignCenter);
		//typeLabel->setAlignment(Qt::AlignCenter);
		//shapeIdLabel->setAlignment(Qt::AlignCenter);
		//offsetLabel->setAlignment(Qt::AlignCenter);
		rigidLabel->setAlignment(Qt::AlignCenter);

		QObject::connect(addItembutton, SIGNAL(pressed()), this, SLOT(addRigidBodyItemWidget()));
		QObject::connect(addItembutton, SIGNAL(pressed()), this, SLOT(updateJointComboBox()));

		//Joint UI
		auto jointUI = new QVBoxLayout;
		jointUI->setContentsMargins(0, 0, 0, 0);
		QHBoxLayout* jointLayout = new QHBoxLayout;

		//QLabel* jointNumLabel = new QLabel("<b>No.</b>",this);
		//QLabel* actor1 = new QLabel("<b>RigidBody1</b>",this);
		//QLabel* actor2 = new QLabel("<b>RigidBody2</b>",this);
		//QLabel* jointTypeLabel = new QLabel("<b>Type</b>",this);
		//QLabel* moterLabel = new QLabel("<b>Moter</b>",this);
		//QLabel* anchorOffsetLabel = new QLabel("<b>Edit</b>",this);
		QLabel* jointConfigLabel = new QLabel("<b>Joint Configs</b>", this);

		QPushButton* addJointItembutton = new QPushButton("Add Item",this);
		addJointItembutton->setFixedSize(80, 30);

		//jointLayout->addWidget(jointNumLabel);
		//jointLayout->addWidget(actor1);
		//jointLayout->addWidget(actor2);
		//jointLayout->addWidget(jointTypeLabel);
		//jointLayout->addWidget(moterLabel);
		//jointLayout->addWidget(anchorOffsetLabel);
		jointLayout->addStretch(1);
		jointLayout->addWidget(jointConfigLabel);
		jointLayout->addStretch(1);
		jointLayout->addWidget(addJointItembutton);

		jointUI->addLayout(jointLayout);
		jointUI->setContentsMargins(0, 0, 0, 0);
		addJointItembutton->setFixedWidth(120);

		mMainLayout->addLayout(jointUI);
		mJointsLayout = new QVBoxLayout;
		mMainLayout->addLayout(mJointsLayout);

		jointConfigLabel->setFixedWidth(220);
		jointConfigLabel->setAlignment(Qt::AlignCenter);

		QObject::connect(addJointItembutton, SIGNAL(pressed()), this, SLOT(addJointItemWidget()));
		QObject::connect(this, SIGNAL(vectorChange()), this, SLOT(updateField()));

		FVar<MultiBodyBind>* f = TypeInfo::cast<FVar<MultiBodyBind>>(field);
		if (f != nullptr)
		{
			mVec = f->getValue();
		}

		updateWidget();

	};

	void QVehicleInfoWidget::updateField()
	{
		FVar<MultiBodyBind>* f = TypeInfo::cast<FVar<MultiBodyBind>>(field());
		if (f != nullptr)
		{
			f->setValue(mVec);
		}

		//printField();
	};

	QVehicleInfoWidget::~QVehicleInfoWidget()
	{
		delete mMainLayout;
		mName2RigidId.clear();
		mObjID2Name.clear();
	}

	void QVehicleInfoWidget::updateWidget()
	{
		for (size_t i = 0; i < mVec.rigidBodyConfigs.size(); i++)
		{
			buildItemWidget(mVec.rigidBodyConfigs[i]);
		}

		bulidQueryMap();

		for (size_t i = 0; i < mVec.jointConfigs.size(); i++)
		{
			createJointItemWidget(mVec.jointConfigs[i]);
		}
		updateJointComboBox();
		updateVector();
	}

	void QVehicleInfoWidget::bulidQueryMap()
	{
		mObjID2Name.clear();
		mName2RigidId.clear();

		for (auto it : mRigidBodyItems)
		{
			mObjID2Name[it->getObjID()] = it->mNameInput->text().toStdString();
			mName2RigidId[it->mNameInput->text().toStdString()] = it->getRigidID();
		}
	}

	void QVehicleInfoWidget::updateJointComboBox()
	{
		bulidQueryMap();

		for (auto itemLayout : mJointItems)
		{
			//update RigidBody name to widget
			updateJointLayoutComboxText(itemLayout);

			//set index
			itemLayout->mNameInput1->blockSignals(true);
			itemLayout->mNameInput2->blockSignals(true);

			if (itemLayout->mName1_ObjID != -1)
			{
				auto str = mObjID2Name[itemLayout->mName1_ObjID];
				if (str == std::string(""))
				{
					setJointNameComboxEmpty(itemLayout, true, 1, -1);
				}
				else
				{
					auto index = mName2RigidId[str];
					itemLayout->mNameInput1->setCurrentIndex(index);
				}
			}

			if (itemLayout->mName2_ObjID != -1)
			{
				auto str = mObjID2Name[itemLayout->mName2_ObjID];
				if (str == std::string(""))
				{
					setJointNameComboxEmpty(itemLayout, true, 2, -1);
				}
				else
				{
					auto index = mName2RigidId[str];
					itemLayout->mNameInput2->setCurrentIndex(index);
				}
			}


			if (itemLayout->mName1_ObjID == -1)
			{
				setJointNameComboxEmpty(itemLayout, true, 1, -1);
			}

			if (itemLayout->mName2_ObjID == -1)
			{
				setJointNameComboxEmpty(itemLayout, true, 2, -1);
			}
			itemLayout->mNameInput1->blockSignals(false);
			itemLayout->mNameInput2->blockSignals(false);

		}

		updateJoint_RigidObjID();

	}

	void QVehicleInfoWidget::setJointNameComboxEmpty(mJointItemLayout* item, bool emitSignal, int select, int index)
	{
		if (select == 1)
		{
			if (emitSignal)
				item->mNameInput1->blockSignals(false);
			else
				item->mNameInput1->blockSignals(true);
			item->mNameInput1->setCurrentIndex(index);
		}
		else if (select == 2)
		{
			if (emitSignal)
				item->mNameInput2->blockSignals(false);
			else
				item->mNameInput2->blockSignals(true);
			item->mNameInput2->setCurrentIndex(index);
		}

		item->mNameInput1->blockSignals(false);
		item->mNameInput2->blockSignals(false);


	};


	void QVehicleInfoWidget::updateVector()
	{
		//*******************************  UpdateData  *******************************//
		mVec.rigidBodyConfigs.clear();
		for (size_t i = 0; i < mRigidBodyItems.size(); i++)
		{
			mVec.rigidBodyConfigs.push_back(mRigidBodyItems[i]->value());
		}

		//*******************************  bulidQueryMap  *******************************//
		bulidQueryMap();

		//*******************************  UpdateData  *******************************//
		// 
		//update Rigid ID
		mVec.jointConfigs.clear();
		for (size_t i = 0; i < mJointItems.size(); i++)
		{
			auto& jointItem = mJointItems[i];
			mVec.jointConfigs.push_back(jointItem->value());
			auto& jointInfo = mVec.jointConfigs[i];

			if (jointInfo.mRigidBodyName_1.name != std::string(""))
				jointInfo.mRigidBodyName_1.rigidBodyId = mName2RigidId[jointInfo.mRigidBodyName_1.name];

			if (jointInfo.mRigidBodyName_2.name != std::string(""))
				jointInfo.mRigidBodyName_2.rigidBodyId = mName2RigidId[jointInfo.mRigidBodyName_2.name];
		}
		emit vectorChange();
	}

	void QVehicleInfoWidget::removeJointItemWidgetById(int id)
	{
		mJointsLayout->removeItem(mJointItems[id]);
		delete mJointItems[id];
		mJointItems.erase(mJointItems.begin() + id);
		for (size_t i = 0; i < mJointItems.size(); i++)
		{
			mJointItems[i]->setId(i);
		}

		updateVector();
	}


	void QVehicleInfoWidget::updateJoint_RigidObjID()
	{
		for (size_t i = 0; i < mJointItems.size(); i++)
		{
			auto& jointItem = mJointItems[i];
			jointItem->mName1_ObjID = getRigidItemObjID(jointItem->mNameInput1->currentText().toStdString());
			//std::cout << jointItem->mNameInput1->currentText().toStdString() << ", " << jointItem->mName1_ObjID << std::endl;
			jointItem->mName2_ObjID = getRigidItemObjID(jointItem->mNameInput2->currentText().toStdString());

		}
	}

	void QVehicleInfoWidget::addRigidBodyItemWidget()
	{
		addRigidBodyItemWidgetByID();

	}

	void QVehicleInfoWidget::addRigidBodyItemWidgetByID()
	{

		mVec.rigidBodyConfigs.push_back(RigidBodyConfig());
		RigidBodyItemLayout* itemLayout = new RigidBodyItemLayout(mRigidBodyItems.size(),mVec.rigidBodyConfigs[mVec.rigidBodyConfigs.size() - 1]);
		mRigidBodyItems.append(itemLayout);
		itemLayout->setObjId(mVec.rigidBodyConfigs.size() - 1);
		mRigidsLayout->addLayout(itemLayout);

		connectRigidWidgetSignal(itemLayout);

		updateVector();
	}


	void QVehicleInfoWidget::removeRigidBodyItemWidgetById(int id)
	{
		mRigidsLayout->removeItem(mRigidBodyItems[id]);
		mRigidBodyItems[id]->deleteLater();
		mRigidBodyItems.takeAt(id);

		for (size_t i = 0; i < mRigidBodyItems.size(); i++)
		{
			mRigidBodyItems[i]->setId(i);
		}

		//*******************************  UpdateData  *******************************//
		updateVector();
		bulidQueryMap();
		updateJointComboBox();
	}

	void QVehicleInfoWidget::addJointItemWidget()
	{
		mJointItemLayout* itemLayout = new mJointItemLayout(mJointItems.size());

		connectJointWidgetSignal(itemLayout);

		mJointsLayout->addLayout(itemLayout);
		mJointItems.push_back(itemLayout);

		updateJointLayoutComboxText(itemLayout);

		updateVector();
	}

	int QVehicleInfoWidget::getRigidItemObjID(std::string str)
	{
		for (const auto& pair : mObjID2Name) {
			if (pair.second == str) {
				return  pair.first;
			}
		}
		return -1;
	}

	void QVehicleInfoWidget::updateJointLayoutComboxText(mJointItemLayout* itemLayout)
	{
		itemLayout->mNameInput1->blockSignals(true);
		itemLayout->mNameInput2->blockSignals(true);

		itemLayout->mNameInput1->clear();
		itemLayout->mNameInput2->clear();

		for (auto it : mObjID2Name)
		{
			auto name = it.second;
			itemLayout->mNameInput1->addItem(name.c_str());
			itemLayout->mNameInput2->addItem(name.c_str());
		}

		itemLayout->mNameInput1->setCurrentIndex(-1);
		itemLayout->mNameInput2->setCurrentIndex(-1);

		itemLayout->mNameInput1->blockSignals(false);
		itemLayout->mNameInput2->blockSignals(false);
	}

	void QVehicleInfoWidget::buildItemWidget(const RigidBodyConfig& rigidBody)
	{
		RigidBodyItemLayout* itemLayout = new RigidBodyItemLayout(mRigidBodyItems.size(), rigidBody);
		itemLayout->setObjId(mRigidCounter);
		mRigidCounter++;

		itemLayout->setValue(rigidBody);
		connectRigidWidgetSignal(itemLayout);
		mRigidsLayout->addLayout(itemLayout);
		mRigidBodyItems.push_back(itemLayout);

	}

	void QVehicleInfoWidget::createJointItemWidget(const MultiBodyJointConfig& jointInfo)
	{
		mJointItemLayout* itemLayout = new mJointItemLayout(mJointItems.size());
		itemLayout->setValue(jointInfo);
		connectJointWidgetSignal(itemLayout);
		mJointsLayout->addLayout(itemLayout);
		mJointItems.push_back(itemLayout);
	}

	void QVehicleInfoWidget::connectJointWidgetSignal(mJointItemLayout* itemLayout)
	{
		QObject::connect(itemLayout, SIGNAL(removeByElementIndexId(int)), this, SLOT(removeJointItemWidgetById(int)));
		QObject::connect(itemLayout, SIGNAL(valueChange(int)), this, SLOT(updateVector()));

		QObject::connect(itemLayout->mNameInput1, SIGNAL(currentIndexChanged(int)), this, SLOT(updateJoint_RigidObjID()));
		QObject::connect(itemLayout->mNameInput2, SIGNAL(currentIndexChanged(int)), this, SLOT(updateJoint_RigidObjID()));

	}

	void QVehicleInfoWidget::connectRigidWidgetSignal(RigidBodyItemLayout* itemLayout)
	{
		QObject::connect(itemLayout, SIGNAL(removeByElementIndexId(int)), this, SLOT(removeRigidBodyItemWidgetById(int)));
		//QObject::connect(itemLayout, QOverload<int>::of(&RigidBodyItemLayout::removeByElementIndexId), [=]() {removeRigidBodyItemWidgetById(); });
		QObject::connect(itemLayout, QOverload<int>::of(&RigidBodyItemLayout::nameChange), [=]() {updateJointComboBox(); });
		QObject::connect(itemLayout, QOverload<int>::of(&RigidBodyItemLayout::rigidChange), [=]() {updateVector(); });
		QObject::connect(itemLayout->mNameInput, QOverload<>::of(&QLineEdit::editingFinished), [=]() {bulidQueryMap(); });
	}


}


