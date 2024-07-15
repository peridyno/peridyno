#include "QVehicleInfoWidget.h"

#include <QGridLayout>

#include "Field.h"
#include "QPiecewiseSpinBox.h"

#include <QPushButton.h>
#include <QWheelEvent>

void QComboBox::wheelEvent(QWheelEvent* e){}

void QAbstractSpinBox::wheelEvent(QWheelEvent* e){}

namespace dyno
{
	//RigidBody Detail	

	QRigidBodyDetail::QRigidBodyDetail(VehicleRigidBodyInfo& rigidInfo)
	{
		mRigidBodyData = &rigidInfo;
		this->setWindowFlags(Qt::WindowStaysOnTopHint);

		this->setContentsMargins(0, 0, 0, 0);

		mCurrentType = rigidInfo.shapeType;

		auto mainLayout = new QVBoxLayout;
		mainLayout->setContentsMargins(0, 0, 0, 0);
		mainLayout->setAlignment(Qt::AlignLeft);
		mainLayout->setSpacing(0);

		this->setLayout(mainLayout);

		auto title = new QLabel(QString((std::string("<b>") + std::string("Rigid Body Name:  ") + rigidInfo.shapeName.name + std::string("</b>")).c_str()), this);
		
		title->setAlignment(Qt::AlignCenter);
		auto titleLayout = new QHBoxLayout;
		titleLayout->addWidget(title);
		titleLayout->setAlignment(Qt::AlignHCenter);
		titleLayout->setContentsMargins(0, 10, 0, 15);
		mainLayout->addItem(titleLayout);
		//Offset
		mOffsetWidget = new mVec3fWidget(rigidInfo.Offset, std::string("Offset"), this);
		mainLayout->addWidget(mOffsetWidget);

		QObject::connect(mOffsetWidget, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });

		//Transform
		Vec3f R;
		Quat<Real>(rigidInfo.transform.rotation()).toEulerAngle(R[2], R[1], R[0]);

		mTranslationWidget = new mVec3fWidget(rigidInfo.transform.translation(), std::string("Translation"), this);
		mRotationWidget = new mVec3fWidget(R * 180 / M_PI, std::string("Rotation"), this);
		mScaleWidget = new mVec3fWidget(rigidInfo.transform.scale(), std::string("Scale"), this);

		mainLayout->addWidget(mTranslationWidget);
		mainLayout->addWidget(mRotationWidget);
		mainLayout->addWidget(mScaleWidget);

		mMotionWidget = new QComboBox(this);
		for (auto it : mAllConfigMotionTypes)
		{
			switch (it)
			{
			case dyno::ConfigMotionType::Static:
				mMotionWidget->addItem("Static");
				break;
			case dyno::ConfigMotionType::Kinematic:
				mMotionWidget->addItem("Kinematic");
				break;
			case dyno::ConfigMotionType::Dynamic:
				mMotionWidget->addItem("Dynamic");
				break;
			default:
				break;
			}
		}

		QHBoxLayout* motionLayout = new QHBoxLayout;
		motionLayout->addWidget(new QLabel("Motion Type", this));
		motionLayout->addWidget(mMotionWidget);
		motionLayout->setContentsMargins(9, 0, 8, 0);
		mainLayout->addItem(motionLayout);

		QObject::connect(mTranslationWidget, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
		QObject::connect(mRotationWidget, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
		QObject::connect(mScaleWidget, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
		QObject::connect(mMotionWidget, QOverload<int>::of(&QComboBox::currentIndexChanged), [=]() {updateData(); });


		if (true)
		{
			switch (mCurrentType)
			{
			case dyno::Box:
				mHalfLengthWidget = new mVec3fWidget(rigidInfo.halfLength, std::string("Half Length"));
				mainLayout->addWidget(mHalfLengthWidget);
				QObject::connect(mHalfLengthWidget, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });

				break;

			case dyno::Tet:
				mTetWidget_0 = new mVec3fWidget(rigidInfo.tet[0], std::string("Tet[0]"), this);
				mTetWidget_1 = new mVec3fWidget(rigidInfo.tet[1], std::string("Tet[1]"), this);
				mTetWidget_2 = new mVec3fWidget(rigidInfo.tet[2], std::string("Tet[2]"), this);
				mTetWidget_3 = new mVec3fWidget(rigidInfo.tet[3], std::string("Tet[3]"), this);
				mainLayout->addWidget(mTetWidget_0);
				mainLayout->addWidget(mTetWidget_1);
				mainLayout->addWidget(mTetWidget_2);
				mainLayout->addWidget(mTetWidget_3);
				QObject::connect(mTetWidget_0, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
				QObject::connect(mTetWidget_1, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
				QObject::connect(mTetWidget_2, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
				QObject::connect(mTetWidget_3, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });


				break;
			case dyno::Capsule:
				mRadiusWidget = new mPiecewiseDoubleSpinBox(rigidInfo.radius, "Capsule Radius", this);
				mCapsuleLengthWidget = new mPiecewiseDoubleSpinBox(rigidInfo.capsuleLength, "Capsule Length", this);
				mainLayout->addWidget(mRadiusWidget);
				mainLayout->addWidget(mCapsuleLengthWidget);
				QObject::connect(mRadiusWidget, QOverload<>::of(&mPiecewiseDoubleSpinBox::valueChange), [=]() {updateData(); });
				QObject::connect(mCapsuleLengthWidget, QOverload<>::of(&mPiecewiseDoubleSpinBox::valueChange), [=]() {updateData(); });

				break;
			case dyno::Sphere:
				mRadiusWidget = new mPiecewiseDoubleSpinBox(rigidInfo.radius, "Radius", this);
				mainLayout->addWidget(mRadiusWidget);
				QObject::connect(mRadiusWidget, QOverload<>::of(&mPiecewiseDoubleSpinBox::valueChange), [=]() {updateData(); });

				break;
			case dyno::Tri:
				//;
				break;
			case dyno::OtherShape:
				//;
				break;

			default:
				break;
			}
		}

		mainLayout->addStretch();

	}


	void QRigidBodyDetail::updateData()
	{

		mRigidBodyData->halfLength = mOffsetWidget->getValue();

		Quat<Real> q = Quat<Real>(mRotationWidget->getValue()[2] * M_PI / 180, mRotationWidget->getValue()[1] * M_PI / 180, mRotationWidget->getValue()[0] * M_PI / 180);

		mRigidBodyData->transform = Transform3f(mTranslationWidget->getValue(), q.toMatrix3x3(), mScaleWidget->getValue());


		switch (mCurrentType)
		{
		case dyno::Box:
			mRigidBodyData->halfLength = mHalfLengthWidget->getValue();
			break;

		case dyno::Tet:
			mRigidBodyData->tet[0] = mTetWidget_0->getValue();
			mRigidBodyData->tet[1] = mTetWidget_1->getValue();
			mRigidBodyData->tet[2] = mTetWidget_2->getValue();
			mRigidBodyData->tet[3] = mTetWidget_3->getValue();

			break;
		case dyno::Capsule:
			mRigidBodyData->radius = mRadiusWidget->getValue();
			mRigidBodyData->capsuleLength = mCapsuleLengthWidget->getValue();

			break;
		case dyno::Sphere:
			mRigidBodyData->radius = mRadiusWidget->getValue();

			break;
		case dyno::Tri:
			//;
			break;
		case dyno::OtherShape:
			//;
			break;

		default:
			break;
		}
		emit rigidChange();
	}


	//Joint Detail
	QJointBodyDetail::QJointBodyDetail(VehicleJointInfo& jointInfo)
	{
		mJointData = &jointInfo;

		this->setFixedWidth(400);

		this->setWindowFlags(Qt::WindowStaysOnTopHint);

		this->setContentsMargins(0, 0, 0, 0);

		mCurrentType = jointInfo.type;

		auto mainLayout = new QVBoxLayout;
		mainLayout->setContentsMargins(0, 0, 0, 0);
		mainLayout->setAlignment(Qt::AlignLeft);
		mainLayout->setSpacing(0);
		this->setLayout(mainLayout);



		auto title = new QLabel(QString((std::string("<b>") +std::string("Joint:  ") 
			+ jointInfo.JointName1.name + std::string(" - ") 
			+ jointInfo.JointName2.name + std::string("</b>")).c_str()), this);
		title->setAlignment(Qt::AlignCenter);
		auto titleLayout = new QHBoxLayout;
		titleLayout->addWidget(title);
		titleLayout->setAlignment(Qt::AlignHCenter);
		titleLayout->setContentsMargins(0, 10, 0, 15);
		mainLayout->addItem(titleLayout);

		mAnchorPointWidget = new mVec3fWidget(jointInfo.anchorPoint, std::string("AnchorPoint"), this);
		mAxisWidget = new mVec3fWidget(jointInfo.Axis, std::string("Axis"), this);


		mNameLabel = new QToggleLabel("Range", this);
		mNameLabel->setMinimumWidth(90);
		mUseRangeWidget = new QCheckBox(this);
		mUseRangeWidget->setChecked(jointInfo.useRange);
		mMinWidget = new QPiecewiseDoubleSpinBox(jointInfo.d_min, this);
		mMaxWidget = new QPiecewiseDoubleSpinBox(jointInfo.d_max, this);
		mMinWidget->setMinimumWidth(120);
		mMaxWidget->setMinimumWidth(120);

		QHBoxLayout* rangeLayout = new QHBoxLayout;
		rangeLayout->setContentsMargins(9, 0, 8, 10);
		rangeLayout->setAlignment(Qt::AlignLeft);
		rangeLayout->setSpacing(10);

		rangeLayout->addWidget(mNameLabel);
		rangeLayout->addStretch();
		rangeLayout->addWidget(mUseRangeWidget);
		rangeLayout->addWidget(mMinWidget);
		rangeLayout->addWidget(mMaxWidget);


		QObject::connect(mNameLabel, SIGNAL(toggle(bool)), mMinWidget, SLOT(toggleDecimals(bool)));
		QObject::connect(mNameLabel, SIGNAL(toggle(bool)), mMaxWidget, SLOT(toggleDecimals(bool)));

		QObject::connect(mAnchorPointWidget, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
		QObject::connect(mAxisWidget, QOverload<>::of(&mVec3fWidget::vec3fChange), [=]() {updateData(); });
		QObject::connect(mUseRangeWidget, QOverload<int>::of(&QCheckBox::stateChanged), [=]() {updateData(); });
		QObject::connect(mMinWidget, QOverload<double>::of(&QPiecewiseDoubleSpinBox::valueChanged), [=]() {updateData(); });
		QObject::connect(mMaxWidget, QOverload<double>::of(&QPiecewiseDoubleSpinBox::valueChanged), [=]() {updateData(); });

		mainLayout->addWidget(mAnchorPointWidget);
		mainLayout->addWidget(mAxisWidget);
		mainLayout->addItem(rangeLayout);
		mainLayout->addStretch();
	}


	void QJointBodyDetail::updateData()
	{
		mJointData->anchorPoint = mAnchorPointWidget->getValue();
		mJointData->Axis = mAxisWidget->getValue();
		mJointData->useRange = mUseRangeWidget->checkState();
		mJointData->d_min = mMinWidget->getRealValue();
		mJointData->d_max = mMaxWidget->getRealValue();

		emit jointChange();
	}


	//mRigidBodyItemLayout	//RigidBody Configuration

	RigidBodyItemLayout::RigidBodyItemLayout(int id)
	{
		this->setContentsMargins(0, 0, 0, 0);

		mElementIndex = id;
		mIndexLabel = new QLabel(std::to_string(id).c_str());
		mNameInput = new QLineEdit;
		mTypeCombox = new QComboBox;

		mShapeIDSpin = new QSpinBox;
		mShapeIDSpin->setRange(-1, 2000);
		mShapeIDSpin->setValue(-1);
		mRemoveButton = new QPushButton("Delete");
		mOffsetButton = new QPushButton("Edit");

		for (ConfigShapeType it : mVecShapeType)
		{
			switch (it)
			{
			case dyno::ConfigShapeType::Box:
				mTypeCombox->addItem("Box");
				break;
			case dyno::ConfigShapeType::Tet:
				mTypeCombox->addItem("Tet");
				break;
			case dyno::ConfigShapeType::Capsule:
				mTypeCombox->addItem("Capsule");
				break;
			case dyno::ConfigShapeType::Sphere:
				mTypeCombox->addItem("Sphere");
				break;
			case dyno::ConfigShapeType::Tri:
				mTypeCombox->addItem("Tri");
				break;
			case dyno::ConfigShapeType::OtherShape:
				mTypeCombox->addItem("Other");
				break;
			default:
				break;
			}
		}

		this->addWidget(mIndexLabel, 0);
		this->addWidget(mNameInput, 0);
		this->addWidget(mShapeIDSpin, 0);
		this->addWidget(mTypeCombox, 0);
		this->addWidget(mOffsetButton, 0);
		this->addWidget(mRemoveButton, 0);

		mIndexLabel->setFixedWidth(25);
		mNameInput->setFixedWidth(100);
		mTypeCombox->setFixedWidth(76);
		mShapeIDSpin->setFixedWidth(76);
		mOffsetButton->setFixedWidth(76);
		mRemoveButton->setFixedWidth(76);

		QObject::connect(mNameInput, &QLineEdit::editingFinished, [=]() {emitChange(1); });
		QObject::connect(mNameInput, &QLineEdit::editingFinished, [=]() {emitNameChange(1); });
		QObject::connect(mShapeIDSpin, SIGNAL(valueChanged(int)), this, SLOT(emitChange(int)));
		QObject::connect(mTypeCombox, SIGNAL(currentIndexChanged(int)), this, SLOT(emitChange(int)));

		QObject::connect(mRemoveButton, SIGNAL(pressed()), this, SLOT(emitRemove()));

		this->mNameInput->setText(std::string("Rigid").append(std::to_string(mElementIndex)).c_str());
		this->mTypeCombox->setCurrentIndex(2);

		QObject::connect(mOffsetButton, SIGNAL(released()), this, SLOT(createRigidDetailWidget()));

	};


	RigidBodyItemLayout::~RigidBodyItemLayout()
	{
		delete mNameInput;
		delete mShapeIDSpin;
		delete mRemoveButton;
		delete mIndexLabel;
		delete mOffsetButton;
		delete mTypeCombox;

		for (auto it : mDetailWidgets)
		{
			if (it)
				it->close();
		}
		mDetailWidgets.clear();
	};


	VehicleRigidBodyInfo RigidBodyItemLayout::value()
	{
		mRigidInfo.shapeName.name = mNameInput->text().toStdString();
		mRigidInfo.shapeName.rigidId = mElementIndex;
		mRigidInfo.meshShapeId = mShapeIDSpin->value();
		mRigidInfo.shapeType = mVecShapeType[mTypeCombox->currentIndex()];
		return mRigidInfo;
	};

	void RigidBodyItemLayout::setValue(const VehicleRigidBodyInfo& v)
	{
		mNameInput->setText(QString(v.shapeName.name.c_str()));
		mShapeIDSpin->setValue(v.meshShapeId);
		for (size_t i = 0; i < mVecShapeType.size(); i++)
		{
			if (mVecShapeType[i] == v.shapeType)
				mTypeCombox->setCurrentIndex(i);
		}

		mRigidInfo.transform = v.transform;
		mRigidInfo.Offset = v.Offset;
		mRigidInfo.halfLength = v.halfLength;
		mRigidInfo.radius = v.radius;
		mRigidInfo.tet = v.tet;
		mRigidInfo.capsuleLength = v.capsuleLength;
		mRigidInfo.motion = v.motion;
	}

	void RigidBodyItemLayout::createRigidDetailWidget()
	{
		auto detail = new QRigidBodyDetail(this->mRigidInfo);
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

		mUseMoter = new QCheckBox;
		mMoterInput = new QDoubleSpinBox;
		mEditButton = new QPushButton("Edit");
		mRemoveButton = new QPushButton("Delete");

		for (ConfigJointType it : mVecJointType)
		{
			switch (it)
			{
			case dyno::BallAndSocket:
				mTypeInput->addItem("Ball");
				break;
			case dyno::Slider:
				mTypeInput->addItem("Slider");
				break;
			case dyno::Hinge:
				mTypeInput->addItem("Hinge");
				break;
			case dyno::Fixed:
				mTypeInput->addItem("Fixed");
				break;
			case dyno::Point:
				mTypeInput->addItem("Point");
				break;
			case dyno::OtherJoint:
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
		this->addWidget(mUseMoter, 0);
		this->addWidget(mMoterInput, 0);
		this->addWidget(mEditButton, 0);
		this->addWidget(mRemoveButton, 0);

		mIndex->setFixedWidth(25);
		mNameInput1->setFixedWidth(90);
		mNameInput2->setFixedWidth(90);
		mTypeInput->setFixedWidth(65);
		mUseMoter->setFixedWidth(20);
		mMoterInput->setFixedWidth(45);
		mEditButton->setFixedWidth(50);
		mRemoveButton->setFixedWidth(75);

		QObject::connect(mNameInput1, SIGNAL(currentIndexChanged(int)), this, SLOT(emitChange(int)));
		QObject::connect(mNameInput2, SIGNAL(currentIndexChanged(int)), this, SLOT(emitChange(int)));
		QObject::connect(mNameInput2, SIGNAL(currentIndexChanged(int)), this, SLOT(emitChange(int)));
		QObject::connect(mUseMoter, SIGNAL(stateChanged(int)), this, SLOT(emitChange(int)));
		QObject::connect(mMoterInput, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double value)
			{
				int intValue = static_cast<int>(value);

				emitChange(intValue);
			});
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
		delete mUseMoter;
		delete mMoterInput;
		delete mEditButton;
		delete mRemoveButton;
		for (auto it : mDetailWidgets)
		{
			if (it)
				it->close();
		}
		mDetailWidgets.clear();
	}

	VehicleJointInfo mJointItemLayout::value()
	{
		//jointInfo.Joint_Actor = ActorId;
		mJointInfo.JointName1.name = mNameInput1->currentText().toStdString();
		mJointInfo.JointName2.name = mNameInput2->currentText().toStdString();

		mJointInfo.type = mVecJointType[mTypeInput->currentIndex()];
		mJointInfo.useMoter = mUseMoter->isChecked();
		mJointInfo.v_moter = mMoterInput->value();


		return mJointInfo;
	}

	void mJointItemLayout::setValue(const VehicleJointInfo& v)
	{
		mName1_ObjID = (v.JointName1.rigidId);
		mName2_ObjID = (v.JointName2.rigidId);

		//Type
		mUseMoter->setChecked(v.useMoter);
		mJointInfo.useRange = v.useRange;
		mJointInfo.anchorPoint = v.anchorPoint;
		mJointInfo.d_min = v.d_min;
		mJointInfo.d_max = v.d_max;
		mMoterInput->setValue(v.v_moter);
		mJointInfo.Axis = v.Axis;
	}

	
	void  mJointItemLayout::createJointDetailWidget()
	{
		auto detail = new QJointBodyDetail(this->mJointInfo);
		detail->show();
		QObject::connect(detail, QOverload<>::of(&QJointBodyDetail::jointChange), [=]() {emitChange(1); });

		mDetailWidgets.push_back(detail);
	}


	IMPL_FIELD_WIDGET(VehicleBind, QVehicleInfoWidget)

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

		QLabel* idLabel = new QLabel("<b>No.</b>",this);
		QLabel* rigidNameLabel = new QLabel("<b>Name</b>", this);
		QLabel* shapeIdLabel = new QLabel("<b>ShapeID</b>", this);
		QLabel* typeLabel = new QLabel("<b>Type</b>", this);
		QLabel* offsetLabel = new QLabel("<b>Edit</b>", this);

		QPushButton* addItembutton = new QPushButton("Add Item", this);
		addItembutton->setFixedSize(80, 30);

		nameLayout->addWidget(idLabel);
		nameLayout->addWidget(rigidNameLabel);
		nameLayout->addWidget(shapeIdLabel);
		nameLayout->addWidget(typeLabel);
		nameLayout->addWidget(offsetLabel);
		nameLayout->addWidget(addItembutton);

		RigidBodyUI->addLayout(nameLayout);
		mMainLayout->addLayout(RigidBodyUI);

		mRigidsLayout = new QVBoxLayout;
		mMainLayout->addLayout(mRigidsLayout);

		idLabel->setFixedWidth(25);
		rigidNameLabel->setFixedWidth(100);
		typeLabel->setFixedWidth(76);
		shapeIdLabel->setFixedWidth(76);
		offsetLabel->setFixedWidth(76);

		idLabel->setAlignment(Qt::AlignCenter);
		rigidNameLabel->setAlignment(Qt::AlignCenter);
		typeLabel->setAlignment(Qt::AlignCenter);
		shapeIdLabel->setAlignment(Qt::AlignCenter);
		offsetLabel->setAlignment(Qt::AlignCenter);

		QObject::connect(addItembutton, SIGNAL(pressed()), this, SLOT(addRigidBodyItemWidget()));
		QObject::connect(addItembutton, SIGNAL(pressed()), this, SLOT(updateJointComboBox()));

		//Joint UI
		auto jointUI = new QVBoxLayout;
		jointUI->setContentsMargins(0, 0, 0, 0);
		QHBoxLayout* jointLayout = new QHBoxLayout;

		QLabel* jointNumLabel = new QLabel("<b>No.</b>",this);
		QLabel* actor1 = new QLabel("<b>RigidBody1</b>",this);
		QLabel* actor2 = new QLabel("<b>RigidBody2</b>",this);
		QLabel* jointTypeLabel = new QLabel("<b>Type</b>",this);
		QLabel* moterLabel = new QLabel("<b>Moter</b>",this);
		QLabel* anchorOffsetLabel = new QLabel("<b>Edit</b>",this);

		QPushButton* addJointItembutton = new QPushButton("Add Item",this);
		addJointItembutton->setFixedSize(80, 30);

		jointLayout->addWidget(jointNumLabel);
		jointLayout->addWidget(actor1);
		jointLayout->addWidget(actor2);
		jointLayout->addWidget(jointTypeLabel);
		jointLayout->addWidget(moterLabel);
		jointLayout->addWidget(anchorOffsetLabel);
		jointLayout->addWidget(addJointItembutton);

		jointUI->addLayout(jointLayout);
		jointUI->setContentsMargins(0, 0, 0, 0);

		mMainLayout->addLayout(jointUI);
		mJointsLayout = new QVBoxLayout;
		mMainLayout->addLayout(mJointsLayout);

		jointNumLabel->setFixedWidth(25);
		actor1->setFixedWidth(90);
		actor2->setFixedWidth(90);
		jointTypeLabel->setFixedWidth(65);
		moterLabel->setFixedWidth(65);
		anchorOffsetLabel->setFixedWidth(50);

		jointNumLabel->setAlignment(Qt::AlignCenter);
		actor1->setAlignment(Qt::AlignCenter);
		actor2->setAlignment(Qt::AlignCenter);
		jointTypeLabel->setAlignment(Qt::AlignCenter);
		moterLabel->setAlignment(Qt::AlignCenter);
		anchorOffsetLabel->setAlignment(Qt::AlignCenter);

		QObject::connect(addJointItembutton, SIGNAL(pressed()), this, SLOT(addJointItemWidget()));
		QObject::connect(this, SIGNAL(vectorChange()), this, SLOT(updateField()));

		FVar<VehicleBind>* f = TypeInfo::cast<FVar<VehicleBind>>(field);
		if (f != nullptr)
		{
			mVec = f->getValue();
		}

		updateWidget();

	};

	void QVehicleInfoWidget::updateField()
	{
		FVar<VehicleBind>* f = TypeInfo::cast<FVar<VehicleBind>>(field());
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
		for (size_t i = 0; i < mVec.vehicleRigidBodyInfo.size(); i++)
		{
			createItemWidget(mVec.vehicleRigidBodyInfo[i]);
		}

		bulidQueryMap();

		for (size_t i = 0; i < mVec.vehicleJointInfo.size(); i++)
		{
			createJointItemWidget(mVec.vehicleJointInfo[i]);
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

		mVec.vehicleRigidBodyInfo.clear();
		for (size_t i = 0; i < mRigidBodyItems.size(); i++)
		{
			mVec.vehicleRigidBodyInfo.push_back(mRigidBodyItems[i]->value());
		}

		//*******************************  bulidQueryMap  *******************************//
		bulidQueryMap();

		//*******************************  UpdateData  *******************************//
		mVec.vehicleJointInfo.clear();

		//update Rigid ID
		for (size_t i = 0; i < mJointItems.size(); i++)
		{
			auto& jointItem = mJointItems[i];
			mVec.vehicleJointInfo.push_back(jointItem->value());
			auto& jointInfo = mVec.vehicleJointInfo[i];

			if (jointInfo.JointName1.name != std::string(""))
				jointInfo.JointName1.rigidId = mName2RigidId[jointInfo.JointName1.name];

			if (jointInfo.JointName2.name != std::string(""))
				jointInfo.JointName2.rigidId = mName2RigidId[jointInfo.JointName2.name];
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
		addRigidBodyItemWidgetByID(mRigidCounter);
		mRigidCounter++;
	}

	void QVehicleInfoWidget::addRigidBodyItemWidgetByID(int objId)
	{
		RigidBodyItemLayout* itemLayout = new RigidBodyItemLayout(mRigidBodyItems.size());
		itemLayout->setObjId(objId);

		connectRigidWidgetSignal(itemLayout);

		mRigidsLayout->addLayout(itemLayout);
		mRigidBodyItems.push_back(itemLayout);

		updateVector();
	}


	void QVehicleInfoWidget::removeRigidBodyItemWidgetById(int id)
	{
		mRigidsLayout->removeItem(mRigidBodyItems[id]);
		delete mRigidBodyItems[id];
		mRigidBodyItems.erase(mRigidBodyItems.begin() + id);
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

	void QVehicleInfoWidget::createItemWidget(const VehicleRigidBodyInfo& rigidBody)
	{		
		RigidBodyItemLayout* itemLayout = new RigidBodyItemLayout(mRigidBodyItems.size());
		itemLayout->setObjId(mRigidCounter);
		mRigidCounter++;

		itemLayout->setValue(rigidBody);
		connectRigidWidgetSignal(itemLayout);
		mRigidsLayout->addLayout(itemLayout);
		mRigidBodyItems.push_back(itemLayout);
	
	}

	void QVehicleInfoWidget::createJointItemWidget(const VehicleJointInfo& rigidBody)
	{
		mJointItemLayout* itemLayout = new mJointItemLayout(mJointItems.size());
		itemLayout->setValue(rigidBody);
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
		QObject::connect(itemLayout, SIGNAL(nameChange(int)), this, SLOT(updateJointComboBox()));
		QObject::connect(itemLayout, SIGNAL(valueChange(int)), this, SLOT(updateVector()));
		QObject::connect(itemLayout->mNameInput, SIGNAL(editingFinished()), this, SLOT(bulidQueryMap()));
	}


}


