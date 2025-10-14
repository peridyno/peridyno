#include "QVectorTransform3FieldWidget.h"

#include <QGridLayout>

#include "Field.h"
#include "QPiecewiseSpinBox.h"

#include <QPushButton>

namespace dyno
{

	IMPL_FIELD_WIDGET(std::vector<Transform3f>, QVectorTransform3FieldWidget)

		QVectorTransform3FieldWidget::QVectorTransform3FieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		mMainLayout = new QVBoxLayout;
		mMainLayout->setContentsMargins(0, 0, 0, 0);
		mMainLayout->setAlignment(Qt::AlignLeft);

		this->setLayout(mMainLayout);

		//Label
		QHBoxLayout* nameLayout = new QHBoxLayout;
		QLabel* name = new QLabel();
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		QPushButton* addItembutton = new QPushButton("add Item",this);
		addItembutton->setFixedSize(100, 40);

		nameLayout->addWidget(name);
		nameLayout->addWidget(addItembutton);

		mMainLayout->addLayout(nameLayout);


		QObject::connect(addItembutton, SIGNAL(pressed()), this, SLOT(addItemWidget()));
		QObject::connect(this, SIGNAL(vectorChange()), this, SLOT(updateField()));

		FVar<std::vector<Transform3f>>* f = TypeInfo::cast<FVar<std::vector<Transform3f>>>(field);
		if (f != nullptr)
		{
			mVec = f->getValue();
		}

		updateWidget();

	};

	void QVectorTransform3FieldWidget::updateWidget()
	{
		for (size_t i = 0; i < mVec.size(); i++)
		{
			createItemWidget(mVec[i]);
		}

	}

	void QVectorTransform3FieldWidget::updateField()
	{
		FVar<std::vector<Transform3f>>* f = TypeInfo::cast<FVar<std::vector<Transform3f>>>(field());
		if (f != nullptr)
		{
			f->setValue(mVec);
		}
	};

	void QVectorTransform3FieldWidget::updateVector()
	{
		mVec.clear();
		for (size_t i = 0; i < mItems.size(); i++)
		{
			mVec.push_back(mItems[i]->value());
		}
		emit vectorChange();
	}

	void QVectorTransform3FieldWidget::addItemWidget()
	{
		mVectorTransformItemLayout* itemLayout = new mVectorTransformItemLayout(mItems.size());
		QObject::connect(itemLayout, SIGNAL(removeById(int)), this, SLOT(removeItemWidgetById(int)));
		QObject::connect(itemLayout, SIGNAL(valueChange(double)), this, SLOT(updateVector()));

		mMainLayout->addLayout(itemLayout);
		mItems.push_back(itemLayout);

		updateVector();
	}

	void QVectorTransform3FieldWidget::removeItemWidgetById(int id)
	{
		mMainLayout->removeItem(mItems[id]);
		delete mItems[id];
		mItems.erase(mItems.begin() + id);
		for (size_t i = 0; i < mItems.size(); i++)
		{
			mItems[i]->setId(i);
		}

		mMainLayout->update();

		updateVector();
	}

	void QVectorTransform3FieldWidget::createItemWidget(Transform3f v)
	{	
		mVectorTransformItemLayout* itemLayout = new mVectorTransformItemLayout(mItems.size());
		itemLayout->setValue(v);

		QObject::connect(itemLayout, SIGNAL(removeById(int)), this, SLOT(removeItemWidgetById(int)));
		QObject::connect(itemLayout, SIGNAL(valueChange(double)), this, SLOT(updateVector()));

		mMainLayout->addLayout(itemLayout);
		mItems.push_back(itemLayout);	
	}

	//************************ mVectorTransformItemLayout ***************************//
	mVectorTransformItemLayout::mVectorTransformItemLayout(int id)
	{
		mId = id;

		mGroup = new QGroupBox;
		this->addWidget(mGroup);

		index = new QLabel(std::to_string(id).c_str());

		mT0 = new QPiecewiseDoubleSpinBox;
		mT1 = new QPiecewiseDoubleSpinBox;
		mT2 = new QPiecewiseDoubleSpinBox;

		mT0->setRealValue(0);
		mT0->setRealValue(0);
		mT0->setRealValue(0);

		mR0 = new QPiecewiseDoubleSpinBox;
		mR1 = new QPiecewiseDoubleSpinBox;
		mR2 = new QPiecewiseDoubleSpinBox;

		mR0->setRealValue(0);
		mR0->setRealValue(0);
		mR0->setRealValue(0);

		mS0 = new QPiecewiseDoubleSpinBox;
		mS1 = new QPiecewiseDoubleSpinBox;
		mS2 = new QPiecewiseDoubleSpinBox;

		mS0->setRealValue(1);
		mS1->setRealValue(1);
		mS2->setRealValue(1);

		QHBoxLayout* layout_T = new QHBoxLayout();
		QHBoxLayout* layout_R = new QHBoxLayout();
		QHBoxLayout* layout_S = new QHBoxLayout();
		layout_T->setSpacing(5);
		layout_R->setSpacing(5);
		layout_S->setSpacing(5);

		int width = 90;
		mTLabel = new QLabel("Location");
		layout_T->addWidget(mTLabel);
		layout_T->addWidget(mT0);
		layout_T->addWidget(mT1);
		layout_T->addWidget(mT2);

		mRLabel = new QLabel("Rotation");
		layout_R->addWidget(mRLabel);
		layout_R->addWidget(mR0);
		layout_R->addWidget(mR1);
		layout_R->addWidget(mR2);

		mSLabel = new QLabel("Scale");
		layout_S->addWidget(mSLabel);
		layout_S->addWidget(mS0);
		layout_S->addWidget(mS1);
		layout_S->addWidget(mS2);

		mTLabel->setMinimumWidth(width);
		mRLabel->setMinimumWidth(width);
		mSLabel->setMinimumWidth(width);
		mT0->setMinimumWidth(width);
		mT1->setMinimumWidth(width);
		mT2->setMinimumWidth(width);
		mR0->setMinimumWidth(width);
		mR1->setMinimumWidth(width);
		mR2->setMinimumWidth(width);
		mS0->setMinimumWidth(width);
		mS1->setMinimumWidth(width);
		mS2->setMinimumWidth(width);

		QVBoxLayout* list = new QVBoxLayout();
		list->setSpacing(5);

		list->addItem(layout_T);
		list->addItem(layout_R);
		list->addItem(layout_S);

		QHBoxLayout* groupLayout = new QHBoxLayout;
		mGroup->setLayout(groupLayout);

		groupLayout->addWidget(index);
		groupLayout->addLayout(list);

		removeButton = new QPushButton("Delete");
		groupLayout->addWidget(removeButton, 0);
		removeButton->setFixedSize(100, 40);

		QObject::connect(removeButton, SIGNAL(pressed()), this, SLOT(emitSignal()));
		QObject::connect(mT0, SIGNAL(editingFinishedWithValue(double)), this, SLOT(emitChange(double)));
		QObject::connect(mT1, SIGNAL(editingFinishedWithValue(double)), this, SLOT(emitChange(double)));
		QObject::connect(mT2, SIGNAL(editingFinishedWithValue(double)), this, SLOT(emitChange(double)));
		QObject::connect(mR0, SIGNAL(editingFinishedWithValue(double)), this, SLOT(emitChange(double)));
		QObject::connect(mR1, SIGNAL(editingFinishedWithValue(double)), this, SLOT(emitChange(double)));
		QObject::connect(mR2, SIGNAL(editingFinishedWithValue(double)), this, SLOT(emitChange(double)));
		QObject::connect(mS0, SIGNAL(editingFinishedWithValue(double)), this, SLOT(emitChange(double)));
		QObject::connect(mS1, SIGNAL(editingFinishedWithValue(double)), this, SLOT(emitChange(double)));
		QObject::connect(mS2, SIGNAL(editingFinishedWithValue(double)), this, SLOT(emitChange(double)));

	};

	mVectorTransformItemLayout::~mVectorTransformItemLayout()
	{
		delete mT0;
		delete mT1;
		delete mT2;

		delete mR0;
		delete mR1;
		delete mR2;

		delete mS0;
		delete mS1;
		delete mS2;

		delete mTLabel;
		delete mRLabel;
		delete mSLabel;

		delete removeButton;
		delete index;
		delete mGroup;
	};

	Transform3f mVectorTransformItemLayout::value()
	{
		auto rot = Vec3f(mR0->getRealValue(), mR1->getRealValue(), mR2->getRealValue());

		Quat<Real> q =
			Quat<Real>(Real(M_PI) * rot[2] / 180, Vec3f(0, 0, 1))
			* Quat<Real>(Real(M_PI) * rot[1] / 180, Vec3f(0, 1, 0))
			* Quat<Real>(Real(M_PI) * rot[0] / 180, Vec3f(1, 0, 0));
		q.normalize();

		Transform3f transform = Transform3f(
			Vec3f(mT0->getRealValue(), mT1->getRealValue(), mT2->getRealValue()),
			q.toMatrix3x3(),
			Vec3f(mS0->getRealValue(), mS1->getRealValue(), mS2->getRealValue())
		);

		return transform;
	}

	void mVectorTransformItemLayout::setValue(Transform3f v)
	{
		mT0->setRealValue(v.translation()[0]);
		mT1->setRealValue(v.translation()[1]);
		mT2->setRealValue(v.translation()[2]);

		Vec3f rot = Vec3f(0);

		Quat<Real>(v.rotation()).toEulerAngle(rot[0], rot[1], rot[2]);
		
		mR0->setRealValue(rot[2] * 180 / M_PI);
		mR1->setRealValue(rot[1] * 180 / M_PI);
		mR2->setRealValue(rot[0] * 180 / M_PI);

		mS0->setRealValue(v.scale()[0]);
		mS1->setRealValue(v.scale()[1]);
		mS2->setRealValue(v.scale()[2]);

	}





}

