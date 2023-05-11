#include "QFieldWidget.h"

#include "Module.h"
#include "Node.h"
#include "Field.h"
#include "SceneGraphFactory.h"

#include <QVBoxLayout>

#include <memory>

namespace dyno
{
	QFieldWidget::QFieldWidget(FBase* field)
	{
		mField = field;

		if (mField != nullptr)
		{
			callback = std::make_shared<FCallBackFunc>(std::bind(&QFieldWidget::syncValueFromField, this));
			mField->attach(callback);
		}
	}

	QFieldWidget::~QFieldWidget()
	{
		if (mField != nullptr) {
			mField->detach(callback);
		}
	}

	void QFieldWidget::syncValueFromField()
	{
		auto node = dynamic_cast<Node*>(mField->parent());
		if (node != nullptr) {
			node->updateGraphicsContext();
		}

		emit fieldChanged();
	}

	mDoubleSpinBox::mDoubleSpinBox(QWidget* parent)
		: QDoubleSpinBox(parent)
	{
		this->lineEdit()->setMouseTracking(true);
	}
	void mDoubleSpinBox::wheelEvent(QWheelEvent* event)
	{

	}
	void mDoubleSpinBox::contextMenuEvent(QContextMenuEvent* event) 
	{
		buildDialog();

	}
	void mDoubleSpinBox::buildDialog() 
	{
		ValueModify = new ValueDialog(this->value());

		ValueModify->SpinBox1 = this->DSB1;
		ValueModify->SpinBox2 = this->DSB2;
		ValueModify->SpinBox3 = this->DSB3;
		for (size_t i = 0; i < 5; i++)
		{
			ValueModify->button[i]->DSB1 = DSB1;
			ValueModify->button[i]->DSB2 = DSB2;
			ValueModify->button[i]->DSB3 = DSB3;

			ValueModify->button[i]->Data1 = DSB1->value();
			ValueModify->button[i]->Data2 = DSB2->value();
			ValueModify->button[i]->Data3 = DSB3->value();
		}
		connect(ValueModify, SIGNAL(DiaValueChange(double)), this, SLOT(ModifyValue(double)));
	}
	void mDoubleSpinBox::mousePressEvent(QMouseEvent* event)

	{
		QDoubleSpinBox::mousePressEvent(event);
		if (event->button() == Qt::RightButton) {

			buildDialog();
		}


	}

	void mDoubleSpinBox::mouseReleaseEvent(QMouseEvent* event)
	{
		QDoubleSpinBox::mouseReleaseEvent(event);
	}

	void mDoubleSpinBox::mouseMoveEvent(QMouseEvent* event)
	{
		QDoubleSpinBox::mouseMoveEvent(event);

	}

	void mDoubleSpinBox::ModifyValue(double v)
	{

		this->setValue(v);
	}

	ValueDialog::ValueDialog(double Data, QWidget* parent) :
		QDialog(parent)
	{
		//¹¹½¨²Ëµ¥
		QVBoxLayout* VLayout = new QVBoxLayout;
		float power = 0.1;

		for (int i = 0; i < 5; i++)
		{
			button[i] = new ValueButton;

			power *= 0.1;

			std::string s = std::to_string(power * 1000);
			QString text = QString::fromStdString(s);

			button[i]->setText(text);
			button[i]->setFixedWidth(200);
			button[i]->setFixedHeight(40);
			button[i]->adjustSize();
			//button[i]->setAlignment(Qt::AlignHCenter| Qt::AlignVCenter);
			button[i]->setStyleSheet("QLabel{color:white;background-color:#346792;border: 1px solid #000000;border-radius:3px; padding: 0px;}");
			button[i]->StartX = QCursor().pos().x();
			button[i]->defaultValue = power * 1000;
			button[i]->SpinBoxData = Data;
			button[i]->parentDialog = this;

			VLayout->addWidget(button[i]);

			connect(button[i], SIGNAL(ValueChange(double)), this, SLOT(ModifyValue(double)));
			connect(button[i], SIGNAL(Release(double)), this, SLOT(initData(double)));
		}
		VLayout->setSpacing(0);

		this->setLayout(VLayout);
		this->setWindowFlags(Qt::WindowStaysOnTopHint | Qt::WindowCloseButtonHint | Qt::Popup);
		this->move(QCursor().pos().x() - button[1]->rect().width() / 2, QCursor().pos().y() - button[1]->rect().height() * 5 / 2 - 5);

		this->setMouseTracking(true);
		this->hasMouseTracking();
		this->setAttribute(Qt::WA_Hover, true);

		this->setWindowTitle("Property Editor");

		this->show();

	}
	void ValueDialog::mouseReleaseEvent(QMouseEvent* event)
	{
		//this->close();
	}

	void ValueDialog::ModifyValue(double v)
	{
		emit DiaValueChange(v);
	}
	void  ValueDialog::keyPressEvent(QKeyEvent* event)
	{
		QDialog::keyPressEvent(event);
		if (event->key() == Qt::Key_Shift)
		{
			for (size_t i = 0; i < 5; i++)
			{
				this->button[i]->shiftPress = true;
			}
		}
	}
	void  ValueDialog::keyReleaseEvent(QKeyEvent* event)
	{
		printf("dialog\n");
		QDialog::keyReleaseEvent(event);
		if (event->key() == Qt::Key_Shift)
		{
			for (size_t i = 0; i < 5; i++)
			{
				this->button[i]->shiftPress = false;
			}
		}
	}

	void ValueDialog::initData(double v)
	{
		for (int i = 0; i < 5; i++)
		{
			button[i]->SpinBoxData = v;

			button[i]->Data1 = SpinBox1->value();
			button[i]->Data2 = SpinBox2->value();
			button[i]->Data3 = SpinBox3->value();
		}

	}

	void ValueDialog::mouseMoveEvent(QMouseEvent* event)
	{

	}

	void ValueButton::mouseMoveEvent(QMouseEvent* event)
	{
		EndX = QCursor().pos().x();
		temp = (EndX - StartX) / 10;
		sub = defaultValue * temp;

		str = std::to_string(sub);
		text = QString::fromStdString(str);
		this->setText(text);
		
		if (shiftPress)
		{
			double p = (SpinBoxData+sub)/SpinBoxData;
			
			double d1 = DSB1->value();
			double d2 = DSB2->value();
			double d3 = DSB3->value();

			DSB1->setValue(Data1 * p);
			DSB2->setValue(Data2 * p);
			DSB3->setValue(Data3 * p);
		}
		emit ValueChange(SpinBoxData + sub);
		SceneGraphFactory::instance()->active()->reset();

	}

	ValueButton::ValueButton(QWidget* parent) :
		QPushButton(parent)
	{

	}

	void ValueButton::mousePressEvent(QMouseEvent* event)
	{
		StartX = QCursor().pos().x();
	}
	void ValueButton::mouseReleaseEvent(QMouseEvent* event)
	{
		str = std::to_string(defaultValue);
		text = QString::fromStdString(str);
		this->setText(text);
		SpinBoxData = SpinBoxData + sub;

		emit Release(SpinBoxData);

	}
}

