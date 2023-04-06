#include "PPropertyItem.h"
#include "Module.h"
#include "Node.h"
#include "FilePath.h"
#include "SceneGraphFactory.h"
#include "Ramp.h"

#include "Common.h"
#include "PCustomWidgets.h"

#include <QGroupBox>
#include <QLabel>
#include <QComboBox>
#include <QCheckBox>
#include <QPushButton>

#include <QSlider>
#include <QSpinBox>
#include <QRegularExpression>
#include <QMessageBox>

#include <QDoubleValidator>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QMessageBox>
#include <QFileDialog>
#include <QFile>
#include "qpainter.h"
#include <iostream>     
#include <algorithm>    
#include <vector>       
#include <algorithm>
#include "Math/SimpleMath.h"

namespace dyno
{
	QBoolFieldWidget::QBoolFieldWidget(FBase* field)
		: QGroupBox()
	{
		mField = field;
		FVar<bool>* f = TypeInfo::cast<FVar<bool>>(mField);
		if (f == nullptr) {
			return;
		}

		//this->setStyleSheet("border:none");
		QHBoxLayout* layout = new QHBoxLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedHeight(24);
		name->setText(FormatFieldWidgetName(field->getObjectName()));
		QCheckBox* checkbox = new QCheckBox();
		checkbox->setFixedWidth(20);
		//checkbox->setFixedSize(40, 18);
		layout->addWidget(name, 0);
		layout->addStretch(1);
		layout->addWidget(checkbox, 0);

		connect(checkbox, SIGNAL(stateChanged(int)), this, SLOT(changeValue(int)));

		checkbox->setChecked(f->getData());
	}

	void QBoolFieldWidget::changeValue(int status)
	{
		FVar<bool>* f = TypeInfo::cast<FVar<bool>>(mField);
		if (f == nullptr)
		{
			return;
		}

		if (status == Qt::Checked)
		{
			f->setValue(true);
			f->update();
		}
		else if (status == Qt::PartiallyChecked)
		{
			//m_pLabel->setText("PartiallyChecked");
		}
		else
		{
			f->setValue(false);
			f->update();
		}

		emit fieldChanged();
	}

	QIntegerFieldWidget::QIntegerFieldWidget(FBase* field)
		: QGroupBox()
	{
		mField = field;
		FVar<int>* f = TypeInfo::cast<FVar<int>>(mField);
		if (f == nullptr)
		{
			return;
		}

		//this->setStyleSheet("border:none");
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(100, 18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		QSpinBox* spinner = new QSpinBox;
		spinner->setValue(f->getData());

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinner, 0, 1, Qt::AlignRight);


		this->connect(spinner, SIGNAL(valueChanged(int)), this, SLOT(changeValue(int)));
	}

	void QIntegerFieldWidget::changeValue(int value)
	{
		FVar<int>* f = TypeInfo::cast<FVar<int>>(mField);
		if (f == nullptr)
			return;

		f->setValue(value);
		f->update();

		emit fieldChanged();
	}

	QUIntegerFieldWidget::QUIntegerFieldWidget(FBase* field)
		: QGroupBox()
	{
		mField = field;
		FVar<uint>* f = TypeInfo::cast<FVar<uint>>(mField);
		if (f == nullptr)
		{
			return;
		}
		//this->setStyleSheet("border:none");
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(100, 18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		QSpinBox* spinner = new QSpinBox;
		spinner->setMaximum(1000000);
		spinner->setValue(f->getData());

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinner, 0, 1, Qt::AlignRight);

		this->connect(spinner, SIGNAL(valueChanged(int)), this, SLOT(changeValue(int)));
	}

	void QUIntegerFieldWidget::changeValue(int value)
	{
		FVar<uint>* f = TypeInfo::cast<FVar<uint>>(mField);
		if (f == nullptr)
			return;

		f->setValue(value);
		f->update();

		emit fieldChanged();
	}

	QRealFieldWidget::QRealFieldWidget(FBase* field)
		: QGroupBox()
	{
		mField = field;
		
		//this->setStyleSheet("border:none");
		QHBoxLayout* layout = new QHBoxLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedHeight(24);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		slider = new QDoubleSlider;
		slider->setRange(mField->getMin(), mField->getMax());
		slider->setMinimumWidth(60);

		QDoubleSpinner* spinner = new QDoubleSpinner;
		spinner->setRange(mField->getMin(), mField->getMax());
		spinner->setFixedWidth(80);
		spinner->setDecimals(3);

		layout->addWidget(name, 0);
		layout->addWidget(slider, 1);
		layout->addStretch();
		layout->addWidget(spinner, 2);

		std::string template_name = field->getTemplateName();
		if (template_name == std::string(typeid(float).name()))
		{
			FVar<float>* f = TypeInfo::cast<FVar<float>>(mField);
			slider->setValue((double)f->getValue());
			spinner->setValue((double)f->getValue());
		}
		else if(template_name == std::string(typeid(double).name()))
		{
			FVar<double>* f = TypeInfo::cast<FVar<double>>(mField);
			slider->setValue(f->getValue());
			spinner->setValue((double)f->getValue());
		}

		FormatFieldWidgetName(field->getObjectName());

		QObject::connect(slider, SIGNAL(valueChanged(double)), spinner, SLOT(setValue(double)));
		QObject::connect(spinner, SIGNAL(valueChanged(double)), slider, SLOT(setValue(double)));
		QObject::connect(spinner, SIGNAL(valueChanged(double)), this, SLOT(changeValue(double)));

// 		if (mField != nullptr)
// 		{
// 			callback = std::make_shared<FCallBackFunc>(std::bind(&QRealFieldWidget::fieldUpdated, this));
// 			mField->attach(callback);
// 		}
	}

	QRealFieldWidget::~QRealFieldWidget()
	{
		if (mField != nullptr)
		{
			mField->detach(callback);
		}
	}

	void QRealFieldWidget::changeValue(double value)
	{
		std::string template_name = mField->getTemplateName();

		if (template_name == std::string(typeid(float).name()))
		{
			FVar<float>* f = TypeInfo::cast<FVar<float>>(mField);
			f->setValue((float)value);
			f->update();
		}
		else if (template_name == std::string(typeid(double).name()))
		{
			FVar<double>* f = TypeInfo::cast<FVar<double>>(mField);
			f->setValue(value);
			f->update();
		}

		emit fieldChanged();
	}

	void QRealFieldWidget::fieldUpdated()
	{
		std::string template_name = mField->getTemplateName();
		if (template_name == std::string(typeid(float).name()))
		{
			FVar<float>* f = TypeInfo::cast<FVar<float>>(mField);

			slider->blockSignals(true);
			slider->setValue((double)f->getValue());
			slider->blockSignals(false);
		}
		else if (template_name == std::string(typeid(double).name()))
		{
			FVar<double>* f = TypeInfo::cast<FVar<double>>(mField);

			slider->blockSignals(true);
			slider->setValue(f->getValue());
			slider->blockSignals(false);
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

	QVector3FieldWidget::QVector3FieldWidget(FBase* field)
		: QGroupBox()
	{
		mField = field;

		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(100, 18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		spinner1 = new mDoubleSpinBox;
		spinner1->setMinimumWidth(30);
		spinner1->setRange(mField->getMin(), mField->getMax());

		spinner2 = new mDoubleSpinBox;
		spinner2->setMinimumWidth(30);
		spinner2->setRange(mField->getMin(), mField->getMax());

		spinner3 = new mDoubleSpinBox;
		spinner3->setMinimumWidth(30);
		spinner3->setRange(mField->getMin(), mField->getMax());

		spinner1->DSB1 = spinner1;
		spinner1->DSB2 = spinner2;
		spinner1->DSB3 = spinner3;
		spinner2->DSB1 = spinner1;
		spinner2->DSB2 = spinner2;
		spinner2->DSB3 = spinner3;
		spinner3->DSB1 = spinner1;
		spinner3->DSB2 = spinner2;
		spinner3->DSB3 = spinner3;

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinner1, 0, 1);
		layout->addWidget(spinner2, 0, 2);
		layout->addWidget(spinner3, 0, 3);


		std::string template_name = mField->getTemplateName();

		double v1 = 0;
		double v2 = 0;
		double v3 = 0;

		if (template_name == std::string(typeid(Vec3f).name()))
		{
			FVar<Vec3f>* f = TypeInfo::cast<FVar<Vec3f>>(mField);
			auto v = f->getData();
			v1 = v[0];
			v2 = v[1];
			v3 = v[2];
		}
		else if (template_name == std::string(typeid(Vec3d).name()))
		{
			FVar<Vec3d>* f = TypeInfo::cast<FVar<Vec3d>>(mField);
			auto v = f->getData();

			v1 = v[0];
			v2 = v[1];
			v3 = v[2];
		}

		spinner1->setValue(v1);
		spinner2->setValue(v2);
		spinner3->setValue(v3);

		QObject::connect(spinner1, SIGNAL(valueChanged(double)), this, SLOT(changeValue(double)));
		QObject::connect(spinner2, SIGNAL(valueChanged(double)), this, SLOT(changeValue(double)));
		QObject::connect(spinner3, SIGNAL(valueChanged(double)), this, SLOT(changeValue(double)));

		if (mField != nullptr)
		{
			callback = std::make_shared<FCallBackFunc>(std::bind(&QVector3FieldWidget::fieldUpdated, this));
			mField->attach(callback);
		}
	}

	QVector3FieldWidget::~QVector3FieldWidget()
	{
		if (mField != nullptr) {
			mField->detach(callback);
		}
	}

	void QVector3FieldWidget::changeValue(double value)
	{
		double v1 = spinner1->value();
		double v2 = spinner2->value();
		double v3 = spinner3->value();

		std::string template_name = mField->getTemplateName();

		if (template_name == std::string(typeid(Vec3f).name()))
		{
			FVar<Vec3f>* f = TypeInfo::cast<FVar<Vec3f>>(mField);
			f->setValue(Vec3f((float)v1, (float)v2, (float)v3));
		}
		else if (template_name == std::string(typeid(Vec3d).name()))
		{
			FVar<Vec3d>* f = TypeInfo::cast<FVar<Vec3d>>(mField);
			f->setValue(Vec3d(v1, v2, v3));
		}

		emit fieldChanged();
	}


	void QVector3FieldWidget::fieldUpdated()
	{
		std::string template_name = mField->getTemplateName();

		double v1 = 0;
		double v2 = 0;
		double v3 = 0;

		if (template_name == std::string(typeid(Vec3f).name()))
		{
			FVar<Vec3f>* f = TypeInfo::cast<FVar<Vec3f>>(mField);
			auto v = f->getData();
			v1 = v[0];
			v2 = v[1];
			v3 = v[2];
		}
		else if (template_name == std::string(typeid(Vec3d).name()))
		{
			FVar<Vec3d>* f = TypeInfo::cast<FVar<Vec3d>>(mField);
			auto v = f->getData();

			v1 = v[0];
			v2 = v[1];
			v3 = v[2];
		}

		spinner1->blockSignals(true);
		spinner2->blockSignals(true);
		spinner3->blockSignals(true);

		spinner1->setValue(v1);
		spinner2->setValue(v2);
		spinner3->setValue(v3);

		spinner1->blockSignals(false);
		spinner2->blockSignals(false);
		spinner3->blockSignals(false);

		emit fieldChanged();
	}

	QVector3iFieldWidget::QVector3iFieldWidget(FBase* field)
	{
		mField = field;

		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(100, 18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));
		name->setWordWrap(true);

		spinner1 = new QSpinBox;
		spinner1->setMinimumWidth(30);
		spinner1->setRange(mField->getMin(), mField->getMax());

		spinner2 = new QSpinBox;
		spinner2->setMinimumWidth(30);
		spinner2->setRange(mField->getMin(), mField->getMax());

		spinner3 = new QSpinBox;
		spinner3->setMinimumWidth(30);
		spinner3->setRange(mField->getMin(), mField->getMax());

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinner1, 0, 1);
		layout->addWidget(spinner2, 0, 2);
		layout->addWidget(spinner3, 0, 3);


		std::string template_name = mField->getTemplateName();

		int v1 = 0;
		int v2 = 0;
		int v3 = 0;

		if (template_name == std::string(typeid(Vec3i).name()))
		{
			FVar<Vec3i>* f = TypeInfo::cast<FVar<Vec3i>>(mField);
			auto v = f->getData();
			v1 = v[0];
			v2 = v[1];
			v3 = v[2];
		}
		else if (template_name == std::string(typeid(Vec3u).name()))
		{
			FVar<Vec3u>* f = TypeInfo::cast<FVar<Vec3u>>(mField);
			auto v = f->getData();
			v1 = v[0];
			v2 = v[1];
			v3 = v[2];
		}

		spinner1->setValue(v1);
		spinner2->setValue(v2);
		spinner3->setValue(v3);

		QObject::connect(spinner1, SIGNAL(valueChanged(int)), this, SLOT(changeValue(int)));
		QObject::connect(spinner2, SIGNAL(valueChanged(int)), this, SLOT(changeValue(int)));
		QObject::connect(spinner3, SIGNAL(valueChanged(int)), this, SLOT(changeValue(int)));


// 		if (mField != nullptr)
// 		{
// 			callback = std::make_shared<FCallBackFunc>(std::bind(&QVector3iFieldWidget::fieldUpdated, this));
// 			mField->attach(callback);
// 		}
	}


	QVector3iFieldWidget::~QVector3iFieldWidget()
	{
		if (mField != nullptr) {
			mField->detach(callback);
		}
	}

	void QVector3iFieldWidget::changeValue(int)
	{
		int v1 = spinner1->value();
		int v2 = spinner2->value();
		int v3 = spinner3->value();

		std::string template_name = mField->getTemplateName();

		if (template_name == std::string(typeid(Vec3i).name()))
		{
			FVar<Vec3i>* f = TypeInfo::cast<FVar<Vec3i>>(mField);
			f->setValue(Vec3i(v1, v2, v3));
			f->update();
		}
		else if (template_name == std::string(typeid(Vec3u).name()))
		{
			FVar<Vec3u>* f = TypeInfo::cast<FVar<Vec3u>>(mField);
			f->setValue(Vec3u(v1, v2, v3));
			f->update();
		}

		emit fieldChanged();
	}

	void QVector3iFieldWidget::fieldUpdated()
	{
		std::string template_name = mField->getTemplateName();

		int v1 = 0;
		int v2 = 0;
		int v3 = 0;

		if (template_name == std::string(typeid(Vec3i).name()))
		{
			FVar<Vec3i>* f = TypeInfo::cast<FVar<Vec3i>>(mField);
			auto v = f->getData();
			v1 = v[0];
			v2 = v[1];
			v3 = v[2];
		}
		else if (template_name == std::string(typeid(Vec3u).name()))
		{
			FVar<Vec3u>* f = TypeInfo::cast<FVar<Vec3u>>(mField);
			auto v = f->getData();
			v1 = v[0];
			v2 = v[1];
			v3 = v[2];
		}

		spinner1->blockSignals(true);
		spinner2->blockSignals(true);
		spinner3->blockSignals(true);

		spinner1->setValue(v1);
		spinner2->setValue(v2);
		spinner3->setValue(v3);

		spinner1->blockSignals(false);
		spinner2->blockSignals(false);
		spinner3->blockSignals(false);

		emit fieldChanged();
	}

	QStringFieldWidget::QStringFieldWidget(FBase* field)
		: QGroupBox()
	{
		mField = field;

		FVar<std::string>* f = TypeInfo::cast<FVar<std::string>>(field);

		//this->setStyleSheet("border:none");
		QHBoxLayout* layout = new QHBoxLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(150, 18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		fieldname = new QLineEdit;
		fieldname->setText(QString::fromStdString(f->getValue()));

		layout->addWidget(name, 0);
		layout->addWidget(fieldname, 1);
		layout->setSpacing(5);

		connect(fieldname, &QLineEdit::textChanged, this, &QStringFieldWidget::changeValue);
	}

	void QStringFieldWidget::changeValue(QString str)
	{
		auto f = TypeInfo::cast<FVar<std::string>>(mField);
		if (f == nullptr)
		{
			return;
		}
		f->setValue(str.toStdString());
		f->update();

		emit fieldChanged();
	}

	QFilePathWidget::QFilePathWidget(FBase* field)
		: QGroupBox()
	{
		mField = field;

		FVar<FilePath>* f = TypeInfo::cast<FVar<FilePath>>(field);

		//this->setStyleSheet("border:none");
		QHBoxLayout* layout = new QHBoxLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(100, 18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		location = new QLineEdit;
		location->setText(QString::fromStdString(f->getValue().string()));

		QPushButton* open = new QPushButton("open");
// 		open->setStyleSheet("QPushButton{color: black;   border-radius: 10px;  border: 1px groove black;background-color:white; }"
// 							"QPushButton:hover{background-color:white; color: black;}"  
// 							"QPushButton:pressed{background-color:rgb(85, 170, 255); border-style: inset; }" );
		open->setFixedSize(60, 24);

		layout->addWidget(name, 0);
		layout->addWidget(location, 1);
		layout->addWidget(open, 0, 0);
		layout->setSpacing(5);

		connect(location, &QLineEdit::textChanged, this, &QFilePathWidget::changeValue);

		connect(open, &QPushButton::clicked, this, [=]() {
			QString path = QFileDialog::getOpenFileName(this, tr("Open File"), QString::fromStdString(getAssetPath()), tr("Text Files(*.*)"));
			if (!path.isEmpty()) {
				QFile file(path);
				if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
					QMessageBox::warning(this, tr("Read File"),
						tr("Cannot open file:\n%1").arg(path));
					return;
				}
				location->setText(path);
				file.close();
			}
			else {
				QMessageBox::warning(this, tr("Path"), tr("You do not select any file."));
			}
		});
	}

	void QFilePathWidget::changeValue(QString str)
	{
		auto f = TypeInfo::cast<FVar<FilePath>>(mField);
		if (f == nullptr)
		{
			return;
		}
		f->setValue(str.toStdString());
		f->update();

		emit fieldChanged();
	}

	QStateFieldWidget::QStateFieldWidget(FBase* field)
	{
		mField = field;

		//this->setStyleSheet("border:none");
		QHBoxLayout* layout = new QHBoxLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedHeight(24);
		name->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
		name->setText(FormatFieldWidgetName(field->getObjectName()));
		layout->addWidget(name, 0);
		layout->addStretch(1);

		QCheckBox* checkbox = new QCheckBox();		
		checkbox->setFixedWidth(20);
		layout->addWidget(checkbox, 0);

		if (mField->parent()->findOutputField(field))
		{
			checkbox->setChecked(true);;
		}
		else
		{
			checkbox->setChecked(false);
		}

		//TODO: use another way
		connect(checkbox, SIGNAL(stateChanged(int)), this, SLOT(tagAsOuput(int)));
	}

	void QStateFieldWidget::tagAsOuput(int status)
	{
		emit stateUpdated(mField, status);
	}

	QEnumFieldWidget::QEnumFieldWidget(FBase* field)
	{
		mField = field;

		auto f = TypeInfo::cast<FVar<PEnum>>(mField);
		if (f == nullptr || f->getDataPtr() == nullptr) {
			return;
		}

		QHBoxLayout* layout = new QHBoxLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		QLabel* name = new QLabel();
		name->setFixedHeight(24);
		name->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
		name->setText(FormatFieldWidgetName(field->getObjectName()));
		layout->addWidget(name, 0);

		QComboBox* combox = new QComboBox;
		combox->setMaximumWidth(256);

		auto& enums = f->getDataPtr()->enumMap();
		int num = 0;
		int curIndex = 0;
		for each(auto e in enums)
		{
			mComboxIndexMap[num] = e.first;
			combox->addItem(QString::fromStdString(e.second));

			if (e.first == f->getDataPtr()->currentKey()) {
				curIndex = num;
			}

			num++;
		}

		combox->setCurrentIndex(curIndex);

		layout->addWidget(combox, 1);

		this->setLayout(layout);

		connect(combox, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &QEnumFieldWidget::changeValue);
	}

	void QEnumFieldWidget::changeValue(int index)
	{
		auto f = TypeInfo::cast<FVar<PEnum>>(mField);
		if (f == nullptr || f->getDataPtr() == nullptr) {
			return;
		}

		auto& enums = f->getDataPtr()->enumMap();

		f->getDataPtr()->setCurrentKey(mComboxIndexMap[index]);
		//To notify the field is updated
		f->update();
	}


	ValueDialog::ValueDialog(double Data, QWidget* parent) :
		QDialog(parent)
	{
		//构建菜单
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


	

	QRampWidget::QRampWidget(FBase* field)
		: QGroupBox()
	{
		mField = field;
		FVar<Ramp>* f = TypeInfo::cast<FVar<Ramp>>(mField);
		if (f == nullptr)
		{
			printf("QRamp Nullptr\n");
			return;

		}

		//构建枚举列表
		int curIndex = int(f->getDataPtr()->mode);
		int enumNum = f->getDataPtr()->count;

		QComboBox* combox = new QComboBox;
		combox->setMaximumWidth(256);
		for (size_t i = 0; i < enumNum; i++)
		{
			auto enumName = f->getDataPtr()->DirectionStrings[i];

			combox->addItem(QString::fromStdString(enumName));
		}

		combox->setCurrentIndex(curIndex);

		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);
		this->setLayout(layout);

		QLabel* name = new QLabel();
		
		name->setFixedSize(80, 18);

		name->setText(FormatFieldWidgetName(field->getObjectName()));
		name->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);//


		QPushButton* unfold = new QPushButton("Squard");

		QDrawLabel* DrawLabel = new QDrawLabel();
		DrawLabel->setMode(combox->currentIndex());
		DrawLabel->setBorderMode(int(f->getDataPtr()->Bordermode));
		DrawLabel->setField(f);
		DrawLabel->copyFromField(f->getDataPtr()->Originalcoord);
		
		connect(combox, SIGNAL(currentIndexChanged(int)), DrawLabel, SLOT(changeValue(int)));


		QPushButton* button = new QPushButton("Button");
	
		button->setFixedSize(60, 24);
		button->setMaximumWidth(100);

		layout->addWidget(name, 0, 0, Qt::AlignLeft);
		layout->addWidget(DrawLabel,0,1, Qt::AlignCenter);
		layout->addWidget(button,0,2, Qt::AlignRight);
		layout->addWidget(combox, 1, 1, Qt::AlignLeft);
		layout->addWidget(unfold,1,2,Qt::AlignLeft);

		layout->setColumnStretch(0, 0);
		layout->setColumnStretch(1, 5);
		layout->setColumnStretch(2, 0);
	}

	void QRampWidget::changeValue()
	{
		FVar<Ramp>* f = TypeInfo::cast<FVar<Ramp>>(mField);
		if (f == nullptr)
		{
			printf("QRamp Nullptr ChangeValue\n");
			return;
		}
		printf("QRamp Layout ChangeValue\n");
	}

	void QDrawLabel::paintEvent(QPaintEvent* event)
	{
		printf("paintEvent\n");
		radius = 4;
		int w = this->width();
		int h = this->height();
		minX = 0 + 1.5 * radius;
		maxX = w - 2 * radius;
		minY = 0 + 2 * radius;
		maxY = h - 1.5 * radius;

		if (CoordArray.empty())
		{
			MyCoord FirstCoord;
			MyCoord LastCoord;
			CoordArray.push_back(FirstCoord);
			CoordArray.push_back(LastCoord);
			
			if (Mode == x)
			{
				CoordArray[0].x = minX;
				CoordArray[0].y = (maxY + minY) / 2 ;
				CoordArray[1].x = maxX;
				CoordArray[1].y = (maxY + minY) / 2 ;
			}
			if (Mode == y)
			{
				CoordArray[0].x = (maxX + minX) / 2;
				CoordArray[0].y = minY;
				CoordArray[1].x = (maxX + minX) / 2;
				CoordArray[1].y = maxY;
			}
		}
		

		QPainter painter(this);
		painter.setRenderHint(QPainter::Antialiasing, true);
		//BG
		QBrush brush = QBrush(Qt::black, Qt::SolidPattern);
		painter.setBrush(brush);

		QRectF Bound = QRectF(QPointF(minX, minY), QPointF(maxX, maxY));
		painter.drawRect(Bound);
		//Grid
		QBrush brush2 = QBrush(QColor(100,100,100), Qt::CrossPattern);
		painter.setBrush(brush2);
		painter.drawRect(Bound);

		//Draw Ellipse
		size_t ptNum = CoordArray.size();

		QVector<QPointF> QCoordArray;
		reSortCoordArray.assign(CoordArray.begin(),CoordArray.end());
		reSort(reSortCoordArray);
		for (size_t i = 0; i < reSortCoordArray.size(); i++) 
		{
			QCoordArray.push_back(QPointF(reSortCoordArray[i].x, reSortCoordArray[i].y));
		}
		//绘制曲线
		QPen LinePen = QPen(QPen(QBrush(Qt::white), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
		painter.setPen(LinePen);
		for (size_t i = 0; i < QCoordArray.size() - 1; i++) 
		{
			painter.drawLine(QCoordArray[i], QCoordArray[i+1]);
		}
		//绘制点
		for (size_t i = 0; i < ptNum; i++)
		{
			painter.setBrush(QBrush(Qt::gray, Qt::SolidPattern));
			painter.drawEllipse(CoordArray[i].x - radius, CoordArray[i].y - radius, 2*radius, 2 * radius);
			painter.setPen(QPen(QBrush(QColor(200,200,200), Qt::SolidPattern), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
			painter.drawEllipse(CoordArray[i].x - radius, CoordArray[i].y - radius, 2 * radius, 2 * radius);
			printf("第%d个点 : %d - %d\n",int(i), CoordArray[i].x - radius, CoordArray[i].y - radius);
			
		}
		//Paint SelectPoint
		if (selectPoint != -1) 
		{
			painter.setBrush(QBrush(QColor(80,179,255), Qt::SolidPattern));
			painter.drawEllipse(CoordArray[selectPoint].x - radius, CoordArray[selectPoint].y - radius, 2 * radius, 2 * radius);
			painter.setPen(QPen(QBrush(Qt::white, Qt::SolidPattern), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
			painter.drawEllipse(CoordArray[selectPoint].x - radius, CoordArray[selectPoint].y - radius, 2 * radius, 2 * radius);
		}
		//Paint hoverPoint
		if (hoverPoint != -1)
		{
			painter.setBrush(QBrush(QColor(80, 179, 255), Qt::SolidPattern));
			painter.drawEllipse(CoordArray[hoverPoint].x - radius, CoordArray[hoverPoint].y - radius, 2 * radius, 2 * radius);
			painter.setPen(QPen(QBrush(Qt::white, Qt::SolidPattern), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
			painter.drawEllipse(CoordArray[hoverPoint].x - radius, CoordArray[hoverPoint].y - radius, 2 * radius, 2 * radius);
		}
		printf("绘制时CArray大小：%d\n",CoordArray.size());
	}
	QDrawLabel::~QDrawLabel() 
	{

	}

	QDrawLabel::QDrawLabel(QWidget* parent)
	{
		this->setFixedSize(470, 100);
		this->setMinimumSize(350, 80);
		this->setMaximumSize(1920, 1920);
		w0 = this->width();
		h0 = this->height();
		this->setStyleSheet("background:rgba(110,115,100,1)");
		this->setMouseTracking(true);

	}
	void QDrawLabel::reSort(std::vector<MyCoord>& vector1)
	{
		if (Mode == x)
		{
			std::sort(vector1.begin(), vector1.end(), sortx);
		}

		if (Mode == y)
		{
			sort(vector1.begin(), vector1.end(), sorty);
		}
	}

	void QDrawLabel::mousePressEvent(QMouseEvent* event) 
	{
		MyCoord pressCoord;
		pressCoord.x = event->pos().x();
		pressCoord.y = event->pos().y();

		for (size_t i = 0; i < CoordArray.size();i++) 
		{
			int temp = sqrt(std::pow((pressCoord.x - CoordArray[i].x),2) + std::pow((pressCoord.y - CoordArray[i].y),2));

			if (temp < selectDistance)
			{
				selectPoint = i;
				isSelect = true;
				break;
			}
		}

		if (!isSelect) 
		{
			CoordArray.push_back(pressCoord);
			selectPoint = CoordArray.size() - 1;
			isSelect = true;

		}
		
		this->update();

	}


	void QDrawLabel::mouseMoveEvent(QMouseEvent* event)
	{
		//移动约束 
		if (isSelect) 
		{
			//首位移动约束 
			if (borderMode == BorderMode::Close && selectPoint <= 1 )
			{
				if (Mode == Dir::x)
				{
					CoordArray[selectPoint].y = dyno::clamp(event->pos().y(),minY, maxY);
				}
				else if (Mode == Dir::y) 
				{
					CoordArray[selectPoint].x = dyno::clamp(event->pos().x(),minX, maxX );
				}
			}
			else
			{
				CoordArray[selectPoint].x = dyno::clamp(event->pos().x(), minX, maxX);
				CoordArray[selectPoint].y = dyno::clamp(event->pos().y(), minY, maxY);
			}
			update();
		}

		if (isHover == true)
		{
			int tempHover = sqrt(std::pow((event->pos().x() - CoordArray[hoverPoint].x), 2) + std::pow((event->pos().y() - CoordArray[hoverPoint].y), 2));
			if (tempHover >= selectDistance)
			{
				hoverPoint = -1;
				isHover = false;
			}
		}
		else 
		{
			for (size_t i = 0; i < CoordArray.size(); i++)
			{
				int temp = sqrt(std::pow((event->pos().x() - CoordArray[i].x), 2) + std::pow((event->pos().y() - CoordArray[i].y), 2));

				if (temp < selectDistance)
				{
					hoverPoint = i;
					isHover = true;
					break;
				}
			}
			update();
		}
		printf("xy : %d - %d\n", event->pos().x(), event->pos().y());


	}
	void QDrawLabel::mouseReleaseEvent(QMouseEvent* event)
	{
		selectPoint = -1;
		isSelect = false;
		//更新数据到field 
		if (field == nullptr){return;}
		else
		{
			updateFloatCoordArray();
			CoordtoField(floatCoord,field);
		}

	}

	void QDrawLabel::changeValue(int s) 
	{
		this->Mode = (Dir)s;

		initializeLine(Mode);
	}

	void QDrawLabel::initializeLine(Dir mode) 
	{
		if (mode == x)
		{
			CoordArray[0].x = minX;
			CoordArray[0].y = (maxY + minY) / 2 ;
			CoordArray[1].x = maxX;
			CoordArray[1].y = (maxY + minY) / 2;
		}
		if (mode == y)
		{
			CoordArray[0].x = (maxX + minX) / 2 ;
			CoordArray[0].y = minY;
			CoordArray[1].x = (maxX + minX) / 2;
			CoordArray[1].y = maxY;
		}

	}



}

