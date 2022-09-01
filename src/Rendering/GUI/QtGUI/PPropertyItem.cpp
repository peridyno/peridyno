#include "PPropertyItem.h"
#include "Module.h"
#include "Node.h"
#include "FilePath.h"

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

		QDoubleSlider* slider = new QDoubleSlider;
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

		QObject::connect(slider, SIGNAL(valueChanged(double)), spinner, SLOT(setValue(double)));
		QObject::connect(spinner, SIGNAL(valueChanged(double)), slider, SLOT(setValue(double)));
		QObject::connect(spinner, SIGNAL(valueChanged(double)), this, SLOT(changeValue(double)));

		std::string template_name = field->getTemplateName();
		if (template_name == std::string(typeid(float).name()))
		{
			FVar<float>* f = TypeInfo::cast<FVar<float>>(mField);
			slider->setValue((double)f->getValue());
		}
		else if(template_name == std::string(typeid(double).name()))
		{
			FVar<double>* f = TypeInfo::cast<FVar<double>>(mField);
			slider->setValue(f->getValue());
		}

		FormatFieldWidgetName(field->getObjectName());
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

	mDoubleSpinBox::mDoubleSpinBox(QWidget* parent)
		: QDoubleSpinBox(parent)
	{
		
	}
	void mDoubleSpinBox::wheelEvent(QWheelEvent* event)
	{
		
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
			f->update();
		}
		else if (template_name == std::string(typeid(Vec3d).name()))
		{
			FVar<Vec3d>* f = TypeInfo::cast<FVar<Vec3d>>(mField);
			f->setValue(Vec3d(v1, v2, v3));
			f->update();
		}

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

		spinner1->setValue(v1);
		spinner2->setValue(v2);
		spinner3->setValue(v3);

		QObject::connect(spinner1, SIGNAL(valueChanged(int)), this, SLOT(changeValue(int)));
		QObject::connect(spinner2, SIGNAL(valueChanged(int)), this, SLOT(changeValue(int)));
		QObject::connect(spinner3, SIGNAL(valueChanged(int)), this, SLOT(changeValue(int)));
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
		layout->addStretch(1);

		QComboBox* combox = new QComboBox;
		combox->setFixedHeight(24);
		combox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

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

}
