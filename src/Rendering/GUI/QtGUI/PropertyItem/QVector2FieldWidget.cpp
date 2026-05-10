#include "QVector2FieldWidget.h"

#include <QVBoxLayout>

#include "Field.h"
#include "QPiecewiseDoubleSpinBox.h"


namespace dyno
{
	//IMPL_FIELD_WIDGET(Vec2f, QVector2FieldWidget)

	int QVector2FieldWidget::reg_field_widget = []() {
		dyno::PPropertyWidget::registerWidget({ &typeid(Vec2f), &QVector2FieldWidget::createWidget });
		dyno::PPropertyWidget::registerWidget({ &typeid(Vec2d), &QVector2FieldWidget::createWidget });
		return 0;
	}();

	QWidget* QVector2FieldWidget::createWidget(dyno::FBase* f) {
			return new QVector2FieldWidget(f);
	}

	QVector2FieldWidget::QVector2FieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QToggleLabel* name = new QToggleLabel();
		QString str = FormatFieldWidgetName(field->getObjectName());
		name->setFixedSize(100, 18);
		QFontMetrics fontMetrics(name->font());
		QString elide = fontMetrics.elidedText(str, Qt::ElideRight, 100);
		name->setText(elide);
		//Set label tips
		name->setToolTip(str);

		spinner1 = new QPiecewiseDoubleSpinBox;
		spinner1->setMinimumWidth(30);
		spinner1->setRange(field->getMin(), field->getMax());

		spinner2 = new QPiecewiseDoubleSpinBox;
		spinner2->setMinimumWidth(30);
		spinner2->setRange(field->getMin(), field->getMax());

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinner1, 0, 1);
		layout->addWidget(spinner2, 0, 2);
		layout->setSpacing(3);

		std::string template_name = field->getTemplateName();

		double v1 = 0;
		double v2 = 0;

		bool isDouble = false;

		if (template_name == std::string(typeid(Vec2f).name()))
		{
			FVar<Vec2f>* f = TypeInfo::cast<FVar<Vec2f>>(field);
			auto v = f->getValue();
			v1 = v[0];
			v2 = v[1];
		}
		else if (template_name == std::string(typeid(Vec2d).name()))
		{
			FVar<Vec2d>* f = TypeInfo::cast<FVar<Vec2d>>(field);
			auto v = f->getValue();

			v1 = v[0];
			v2 = v[1];
			isDouble = true;
		}
		spinner1->setRealValue(v1);
		spinner2->setRealValue(v2);

		spinner1->setDouble(isDouble);
		spinner2->setDouble(isDouble);

		QObject::connect(spinner1, SIGNAL(editingFinishedWithValue(double)), this, SLOT(updateField(double)));
		QObject::connect(spinner2, SIGNAL(editingFinishedWithValue(double)), this, SLOT(updateField(double)));
		
		QObject::connect(name, SIGNAL(toggle(bool)), spinner1, SLOT(toggleDecimals(bool)));
		QObject::connect(name, SIGNAL(toggle(bool)), spinner2, SLOT(toggleDecimals(bool)));

		QObject::connect(this, SIGNAL(fieldChanged()), this, SLOT(updateWidget()));
	}


	QVector2FieldWidget::QVector2FieldWidget(QString name, Vec2f v)
	{
		value = v;

		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QToggleLabel* nameLabel = new QToggleLabel();
		
		nameLabel->setFixedSize(100, 18);
		QFontMetrics fontMetrics(nameLabel->font());
		QString elide = fontMetrics.elidedText(name, Qt::ElideRight, 100);
		nameLabel->setText(elide);
		//Set label tips
		nameLabel->setToolTip(name);

		spinner1 = new QPiecewiseDoubleSpinBox(value[0]);
		spinner1->setMinimumWidth(30);
		spinner1->setRange(-100000, 100000);

		spinner2 = new QPiecewiseDoubleSpinBox(value[1]);
		spinner2->setMinimumWidth(30);
		spinner2->setRange(-100000, 100000);

		layout->addWidget(nameLabel, 0, 0);
		layout->addWidget(spinner1, 0, 1);
		layout->addWidget(spinner2, 0, 2);
		layout->setSpacing(3);

		QObject::connect(nameLabel, SIGNAL(toggle(bool)), spinner1, SLOT(toggleDecimals(bool)));
		QObject::connect(nameLabel, SIGNAL(toggle(bool)), spinner2, SLOT(toggleDecimals(bool)));

		QObject::connect(spinner1, SIGNAL(valueChanged(double)), this, SLOT(vec2fValueChange(double)));
		QObject::connect(spinner2, SIGNAL(valueChanged(double)), this, SLOT(vec2fValueChange(double)));
	}


	QVector2FieldWidget::~QVector2FieldWidget()
	{
	}

	void QVector2FieldWidget::updateField(double value)
	{
		double v1 = spinner1->getRealValue();
		double v2 = spinner2->getRealValue();

		std::string template_name = field()->getTemplateName();

		if (template_name == std::string(typeid(Vec2f).name()))
		{
			FVar<Vec2f>* f = TypeInfo::cast<FVar<Vec2f>>(field());
			f->setValue(Vec2f((float)v1, (float)v2),false);
		}
		else if (template_name == std::string(typeid(Vec2d).name()))
		{
			FVar<Vec2d>* f = TypeInfo::cast<FVar<Vec2d>>(field());
			f->setValue(Vec2d(v1, v2), false);
		}

		emit fieldChanged();
	}


	void QVector2FieldWidget::updateWidget()
	{
		std::string template_name = field()->getTemplateName();

		double v1 = 0;
		double v2 = 0;

		if (template_name == std::string(typeid(Vec2f).name()))
		{
			FVar<Vec2f>* f = TypeInfo::cast<FVar<Vec2f>>(field());
			auto v = f->getValue();
			v1 = v[0];
			v2 = v[1];
		}
		else if (template_name == std::string(typeid(Vec2d).name()))
		{
			FVar<Vec2d>* f = TypeInfo::cast<FVar<Vec2d>>(field());
			auto v = f->getValue();

			v1 = v[0];
			v2 = v[1];

		}

		spinner1->blockSignals(true);
		spinner2->blockSignals(true);

		spinner1->setRealValue(v1);
		spinner2->setRealValue(v2);

		spinner1->blockSignals(false);
		spinner2->blockSignals(false);
	}

	void QVector2FieldWidget::vec2fValueChange(double)
	{
		value = Vec2f(spinner1->getRealValue(), spinner2->getRealValue());
		emit vec2fChange(double(value[0]), double(value[1]));
	}

}
