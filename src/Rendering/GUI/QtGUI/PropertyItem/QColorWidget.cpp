#include "QColorWidget.h"

#include <QGridLayout>
#include <QPainter>
#include <QColorDialog>

//RenderCore
#include "Color.h"

namespace dyno
{
	QColorButton::QColorButton(QWidget* pParent) :
		QPushButton(pParent),
		mMargin(5),
		mRadius(4),
		mColor(Qt::gray)
	{
		setText("");
	}

	void QColorButton::paintEvent(QPaintEvent* event)
	{
		setText("");

		QPushButton::paintEvent(event);

		QPainter Painter(this);

		// Get button rectangle
		QRect rect = event->rect();

		// Deflate it
		rect.adjust(mMargin, mMargin, -mMargin, -mMargin);

		// Use anti aliasing
		Painter.setRenderHint(QPainter::Antialiasing);

		// Rectangle styling
		Painter.setBrush(QBrush(isEnabled() ? mColor : Qt::lightGray));
		Painter.setPen(QPen(isEnabled() ? QColor(25, 25, 25) : Qt::darkGray, 0.5));

		// Draw
		Painter.drawRoundedRect(rect, mRadius, Qt::AbsoluteSize);
	}

	void QColorButton::mousePressEvent(QMouseEvent* event)
	{
		QColorDialog colorDialog;

		connect(&colorDialog, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(onColorChanged(const QColor&)));

		//ColorDialog.setWindowIcon(GetIcon("color--pencil"));
		colorDialog.setCurrentColor(mColor);
		colorDialog.exec();

		disconnect(&colorDialog, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(onColorChanged(const QColor&)));
	}

	int QColorButton::getMargin(void) const
	{
		return mMargin;
	}

	void QColorButton::setMargin(const int& Margin)
	{
		mMargin = mMargin;
		update();
	}

	int QColorButton::getRadius(void) const
	{
		return mRadius;
	}

	void QColorButton::setRadius(const int& Radius)
	{
		mRadius = mRadius;
		update();
	}

	QColor QColorButton::getColor(void) const
	{
		return mColor;
	}

	void QColorButton::setColor(const QColor& Color, bool BlockSignals)
	{
		blockSignals(BlockSignals);

		mColor = Color;
		update();

		blockSignals(false);
	}

	void QColorButton::onColorChanged(const QColor& Color)
	{
		setColor(Color);

		emit colorChanged(mColor);
	}

	QColorWidget::QColorWidget(FBase* field)
		: QFieldWidget(field)
	{
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setHorizontalSpacing(3);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		QString str = FormatFieldWidgetName(field->getObjectName());
		name->setFixedSize(100, 18);
		QFontMetrics fontMetrics(name->font());
		QString elide = fontMetrics.elidedText(str, Qt::ElideRight, 100);
		name->setText(elide);
		//Set label tips
		name->setToolTip(str);

		spinner1 = new QSpinBox;
		spinner1->setMinimumWidth(30);
		spinner1->setRange(0, 255);

		spinner2 = new QSpinBox;
		spinner2->setMinimumWidth(30);
		spinner2->setRange(0, 255);

		spinner3 = new QSpinBox;
		spinner3->setMinimumWidth(30);
		spinner3->setRange(0, 255);

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinner1, 0, 1);
		layout->addWidget(spinner2, 0, 2);
		layout->addWidget(spinner3, 0, 3);

		colorButton = new QColorButton;
		colorButton->setFixedSize(30, 30);

		layout->addWidget(colorButton, 0, 4);

		std::string template_name = field->getTemplateName();
		int R = 0;
		int G = 0;
		int B = 0;

		if (template_name == std::string(typeid(Color).name()))
		{
			FVar<Color>* f = TypeInfo::cast<FVar<Color>>(field);
			auto v = f->getData();

			int r = int(v.r * 255) % 255;
			int g = int(v.g * 255) % 255;
			int b = int(v.b * 255) % 255;

			spinner1->setValue(r);
			spinner2->setValue(g);
			spinner3->setValue(b);

			colorButton->setColor(QColor(r, g, b), true);
		}

		QObject::connect(spinner1, SIGNAL(valueChanged(int)), this, SLOT(updateField(int)));
		QObject::connect(spinner2, SIGNAL(valueChanged(int)), this, SLOT(updateField(int)));
		QObject::connect(spinner3, SIGNAL(valueChanged(int)), this, SLOT(updateField(int)));

		QObject::connect(colorButton, SIGNAL(colorChanged(const QColor&)), this, SLOT(updateColorWidget(const QColor&)));
	}

	QColorWidget::~QColorWidget()
	{
	}

	void QColorWidget::updateField(int)
	{
		int v1 = spinner1->value();
		int v2 = spinner2->value();
		int v3 = spinner3->value();

		std::string template_name = field()->getTemplateName();

		if (template_name == std::string(typeid(Color).name()))
		{
			FVar<Color>* f = TypeInfo::cast<FVar<Color>>(field());

			float r = float(v1) / 255;
			float g = float(v2) / 255;
			float b = float(v3) / 255;

			f->setValue(Color(r, g, b));
		}

		colorButton->setColor(QColor(v1, v2, v3), true);
	}

	void QColorWidget::updateColorWidget(const QColor& color)
	{
		spinner1->blockSignals(true);
		spinner2->blockSignals(true);
		spinner3->blockSignals(true);

		spinner1->setValue(color.red());
		spinner2->setValue(color.green());
		spinner3->setValue(color.blue());

		spinner1->blockSignals(false);
		spinner2->blockSignals(false);
		spinner3->blockSignals(false);

		updateField(0);
	}
}

