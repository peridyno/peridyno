#include "QFilePathWidget.h"

#include "Field.h"
#include "Field/FilePath.h"

#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>

namespace dyno
{
	IMPL_FIELD_WIDGET(std::string, QStringFieldWidget)
	IMPL_FIELD_WIDGET(FilePath, QFilePathWidget)

	QStringFieldWidget::QStringFieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		f = TypeInfo::cast<FVar<std::string>>(field);

		//this->setStyleSheet("border:none");
		QHBoxLayout* layout = new QHBoxLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		//Label
		QLabel* name = new QLabel();
		QString str = FormatFieldWidgetName(field->getObjectName());
		name->setFixedSize(100, 18);
		QFontMetrics fontMetrics(name->font());
		QString elide = fontMetrics.elidedText(str, Qt::ElideRight, 100);
		name->setText(elide);
		//Set label tips
		name->setToolTip(str);

		fieldname = new QLineEdit;
		fieldname->setText(QString::fromStdString(f->getValue()));

		layout->addWidget(name, 0);
		layout->addWidget(fieldname, 1);
		layout->setSpacing(3);

		connect(fieldname, &QLineEdit::textChanged, this, &QStringFieldWidget::updateField);
		QObject::connect(this, SIGNAL(fieldChanged()), this, SLOT(updateWidget()));
	}

	void QStringFieldWidget::updateField(QString str)
	{
		auto f = TypeInfo::cast<FVar<std::string>>(field());
		if (f == nullptr)
		{
			return;
		}
		f->setValue(str.toStdString());
		f->update();
	}

	void QStringFieldWidget::updateWidget()
	{
		std::string str = f->getValue();
		fieldname->setText(QString::fromStdString(str));
	}

	QFilePathWidget::QFilePathWidget(FBase* field)
		: QFieldWidget(field)
	{
		FVar<FilePath>* f = TypeInfo::cast<FVar<FilePath>>(field);

		//this->setStyleSheet("border:none");
		QHBoxLayout* layout = new QHBoxLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		//Label
		QLabel* name = new QLabel();
		QString str = FormatFieldWidgetName(field->getObjectName());
		name->setFixedSize(100, 18);
		QFontMetrics fontMetrics(name->font());
		QString elide = fontMetrics.elidedText(str, Qt::ElideRight, 100);
		name->setText(elide);
		//Set label tips
		name->setToolTip(str);

		location = new QLineEdit;
		location->setText(QString::fromStdString(f->getValue().string()));

		QPushButton* open = new QPushButton("Open");
// 		open->setStyleSheet("QPushButton{color: black;   border-radius: 10px;  border: 1px groove black;background-color:white; }"
// 							"QPushButton:hover{background-color:white; color: black;}"  
// 							"QPushButton:pressed{background-color:rgb(85, 170, 255); border-style: inset; }" );
		open->setFixedSize(60, 24);

		layout->addWidget(name, 0);
		layout->addWidget(location, 1);
		layout->addWidget(open, 2);
		layout->setSpacing(3);

		connect(location, &QLineEdit::textChanged, this, &QFilePathWidget::updateField);

		connect(open, &QPushButton::clicked, this, [=]() {

			bool bPath = f->constDataPtr()->is_path();
			if (bPath)
			{
				QString path = QFileDialog::getExistingDirectory(this, tr("Open File"), QString::fromStdString(getAssetPath()), QFileDialog::ReadOnly);
				if (!path.isEmpty()) {
					//Windows: "\\"; Linux: "/"
					path = QDir::toNativeSeparators(path);
					location->setText(path);
				}
				else
					QMessageBox::warning(this, tr("Path"), tr("You do not select any path."));
			}
			else
			{
				QString path = QFileDialog::getOpenFileName(this, tr("Open File"), QString::fromStdString(getAssetPath()), tr("Text Files(*.*)"));
				if (!path.isEmpty()) {
					//Windows: "\\"; Linux: "/"
					path = QDir::toNativeSeparators(path);
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
			}
		});
	}

	void QFilePathWidget::updateField(QString str)
	{
		auto f = TypeInfo::cast<FVar<FilePath>>(field());
		if (f == nullptr)
		{
			return;
		}

		auto path = f->getValue();

		path.set_path(str.toStdString());

		f->setValue(path);
		f->update();

		emit fieldChanged();
	}


}

