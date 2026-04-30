#include "QArrayWidget.h"

namespace dyno 
{
    ArrayWidget::ArrayWidget(FList* t) : QFieldWidget(t)
    {
        mFlist = t;

        this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

        QString name = FormatFieldWidgetName(t->getObjectName());
        name = QString("[ ") + name + QString("]");
        this->setTitle(name);

        this->setStyleSheet(R"(
					QGroupBox {
						margin-top: 12px;
						border: 2px solid #454545;
					}
					QGroupBox::title {
						subcontrol-origin: margin;
						subcontrol-position:top center;
					}
				)");
        auto button = new LockerButton;
        button->setContentsMargins(8, 0, 0, 0);
        button->SetTextLabel(QString("[ ") + FormatFieldWidgetName(t->getObjectName()) + QString(" Elements") + QString("]"));
        button->SetImageLabel(QPixmap((getAssetPath() + "/icon/arrow_right_pressed.png").c_str()));
        button->GetTextHandle()->setAttribute(Qt::WA_TransparentForMouseEvents, true);
        button->GetImageHandle()->setAttribute(Qt::WA_TransparentForMouseEvents, true);

        button->setStyleSheet(R"(
                LockerButton {
                    background-color: #464646;
                    border: 1px solid #000000;
                    border-radius: 4px;
                }
                LockerButton:hover {
                    background-color: #616161;
                }
                LockerButton:pressed {
                    background-color: #000000;
                }
            )");

        sizeLabel = new QLabel;
        sizeLabel->setText("Size: "); 
        sizeLabel->setStyleSheet(R"(
            QLabel {
                background-color: #464646;  
                color: #ffffff;
                border-radius: 4px;        
                border: 1px solid #000000; 
            }
        )");

        const QString btnStyle = R"(
			QPushButton {
				background-color: #464646;
				border-radius: 4px;
			}
			QPushButton:hover {
				background-color: #616161;
			}
			QPushButton:pressed {
				background-color: #000000;
			}
		)";

        addButton = new QPushButton(" Add ");
        clearButton = new QPushButton("Clear");
        addButton->setStyleSheet(btnStyle);
        clearButton->setStyleSheet(btnStyle);

        QHBoxLayout* controlLayout = new QHBoxLayout;
        controlLayout->addWidget(sizeLabel,1);
        controlLayout->addWidget(addButton,0.5);
        controlLayout->addWidget(clearButton,0.5);
        controlLayout->setSpacing(0);

        sizeLabel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        addButton->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        clearButton->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizeLabel->setAlignment(Qt::AlignLeft);

        auto mainlayout = new QVBoxLayout;
        setLayout(mainlayout);

        auto elementWidget = new QWidget;
        listLayout = new QVBoxLayout;
        listLayout->setSpacing(0);
        listLayout->setContentsMargins(0, 0, 0, 0);
        this->setContentsMargins(0, 0, 0, 0);

        elementWidget->setLayout(listLayout);
        QHBoxLayout* titleLayout = new QHBoxLayout;
        titleLayout->addWidget(button,2);
        titleLayout->addLayout(controlLayout,1);
        titleLayout->setSpacing(0);

        mainlayout->addLayout(titleLayout);
        mainlayout->addWidget(elementWidget);
        elementWidget->setVisible(false);

        connect(button, &LockerButton::clicked, [this, button, elementWidget]() {
            if (!mFlag)
            {
                button->SetImageLabel(QPixmap((getAssetPath() + "/icon/arrow_right_pressed.png").c_str()));
                elementWidget->setVisible(false);
            }
            else
            {
                button->SetImageLabel(QPixmap((getAssetPath() + "/icon/arrow_down_pressed.png").c_str()));
                elementWidget->setVisible(true);
            }
            mFlag = !mFlag;
        });

        connect(addButton, &QPushButton::clicked, [&]() {
            this->addElement();
        });
        connect(clearButton, &QPushButton::clicked, [&]() {
            this->clearList();
        });
    }

	ArrayWidget::~ArrayWidget()
	{
        mItemMapper.clear();
	}

	void ArrayWidget::addItem(QWidget* item, int id, FBase* field)
	{
		if (!item)
			return;

        mItemMapper[id] = field;

		QHBoxLayout* elementLayout = new QHBoxLayout;
		elementLayout->setAlignment(Qt::AlignVCenter);
		QLabel* label = new QLabel;
		label->setFixedWidth(30);
		elementLayout->addWidget(label);
		elementLayout->addWidget(item);
		listLayout->addLayout(elementLayout);

		auto deleteButton = new QPushButton("-");
		elementLayout->addWidget(deleteButton);
		deleteButton->setFixedSize(20, 20);
		deleteButton->setStyleSheet(R"(
                QPushButton {
                    background-color: #464646;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #616161;
                }
                QPushButton:pressed {
                    background-color: #000000;
                }
            )");

		connect(deleteButton, &QPushButton::clicked, [this, elementLayout]() {
			int index = listLayout->indexOf(elementLayout);
			this->deleteElement(index);
			});
	};

}
