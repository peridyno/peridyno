#include "QArrayWidget.h"

namespace dyno 
{
    ArrayWidgetBase::ArrayWidgetBase(FBase* t) : QFieldWidget(t)

    {
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
                    background-color: #2e2e2e;
                }
                LockerButton:pressed {
                    background-color: #000000;
                }
            )");

        auto buttonLayout = new QHBoxLayout;
        buttonLayout->addWidget(button);
        sizeLabel = new QLabel;
        sizeLabel->setText("Size"); 

        addButton = new LockerButton;
        deleteButton = new LockerButton;

        buttonLayout->addWidget(sizeLabel);




        auto mainlayout = new QVBoxLayout;
        setLayout(mainlayout);

        auto elementWidget = new QWidget;
        layout = new QVBoxLayout;
        layout->setSpacing(0);
        elementWidget->setLayout(layout);
        mainlayout->addWidget(button);
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

    }

    void ArrayWidgetBase::connectSignal(QFieldWidget* field, int id)
    {
        QObject::connect(field, QOverload<>::of(&QFieldWidget::fieldChanged), [this, id]()
            {
                OnElementUpdate(id);
            }
        );
    }

}
