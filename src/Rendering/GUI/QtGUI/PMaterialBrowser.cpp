#include "PMaterialBrowser.h"
#include "Platform.h"
#include <QListWidget>
#include <QDir>
#include <QStringList>
#include <QHBoxLayout>
#include <QListView>
#include <QPixmap>
#include "Topology/MaterialManager.h"
#include <QStandardItemModel>
#include "PSimulationThread.h"
#include <QTreeView>
#include "Topology/MaterialManager.h"
#include <QShortcut>
#include <QKeySequence>
#include "NodeEditor/QtMaterialFlowScene.h"
#include "NodeEditor/QtMaterialFlowWidget.h"
#include <QCheckBox>
#include "PDockWidget.h"
#include "PMaterialEditorToolBar.h"
#include "PSimulationThread.h"

namespace dyno
{
    void buildIconLabel(QLabel* Label, QPixmap* Icon, QPushButton* btn, int IconSize) {

        Label->setScaledContents(true);							
        Label->setStyleSheet("background: transparent;");
        Label->setPixmap(*Icon);								
        Label->setFixedSize(IconSize, IconSize);
        Label->setStyleSheet("padding: 6px;");
        QHBoxLayout* iconLayout = new QHBoxLayout();			
        iconLayout->addWidget(Label);
        iconLayout->setSizeConstraint(QLayout::SetFixedSize);
        iconLayout->setContentsMargins(0, 0, 0, 0);
        btn->setLayout(iconLayout);		
    }

    void MaterialModuleEditor::resetSimulation()
    {
        PSimulationThread::instance()->reset();
    }

    MaterialModuleEditor::MaterialModuleEditor(std::shared_ptr<MaterialManagedModule> matPtr, QWidget* parent)
        : QDialog(parent)
    {
        QHBoxLayout* toolBarLayout = new QHBoxLayout();

        QLabel* textLabel = new QLabel("Reset Scene");
        auto mResetSim = new QPushButton();
        toolBarLayout->addWidget(textLabel);
        toolBarLayout->addWidget(mResetSim);

        mResetSim->setFixedSize(QSize(30,30));
        mResetSim->setStyleSheet("padding: 0px;");
        auto mResetIcon = new QPixmap(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/AnimationSlider/Reset.png"));
        connect(mResetSim, SIGNAL(released()), this, SLOT(resetSimulation()));

        auto mResetLabel = new QLabel;
        buildIconLabel(mResetLabel, mResetIcon, mResetSim, 30);
        mResetSim->setCheckable(false);



        qRegisterMetaType<std::shared_ptr<MaterialManagedModule>>("std::shared_ptr<MaterialManagedModule>");
        qRegisterMetaType<std::shared_ptr<MaterialManagedModule>*>("std::shared_ptr<MaterialManagedModule>*");

        setWindowTitle(QString(std::string("Material Editor : ").c_str()) + QString(matPtr->getName().c_str()));
        //setModal(true);
        resize(900, 1500);

        QVBoxLayout* layout = new QVBoxLayout(this);

        auto propertyWidget = new PPropertyWidget();
        layout->addLayout(toolBarLayout);
        layout->addWidget(propertyWidget);

        if (matPtr)
            propertyWidget->showModuleProperty(matPtr);

    }

    void PMaterialEditor::setAutoUpdatePipline(int status)
    {
            mMaterialPipline->autoUpdate = status;
    }

    PMaterialEditor::PMaterialEditor(std::shared_ptr<CustomMaterial> customMat)
    {

        //Set up property dock widget
        PMaterialEditorToolBar* matToolBar = new PMaterialEditorToolBar();

        QDockWidget* toolBarDocker = new QDockWidget();
        this->addDockWidget(Qt::TopDockWidgetArea, toolBarDocker);
        auto titleBar = toolBarDocker->titleBarWidget();
        toolBarDocker->setFixedHeight(96);
        toolBarDocker->setTitleBarWidget(new QWidget());
        delete titleBar;
        
        matToolBar->setRealtime(customMat->materialPipeline()->autoUpdate);
        toolBarDocker->setWidget(matToolBar);

        auto matFlowView = new Qt::QtMaterialFlowWidget(customMat, nullptr);

        mModuleFlowScene = matFlowView->getModuleFlowScene();
        mMaterialPipline = customMat->materialPipeline();

        this->setCentralWidget(matFlowView);

        //Set up property dock widget
        PDockWidget* propertyDockWidget = new PDockWidget(tr("Property"), this, Qt::WindowFlags(0));
        
       this->setWindowTitle(QString(std::string("Material Editor : ").c_str()) + QString(customMat->getName().c_str()));

        this->addDockWidget(Qt::LeftDockWidgetArea, propertyDockWidget);

        PPropertyWidget* propertyWidget = new PPropertyWidget();
        propertyDockWidget->setWidget(propertyWidget);
        propertyDockWidget->setMinimumWidth(480);

        connect(matFlowView->mMaterialFlow, &Qt::QtMaterialFlowScene::nodeSelected, propertyWidget, &PPropertyWidget::showProperty);
        connect(matFlowView->mMaterialFlow, &Qt::QtMaterialFlowScene::nodeDeselected, propertyWidget, &PPropertyWidget::clearProperty);

        connect(propertyWidget, &PPropertyWidget::moduleUpdated, this, &PMaterialEditor::updateMaterialPipline);


        connect(matToolBar->updateAction(), &QAction::triggered, matFlowView->mMaterialFlow, &Qt::QtMaterialFlowScene::reconstructActivePipeline);
        connect(matToolBar->reorderAction(), &QAction::triggered, matFlowView->mMaterialFlow, &Qt::QtMaterialFlowScene::reorderAllModules);
        connect(matToolBar->realTimeUpdateAction(), &QAction::toggled, this, &PMaterialEditor::setAutoUpdatePipline);

        //connect(autoUpdateCheckBox, SIGNAL(stateChanged(int)), this, SLOT(setAutoUpdatePipline(int)))
    }

	PMaterialBrowser::PMaterialBrowser(QWidget* parent /*= nullptr*/)
		: QWidget()
	{
        mModel = new QStandardItemModel;
        mModel->sort(0, Qt::AscendingOrder);

        mListView = new QListView(this);
        mListView->setResizeMode(QListView::ResizeMode::Adjust);

        auto layout = new QHBoxLayout(this);
        layout->addWidget(mListView, 1);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(0);
        setLayout(layout);

        createItem();

        MaterialManager::addObserver(this);
        connect(this, SIGNAL(materialUpdated()), this, SLOT(createItem()));

        QVBoxLayout* layout2 = new QVBoxLayout();
        layout->addLayout(layout2, 0);

        QPushButton* btnDelete = new QPushButton("Delete");
        QPushButton* btnCopy = new QPushButton("Copy");

        layout2->addWidget(btnDelete);
        layout2->addWidget(btnCopy);
        layout2->addStretch();

        connect(btnDelete, &QPushButton::clicked, this, &PMaterialBrowser::deleteSelectedItems);
        connect(btnCopy, &QPushButton::clicked, this, &PMaterialBrowser::copySelectedItems);

        //QShortcut* delShortcut = new QShortcut(QKeySequence::Delete, this);
        //connect(delShortcut, &QShortcut::activated, this, &PMaterialBrowser::deleteSelectedItems);

        QShortcut* copyShortcut = new QShortcut(QKeySequence::Copy, this);
        connect(copyShortcut, &QShortcut::activated, this, &PMaterialBrowser::copySelectedItems);

        QObject::connect(mListView, &QListView::doubleClicked, [&](const QModelIndex& index) {
            if (!index.isValid()) return;

            QVariant var = mModel->data(index, Qt::UserRole + 1);
            std::shared_ptr<MaterialManagedModule> mat = var.value<std::shared_ptr<MaterialManagedModule>>();
            if (!mat) return;

            auto customMat = std::dynamic_pointer_cast<CustomMaterial>(mat);
            if (customMat)
            {
                PMaterialEditor* MatFlowScene = new PMaterialEditor(customMat);
                MatFlowScene->setAttribute(Qt::WA_DeleteOnClose);
                MatFlowScene->show();
            }
            else 
            {

                MaterialModuleEditor* editor = new MaterialModuleEditor(mat);
                editor->setAttribute(Qt::WA_DeleteOnClose);
                editor->show();

            }
            //QObject::connect(editor, &MaterialEditor::materialUpdated, [&]() {

            //    });
            }
        );
	}

	void PMaterialBrowser::createItem()
	{
        mModel->clear();
        auto matModules = MaterialManager::materialManagedModules();
        std::string iconPath = getAssetPath() + "/icon/ContentBrowser/3dModel.png";

        QIcon icon(QString::fromStdString(iconPath));
        
        for (const auto& pair : matModules) {
            if (pair.second) {  
                auto mat = pair.second;
                std::string name = pair.second->getName();
                std::cout << "Material name: " << name << std::endl;
                QStandardItem* item = new QStandardItem(icon, QString(name.c_str()));

                item->setData(QVariant::fromValue(mat), Qt::UserRole + 1);
                mModel->appendRow(item);
            }
        }

        mListView->setModel(mModel);
        mListView->setViewMode(QListView::ListMode);
        mListView->setTextElideMode(Qt::ElideRight);
        mListView->setSelectionMode(QAbstractItemView::ExtendedSelection);
        mListView->setIconSize(QSize(64, 64));         
        mListView->setGridSize(QSize(230, 70));
        mListView->setUniformItemSizes(true);
        mListView->setResizeMode(QListView::Adjust);
        mListView->setFlow(QListView::LeftToRight);
        mListView->setWrapping(true);               
        mListView->setSpacing(0);
        mListView->setResizeMode(QListView::Adjust);
        //mListView->setWindowTitle("Material Browser");

	}

    void PMaterialBrowser::deleteSelectedItems()
    {
        this->blockSignals(true);
        auto selectedIndexes = mListView->selectionModel()->selectedIndexes();
        std::vector<std::shared_ptr<MaterialManagedModule>> deleteMatModules;

        for (const QModelIndex& index : selectedIndexes) {
            QVariant var = mModel->data(index, Qt::UserRole + 1);
            std::shared_ptr<MaterialManagedModule> mat = var.value<std::shared_ptr<MaterialManagedModule>>();
            if (mat) {
                deleteMatModules.push_back(mat);
            }
        }

        for (auto it : deleteMatModules)
        {
            MaterialManager::removeMaterialManagedModule(it->getName());
        }
        this->blockSignals(false);
        this->createItem();
        auto parentWidget = this->parentWidget();
        this->parentWidget()->adjustSize();
        parentWidget->updateGeometry();      
    }

    void PMaterialBrowser::copySelectedItems()
    {
        copiedMaterials.clear();
        auto selectedIndexes = mListView->selectionModel()->selectedIndexes();

        for (const QModelIndex& index : selectedIndexes) {
            QVariant var = mModel->data(index, Qt::UserRole + 1);
            std::shared_ptr<MaterialManagedModule> mat = var.value<std::shared_ptr<MaterialManagedModule>>();
            if (mat) {
                copiedMaterials.push_back(mat);
            }
        }

        for (auto it : copiedMaterials)
        {
            MaterialManager::copyMaterialManagedModule(it);
        }

        std::cout << "Copy Mat Num : " << copiedMaterials.size() << "\n";

        auto parentWidget = this->parentWidget();
        this->parentWidget()->adjustSize();
        parentWidget->updateGeometry();     
    }

    void PMaterialBrowser::onMaterialChanged(std::shared_ptr<MaterialManagedModule> mat)
    {
        emit materialUpdated();
    }

    void PMaterialBrowser::keyPressEvent(QKeyEvent* event)
    {
        if (event->key() == Qt::Key_Delete) {
            deleteSelectedItems();
            event->accept();
            return;
        }

        QWidget::keyPressEvent(event);
    }

    void PMaterialEditor::updateMaterialPipline(std::shared_ptr<Module> node)
    {
        if (mMaterialPipline)
        {
            if(mMaterialPipline->autoUpdate)
                mMaterialPipline->update();
        }
    }

}