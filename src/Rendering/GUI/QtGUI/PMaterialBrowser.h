
#include <QWidget>
#include <QFileSystemModel>
#include <QTreeView>
#include <QListView>

#include <QTextEdit.h>
#include <QPushButton.h>
#include <QMessageBox.h>

#include <QApplication>
#include <QListView>
#include <QStandardItemModel>
#include <QStandardItem>
#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QMessageBox>
#include <QKeyEvent>
#include "Topology/MaterialManager.h"
#include "PPropertyWidget.h"
#include <QMainWindow>
#include "NodeEditor/QtMaterialFlowWidget.h"

Q_DECLARE_METATYPE(std::shared_ptr<dyno::MaterialManagedModule>)
Q_DECLARE_METATYPE(std::shared_ptr<dyno::MaterialManagedModule>*)
namespace dyno
{
	class MaterialManager;


    class MaterialModuleEditor : public QDialog {
        Q_OBJECT
    public:
        explicit MaterialModuleEditor(std::shared_ptr<MaterialManagedModule> matPtr, QWidget* parent = nullptr);

    public slots:
        void resetSimulation();

    private:
        std::shared_ptr<Material> mMaterial;
        QLineEdit* nameEdit;
    };

    class PMaterialEditor :
        public QMainWindow
    {
        Q_OBJECT
    public:
        PMaterialEditor(std::shared_ptr<CustomMaterial> customMat);

        //PModuleEditorToolBar* toolBar() { return mToolBar; }

        //Qt::QtModuleFlowScene* moduleFlowScene() { return mModuleFlowScene; }

    signals:
        void changed(Node* node);

    private:
        Qt::QtMaterialFlowScene* mModuleFlowScene;

    };

	class PMaterialBrowser : public QWidget, public MaterialManagerObserver
	{
		Q_OBJECT
	public:
		explicit PMaterialBrowser(QWidget* parent = nullptr);

	signals:

	Q_SIGNALS:
		void materialUpdated();

	public slots:
		void createItem();
        void deleteSelectedItems();
        void copySelectedItems();

    public:
        void onMaterialChanged(std::shared_ptr<MaterialManagedModule> mat) override;

    protected:
        void keyPressEvent(QKeyEvent* event)override;

    private:
        QListView* mListView;
        QStandardItemModel* mModel;
        std::vector<std::shared_ptr<MaterialManagedModule>> copiedMaterials;
	};
}

