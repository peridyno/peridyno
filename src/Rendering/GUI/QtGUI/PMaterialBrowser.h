
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
#include "MaterialManager.h"
#include "PPropertyWidget.h"
#include <QComboBox>
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

    signals:
        void changed(Node* node);

    public slots:
        void setAutoUpdatePipline(int state);
        void updateMaterialPipline(std::shared_ptr<Module> node);
    private:
        Qt::QtMaterialFlowScene* mModuleFlowScene = NULL;
        std::shared_ptr<MaterialPipeline> mMaterialPipline = NULL;

    };

    enum class MaterialBrowserFilter {
        All,
        MaterialLoader,
        CustomMaterial,
        Other
    };

	class PMaterialBrowser : public QWidget, public MaterialManagerObserver
	{
		Q_OBJECT
	public:
		explicit PMaterialBrowser(QWidget* parent = nullptr);

	signals:

	Q_SIGNALS:
		void materialListChanged();

	public slots:
		void createItem();
        void deleteSelectedItems();
        void copySelectedItems();
        void onFilterChanged(int index);

    public:
        void onMaterialListChanged(std::shared_ptr<MaterialManagedModule> mat) override;

    protected:
        void keyPressEvent(QKeyEvent* event)override;

    private:
        QComboBox* comboBox;
        QListView* mListView;
        QStandardItemModel* mModel;
        std::vector<std::shared_ptr<MaterialManagedModule>> copiedMaterials;
        MaterialBrowserFilter filter = MaterialBrowserFilter::All;
	};
}

