#include "PConsoleWidget.h"
#include "Platform.h"
#include <QListWidget>
#include <QDir>
#include <QStringList>
#include <QHBoxLayout>
#include <QListView>
#include <QPixmap>
#include "NodeFactory.h"

#include "PSimulationThread.h"

namespace dyno
{
	class CustomFileSystemModel : public QFileSystemModel
	{
	public:
		explicit CustomFileSystemModel(QObject* parent = nullptr) {};
		~CustomFileSystemModel() {};
	private:
		QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const
		{
			if (index.isValid())
			{
				if (role == Qt::DecorationRole)
				{
					QFileInfo info = CustomFileSystemModel::fileInfo(index);
					if (info.isFile())
					{
						if (info.suffix() == "png" || info.suffix() == "jpg" || info.suffix() == "bmp")
						{
							std::string iconPath = getAssetPath() + "/icon/ContentBrowser/image.png";
							return QPixmap(iconPath.c_str());
						}
						else if (info.suffix() == "obj" || info.suffix() == "gltf" || info.suffix() == "glb" || info.suffix() == "fbx" || info.suffix() == "STL" || info.suffix() == "stl")
						{
							std::string iconPath = getAssetPath() + "/icon/ContentBrowser/3dModel.png";
							return QPixmap(iconPath.c_str());
						}
					}
				}
			}

			return QFileSystemModel::data(index, role);
		}
	};


	PConsoleWidget::PConsoleWidget(QWidget* parent) :
		QWidget(parent)
	{

		QVBoxLayout* layout = new QVBoxLayout(this);
		this->setLayout(layout);

		mCodeEditor = new QsciScintilla(this);
		mCodeEditor->setObjectName("Python Code");

		mCodeEditor->setMinimumSize(100, 100);

		mPythonLexer = new QsciLexerPython(mCodeEditor);
		mCodeEditor->setLexer(mPythonLexer);

		mCodeEditor->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		
		// line numbers
		mCodeEditor->setMarginsBackgroundColor(QColor("#f0f0f0"));
		mCodeEditor->setMarginLineNumbers(1, true);
		mCodeEditor->setMarginWidth(1, "0000");

		// indentation
		mCodeEditor->setIndentationsUseTabs(false);
		mCodeEditor->setIndentationWidth(4);
		mCodeEditor->setTabWidth(4);
		mCodeEditor->setAutoIndent(true);

		// brace match
		mCodeEditor->setBraceMatching(QsciScintilla::SloppyBraceMatch);
		mCodeEditor->setMatchedBraceBackgroundColor(QColor("#ffff00"));

		mCodeEditor->setFolding(QsciScintilla::BoxedTreeFoldStyle);

		// auto completion
		mCodeEditor->setAutoCompletionSource(QsciScintilla::AcsAll);
		mCodeEditor->setAutoCompletionCaseSensitivity(false);
		mCodeEditor->setAutoCompletionThreshold(1);

		layout->addWidget(mCodeEditor);

		

		updateButton = new QPushButton("update", this);
		layout->addWidget(updateButton);


		connect(updateButton, &QPushButton::clicked, this, [this]() {
			const std::string& code = mCodeEditor->text().toStdString();
			execute(code);
			});

	}

	void PConsoleWidget::execute(const std::string& src)
	{
		py::scoped_interpreter guard{};

		try {
			auto locals = py::dict();
			py::exec(src, py::globals(), locals);

			if (locals.contains("scn"))
			{
				auto scene = locals["scn"].cast<std::shared_ptr<dyno::SceneGraph>>();
				if (scene)
				{
					PSimulationThread::instance()->createNewScene(scene);
				}
			}
			else
			{
				std::cout << getPythonErrorDetails() << std::endl;
			}
		}
		catch (const std::exception& e) {
			QMessageBox::critical(nullptr, "Error", e.what());
		}
	}

	std::string PConsoleWidget::getPythonErrorDetails()
	{
		if (!PyErr_Occurred()) {
			return "No Python error occurred";
		}

		PyObject* type, * value, * traceback;
		PyErr_Fetch(&type, &value, &traceback);
		PyErr_NormalizeException(&type, &value, &traceback);

		std::string errorMsg;

		if (value) {
			py::object py_value = py::reinterpret_borrow<py::object>(value);
			errorMsg = py::str(py_value).cast<std::string>();
		}
		else {
			errorMsg = "Unknown Python error";
		}

		// 如果有 traceback，获取堆栈信息
		if (traceback) {
			py::object py_traceback = py::reinterpret_borrow<py::object>(traceback);

			// 导入 traceback 模块来格式化错误信息
			try {
				py::module traceback_module = py::module::import("traceback");
				py::object format_tb = traceback_module.attr("format_tb");
				py::object tb_list = format_tb(py_traceback);
				py::object formatted = traceback_module.attr("format_exception_only")(type, value);

				// 组合完整的错误信息
				std::string traceback_str;
				for (auto item : tb_list) {
					traceback_str += item.cast<std::string>();
				}
				for (auto item : formatted) {
					traceback_str += item.cast<std::string>();
				}

				if (!traceback_str.empty()) {
					errorMsg = traceback_str;
				}
			}
			catch (...) {
				// 如果格式化失败，使用基本错误信息
			}
		}

		PyErr_Restore(type, value, traceback);
		PyErr_Clear();

		return errorMsg;
	}

	QContentBrowser::QContentBrowser(QWidget* parent /*= nullptr*/)
		: QWidget()
	{
		QHBoxLayout* layout = new QHBoxLayout(this);
		this->setLayout(layout);

		std::string path = getAssetPath();
		QDir root(path.c_str());

		//Add file browser
		model = new QFileSystemModel(this);
		model->setRootPath(path.c_str());
		model->setFilter(QDir::Dirs | QDir::NoDotAndDotDot);
		model->sort(0, Qt::AscendingOrder);

//		QObject::connect(model, SIGNAL(directoryLoaded(QString)), SLOT(findDirectory(QString)));

		// File system
		treeView = new QTreeView(this);
		treeView->setModel(model);
		treeView->setHeaderHidden(true);	//Show tree header
		treeView->setFixedWidth(520);
		treeView->hideColumn(1);
		treeView->hideColumn(2);			//Hide second column
		treeView->hideColumn(3);
		treeView->setRootIndex(model->index(path.c_str()));
		layout->addWidget(treeView);



		QStringList filter;
		filter <<"*.png" << "*.jpg" << "*.bmp" << "*.obj" << "*.gltf" << "*.glb" << "*.fbx" << "*.STL" << "*.stl" << "*.xml";
		auto* listModel = new CustomFileSystemModel(this);
		listModel->setRootPath(path.c_str());
		listModel->setFilter(QDir::Files | QDir::NoDotAndDotDot);
		listModel->setNameFilters(filter);
		listModel->setNameFilterDisables(false);
		listModel->sort(0, Qt::AscendingOrder);

		listView = new QListView;
		listView->setModel(listModel);
		listView->setViewMode(QListView::IconMode);
		listView->setIconSize(QSize(80, 80));
		listView->setGridSize(QSize(120, 120));
		listView->setUniformItemSizes(true);
		listView->setResizeMode(QListWidget::Adjust);
		listView->setTextElideMode(Qt::ElideRight);
		listView->setRootIndex(listModel->index(path.c_str()));
		layout->addWidget(listView);

		connect(treeView, SIGNAL(clicked(const QModelIndex&)),
			this, SLOT(treeItemSelected(const QModelIndex&)));

		connect(listView, SIGNAL(clicked(const QModelIndex&)),
			this, SLOT(assetItemSelected(const QModelIndex&)));

		connect(listView, SIGNAL(doubleClicked(const QModelIndex&)),
			this, SLOT(assetDoubleClicked(const QModelIndex&)));
	}

	void QContentBrowser::treeItemSelected(const QModelIndex& index)
	{
		QString name = model->fileName(index);
		QString path = model->fileInfo(index).absolutePath() + "/" + name;

		//A hack
		QStringList filter;
		filter << "*.png" << "*.jpg" << "*.bmp" << "*.obj" << "*.gltf" << "*.glb" << "*.fbx" << "*.STL" << "*.stl" << "*.xml";
		auto* newListModel = new CustomFileSystemModel(this);
		newListModel->setRootPath(path);
		newListModel->setFilter(QDir::Files | QDir::NoDotAndDotDot);
		newListModel->setNameFilters(filter);
		newListModel->setNameFilterDisables(false);
		newListModel->sort(0, Qt::AscendingOrder);

 		listView->setModel(newListModel);
		listView->setRootIndex(newListModel->index(path));
	}

	void QContentBrowser::assetItemSelected(const QModelIndex& index)
	{
		QString name = model->fileName(index);
		QString path = model->fileInfo(index).absolutePath() + "/" + name;


	}

	void QContentBrowser::assetDoubleClicked(const QModelIndex& index)
	{
		QString name = model->fileName(index);
		QString path = model->fileInfo(index).absolutePath() + "/" + name;

		std::cout << path.toStdString() << "\n";

		auto ext = model->fileInfo(index).suffix().toStdString();
		auto ext2Act = NodeFactory::instance()->nodeContentActions();
		if (ext2Act.find(ext) != ext2Act.end())
		{
			auto func = ext2Act[ext];
			if (func != nullptr) {
				auto node = func(path.toStdString());

				emit nodeCreated(node);
			}
		}
	}
}