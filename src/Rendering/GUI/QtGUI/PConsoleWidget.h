#ifndef QCONSOLEWIDGET_H
#define QCONSOLEWIDGET_H

#include <QWidget>
#include <QFileSystemModel>
#include <QTreeView>
#include <QListView>

#include <QTextEdit.h>
#include <QPushButton.h>
#include <QMessageBox.h>

#include <Qsci/qsciscintilla.h>
#include <Qsci/qscilexerpython.h>


// Slots macro definition conflicts with Python
#ifdef slots
#undef slots
#endif

// python
#include <pybind11/embed.h>
namespace py = pybind11;

#define slots Q_SLOTS



namespace dyno
{
	class Node;

	class PConsoleWidget : public QWidget
	{
		Q_OBJECT
	public:
		explicit PConsoleWidget(QWidget *parent = nullptr);


	signals:

	public slots:
		void execute(const std::string& src);

	private:
		std::string getPythonErrorDetails();

	private:
		QsciScintilla* mCodeEditor;
		QsciLexerPython* mPythonLexer;
		QPushButton* updateButton;
	};

	class QContentBrowser : public QWidget
	{
		Q_OBJECT
	public:
		explicit QContentBrowser(QWidget* parent = nullptr);

	signals:

	Q_SIGNALS:
		void nodeCreated(std::shared_ptr<Node> node);

	public slots:
		void treeItemSelected(const QModelIndex& index);
		void assetItemSelected(const QModelIndex& index);
		void assetDoubleClicked(const QModelIndex& index);

	private:
		QFileSystemModel* model;
		QTreeView* treeView;
		QListView* listView;
	};
}

#endif // QCONSOLEWIDGET_H
