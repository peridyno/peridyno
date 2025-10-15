#ifndef QCONSOLEWIDGET_H
#define QCONSOLEWIDGET_H

#include <QWidget>
#include <QFileSystemModel>
#include <QTreeView>
#include <QListView>

#include <QTextEdit.h>
#include <QPushButton.h>
<<<<<<< HEAD
<<<<<<< HEAD
=======
#include <QMessageBox.h>

#include <Qsci/qsciscintilla.h>
#include <Qsci/qscilexerpython.h>
#include <Qsci/qsciapis.h>
#include <QApplication>
#include <QWheelEvent>
>>>>>>> a78b0e817c5a4a4db2f4998dfdb78e8f19b327cd

=======
#include <QMessageBox.h>
>>>>>>> dev

// Slots macro definition conflicts with Python
#ifdef slots
#undef slots
#endif

<<<<<<< HEAD
// python
#include <pybind11/embed.h>
namespace py = pybind11;

#define slots Q_SLOTS

=======
#ifdef PERIDYNO_QT_PYTHON_CONSOLE
// python
#include <pybind11/embed.h>
namespace py = pybind11;
#endif

#define slots Q_SLOTS

#ifdef PERIDYNO_QT_PYTHON_CONSOLE

#include <Qsci/qsciscintilla.h>
#include <Qsci/qscilexerpython.h>
#include <Qsci/qsciapis.h>

#endif //PERIDYNO_QT_PYTHON_CONSOLE
>>>>>>> dev


namespace dyno
{
	class PConsoleWidget : public QWidget
	{
		Q_OBJECT
	public:
		explicit PConsoleWidget(QWidget *parent = nullptr);
		~PConsoleWidget();


	signals:

<<<<<<< HEAD
	public slots:
=======
#ifdef PERIDYNO_QT_PYTHON_CONSOLE
	public Q_SLOTS:
>>>>>>> dev
		void execute(const std::string& src);

	private:
		std::string getPythonErrorDetails();

<<<<<<< HEAD
<<<<<<< HEAD
=======
		void applyDarkTheme(QsciLexerPython* lexer);

		void applyLightTheme(QsciLexerPython* lexer);

>>>>>>> a78b0e817c5a4a4db2f4998dfdb78e8f19b327cd
	private:
		QsciScintilla* mCodeEditor;
		QsciLexerPython* mPythonLexer;
		QPushButton* updateButton;
=======
		void applyDarkTheme(QsciLexerPython* lexer);

		void applyLightTheme(QsciLexerPython* lexer);

	private:
		QsciScintilla* mCodeEditor;
		QsciLexerPython* mPythonLexer;
		QPushButton* updateButton;

#endif //PERIDYNO_QT_PYTHON_CONSOLE
>>>>>>> dev
	};
}




namespace dyno
{
	class Node;

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
