#include "WPythonWidget.h"

#include <Wt/WJavaScript.h>
#include <Wt/WMessageBox.h>
#include <Wt/WPushButton.h>
#include <Wt/WVBoxLayout.h>
#include <Wt/WHBoxLayout.h>

#include <SceneGraph.h>

// python
#include <pybind11/embed.h>
#include <SceneGraphFactory.h>
#include <iomanip>
namespace py = pybind11;

WPythonWidget::WPythonWidget()
{
	this->setLayoutSizeAware(true);
	this->setOverflow(Wt::Overflow::Auto);
	//this->setHeight(Wt::WLength("100%"));
	this->setMargin(0);

	auto layout = this->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);

	mCodeEditor = layout->addWidget(std::make_unique<Wt::WText>(), 1);
	mCodeEditor->setInline(false);
	//mCodeEditor->setWidth(Wt::WLength("100%"));
	

	// ACE editor
	std::string ref = mCodeEditor->jsRef(); // is a text string that will be the element when executed in JS

	std::string command =
		ref + ".editor = ace.edit(" + ref + ");" +
		ref + ".editor.setTheme(\"ace/theme/monokai\");" +
		ref + ".editor.getSession().setMode(\"ace/mode/python\");" +
		ref + ".editor.setFontSize(14);" +
		"ace.require(\"ace/ext/language_tools\");" +
		ref + ".editor.setOptions({enableBasicAutocompletion: true,enableSnippets : true,enableLiveAutocompletion : true});" +
		ref + ".editor.setOption(\"wrap\",\"free\")";
	mCodeEditor->doJavaScript(command);

	// create signal
	auto jsignalExecute = new Wt::JSignal<std::string>(mCodeEditor, "execute");
	jsignalExecute->connect(this, &WPythonWidget::execute);

	auto strExecute = jsignalExecute->createCall({ ref + ".editor.getValue()" });
	std::string commandExecute = "function(object, event) {" + strExecute + ";}";

	auto jsignalUpload = new Wt::JSignal<std::string>(mCodeEditor, "upload");
	jsignalUpload->connect(this, &WPythonWidget::upload);

	auto strUpload = jsignalUpload->createCall({ ref + ".editor.getValue()" });
	std::string commandUpload = "function(object, event) {" + strUpload + ";}";

	auto btnContainer = layout->addWidget(std::make_unique<Wt::WContainerWidget>(), 0);
	auto btnLayout = btnContainer->setLayout(std::make_unique<Wt::WHBoxLayout>());
	btnLayout->setContentsMargins(0, 0, 0, 0);

	mOutputArea = layout->addWidget(std::make_unique<Wt::WText>(), 1);

	outRef = mOutputArea->jsRef(); // is a text string that will be the element when executed in JS

	std::string outCommand =
		outRef + ".editor = ace.edit(" + outRef + ");" +
		outRef + ".editor.setTheme(\"ace/theme/monokai\");" +
		outRef + ".editor.getSession().setMode(\"ace/mode/python\");" +
		outRef + ".editor.setFontSize(14);" +
		"ace.require(\"ace/ext/language_tools\");" +
		outRef + ".editor.setOptions({highlightActiveLine: false, highlightGutterLine: false, showPrintMargin: false, showLineNumbers: false, cursorStyle: 'wide', scrollPastEnd: false, useWorker: false, showGutter: false, vScrollBarAlwaysVisible: true});" +
		outRef + ".editor.setReadOnly(true);" +
		outRef + ".editor.renderer.$cursorLayer.element.style.opacity = '0';" +
		outRef + ".editor.setOption(\"wrap\",\"free\")";
	mOutputArea->doJavaScript(outCommand);

	// some default code here...
	std::string source = R"====(# dyno sample
import PyPeridyno_Modeling as dyno_Modeling

scn = dyno.SceneGraph()

gltf = dyno_Modeling.GltfLoader3f()
scn.addNode(gltf)

gltf.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "Jeep/JeepGltf/jeep.gltf"))

print("!!")
)====";

	setText(source);

	auto btnExecute = btnLayout->addWidget(std::make_unique<Wt::WPushButton>("Execute"), 0);
	auto btnUpload = btnLayout->addWidget(std::make_unique<Wt::WPushButton>("Upload"), 0);
	auto btnClear = btnLayout->addWidget(std::make_unique<Wt::WPushButton>("Clear"), 0);

	btnExecute->clicked().connect(commandExecute);
	btnUpload->clicked().connect(commandUpload);
	btnClear->clicked().connect([this]() {
		this->clear();
		});
}

WPythonWidget::~WPythonWidget()
{
	Wt::log("warning") << "WPythonWidget destory";
}

void WPythonWidget::setText(const std::string& text)
{
	// update code editor content
	std::string ref = mCodeEditor->jsRef();
	std::string command = ref + ".editor.setValue(`" + text + "`, 1);";
	mCodeEditor->doJavaScript(command);
	//mCodeEditor->refresh();
}

void WPythonWidget::execute(const std::string& src)
{
	bool flag = true;
	py::scoped_interpreter guard{};

	try {
		// timestamp
		auto now = std::chrono::system_clock::now();
		auto time_t = std::chrono::system_clock::to_time_t(now);
		std::stringstream timestamp_ss;
		timestamp_ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
		std::string exec_timestamp = "[" + timestamp_ss.str() + "]>";

		auto sys_stdout = py::module_::import("sys").attr("stdout");
		auto sys_stderr = py::module_::import("sys").attr("stderr");

		auto stringio = py::module_::import("io").attr("StringIO")();
		auto stringio_err = py::module_::import("io").attr("StringIO")();

		py::module_::import("sys").attr("stdout") = stringio;
		py::module_::import("sys").attr("stderr") = stringio_err;
		
		auto globals = py::globals();
		py::exec("import QtPathHelper; import PyPeridyno as dyno", globals);

		auto locals = py::dict();
		locals["scn"] = mScene;

		py::exec(src, py::globals(), locals);

		py::module_::import("sys").attr("stdout") = sys_stdout;
		py::module_::import("sys").attr("stderr") = sys_stderr;

		std::string captured_output = py::str(stringio.attr("getvalue")()).cast<std::string>();
		std::string captured_error = py::str(stringio_err.attr("getvalue")()).cast<std::string>();
		
		std::string output = exec_timestamp + captured_output;

		outputRecord += output;

		sendToOutputArea(outputRecord);
	}
	catch (const std::exception& e) {
		py::module_::import("sys").attr("stdout") = py::module_::import("sys").attr("stdout"); 
		py::module_::import("sys").attr("stderr") = py::module_::import("sys").attr("stderr");

		Wt::WMessageBox::show("Error", e.what(), Wt::StandardButton::Ok);
		flag = false;
	}
}

void WPythonWidget::upload(const std::string& src)
{
	bool flag = true;
	py::scoped_interpreter guard{};

	try {

		auto globals = py::globals();
		py::exec("import QtPathHelper; import PyPeridyno as dyno", globals);

		auto locals = py::dict();
		locals["scn"] = mScene;

		py::exec(src, py::globals(), locals);

		if (locals.contains("scn"))
		{
			auto scene = locals["scn"].cast<std::shared_ptr<dyno::SceneGraph>>();
			if (scene) mSignal.emit(scene);
		}
		else
		{
			Wt::WMessageBox::show("Error", "Please define 'scn = dyno.SceneGraph()'", Wt::StandardButton::Ok);
		}
	}
	catch (const std::exception& e) {
		Wt::WMessageBox::show("Error", e.what(), Wt::StandardButton::Ok);
		flag = false;
	}
}

void WPythonWidget::clear()
{
	outputRecord = "";
	sendToOutputArea(outputRecord);		
}

void WPythonWidget::sendToOutputArea(std::string src)
{
	std::string js = outRef + ".editor.setValue(" + Wt::WString(outputRecord).jsStringLiteral() + ", -1);" +
		"var rows = " + outRef + ".editor.getSession().getLength();" +
		outRef + ".editor.scrollToRow(rows - 1);";

	if(mOutputArea)
		mOutputArea->doJavaScript(js);
}
