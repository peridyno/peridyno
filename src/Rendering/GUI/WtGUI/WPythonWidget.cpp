#include "WPythonWidget.h"

#include <Wt/WJavaScript.h>
#include <Wt/WMessageBox.h>
#include <Wt/WPushButton.h>
#include <Wt/WVBoxLayout.h>

#include <SceneGraph.h>

// python
#include <pybind11/embed.h>
#include <SceneGraphFactory.h>
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
	auto jsignal = new Wt::JSignal<std::string>(mCodeEditor, "update");
	jsignal->connect(this, &WPythonWidget::execute);

	auto str = jsignal->createCall({ ref + ".editor.getValue()" });
	command = "function(object, event) {" + str + ";}";
	auto btn = layout->addWidget(std::make_unique<Wt::WPushButton>("Update"), 0);
	btn->clicked().connect(command);

	// some default code here...
	std::string source = R"====(# dyno sample
import PyPeridyno as dyno

class VolumeTest(dyno.Node):
    
    def __init__(self):
        dyno = __import__('PyPeridyno')
        super().__init__()
        self.state_LevelSet = dyno.FInstanceLevelSet3f("LevelSet", "", dyno.FieldTypeEnum.State, self)

        self.set_auto_hidden(True)
        mapper = dyno.VolumeToTriangleSet3f()
        self.state_level_set().connect(mapper.io_volume())
        self.graphics_pipeline().push_module(mapper)

        renderer = dyno.GLSurfaceVisualModule()
        mapper.out_triangle_set().connect(renderer.in_triangle_set())
        self.graphics_pipeline().push_module(renderer)
        
    def get_node_type(self):
        return "Volume"

    def state_level_set(self):
        return self.state_LevelSet


scn = dyno.SceneGraph()

test = VolumeTest()
scn.add_node(test)
)====";

	setText(source);
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
		
		auto globals = py::globals();
		py::exec("import QtPathHelper; import PyPeridyno as dyno", globals);

		auto locals = py::dict();
		locals["scn"] = mScene;

		py::exec(src, py::globals(), locals);

		printf("C++");
		printf("%d", mScene->getFrameNumber());

		//if (locals.contains("scn"))
		//{
		//	/*auto scene = locals["scn"].cast<std::shared_ptr<dyno::SceneGraph>>();
		//	if (scene) mSignal.emit(scene);*/
		//}
		//else
		//{
		//	Wt::WMessageBox::show("Error", "Please define 'scn = dyno.SceneGraph()'", Wt::StandardButton::Ok);
		//}
	}
	catch (const std::exception& e) {
		Wt::WMessageBox::show("Error", e.what(), Wt::StandardButton::Ok);
		flag = false;
	}
}