#include "WPythonWidget.h"

#include <Wt/WTextArea.h>
#include <Wt/WPushButton.h>
#include <Wt/WVBoxLayout.h>
#include <Wt/WMessageBox.h>
#include <Wt/WJavaScript.h>

#include <SceneGraph.h>

// python
#include <pybind11/embed.h>
namespace py = pybind11;

WPythonWidget::WPythonWidget()
{
	this->setLayoutSizeAware(true);
	this->setOverflow(Wt::Overflow::Auto);
	this->setHeight(Wt::WLength("100%"));
	this->setMargin(0);

	auto layout = this->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);

	mCodeEditor = layout->addWidget(std::make_unique<Wt::WText>(), 1);
	mCodeEditor->setInline(false);
	mCodeEditor->setWidth(Wt::WLength("640px"));

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

scene = dyno.SceneGraph()

emitter = dyno.SquareEmitter3f()
emitter.var_location().set_value(dyno.Vector3f([0.5, 0.5, 0.5]))

fluid = dyno.ParticleFluid3f()
fluid.load_particles(dyno.Vector3f([0, 0, 0]), dyno.Vector3f([0.2, 0.2, 0.2]), 0.05)

emitter.connect(fluid.import_particle_emitters())

calculateNorm = dyno.CalculateNorm3f()
colorMapper = dyno.ColorMapping3f()
colorMapper.var_max().set_value(0.5)

ptRender = dyno.GLPointVisualModule()
ptRender.set_color(dyno.Color(1, 0, 0))
ptRender.set_color_map_mode(ptRender.ColorMapMode.PER_VERTEX_SHADER)

fluid.state_velocity().connect(calculateNorm.in_vec())
fluid.state_point_set().connect(ptRender.in_point_set())
calculateNorm.out_norm().connect(colorMapper.in_scalar())
colorMapper.out_color().connect(ptRender.in_color())

fluid.graphics_pipeline().push_module(calculateNorm)
fluid.graphics_pipeline().push_module(colorMapper)
fluid.graphics_pipeline().push_module(ptRender)

container = dyno.StaticBoundary3f()
container.load_cube(dyno.Vector3f([0, 0, 0]), dyno.Vector3f([1.0, 1.0, 1.0]), 0.02, True)

fluid.connect(container.import_particle_systems())

scene.add_node(emitter)
scene.add_node(fluid)
scene.add_node(container)
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
		auto locals = py::dict();
		py::exec(src, py::globals(), locals);

		if (locals.contains("scene"))
		{
			auto scene = locals["scene"].cast<std::shared_ptr<dyno::SceneGraph>>();
			if (scene) mSignal.emit(scene);
		}
		else
		{
			Wt::WMessageBox::show("Error", "Please define 'scene = dyno.SceneGraph()'", Wt::StandardButton::Ok);
		}
	}
	catch (const std::exception& e) {
		Wt::WMessageBox::show("Error", e.what(), Wt::StandardButton::Ok);
		flag = false;
	}

}
