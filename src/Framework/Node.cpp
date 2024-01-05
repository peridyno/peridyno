#include "Node.h"
#include "Action.h"

#include "SceneGraph.h"

namespace dyno
{
Node::Node()
	: OBase()
	, m_node_name("default")
	, m_dt(0.016f)
	, m_mass(1.0f)
{
}


Node::~Node()
{
	m_module_list.clear();

// 	for (auto port : mImportNodes)
// 	{
// 		auto& nodes = port->getNodes();
// 		for (auto node : nodes)
// 		{
// 			node->disconnect(port);
// 		}
// 	}

	for (auto port : mExportNodes)
	{
		this->disconnect(port);
	}

	mImportNodes.clear();
	mExportNodes.clear();
}

void Node::setName(std::string name)
{
	m_node_name = name;
}

std::string Node::getName()
{
	return m_node_name;
}

std::string Node::getNodeType()
{
	return "Default";
}

bool Node::isAutoSync()
{
	return mAutoSync;
}

void Node::setAutoSync(bool con)
{
	mAutoSync = con;
}

bool Node::canExported()
{
	return mExported;
}

void Node::allowExported(bool ex)
{
	mExported = ex;
}

bool Node::isActive()
{
	return mPhysicsEnabled;
}

void Node::setActive(bool active)
{
	mPhysicsEnabled = active;
}

bool Node::isVisible()
{
	return mRenderingEnabled;
}

void Node::setVisible(bool visible)
{
	mRenderingEnabled = visible;

	if (visible)
		this->graphicsPipeline()->enable();
	else
		this->graphicsPipeline()->disable();
}

float Node::getDt()
{
	return m_dt;
}

void Node::setDt(Real dt)
{
	m_dt = dt;
}

void Node::setSceneGraph(SceneGraph* scn)
{
	mSceneGraph = scn;
}

SceneGraph* Node::getSceneGraph()
{
	return mSceneGraph;
}

// Node* Node::addAncestor(Node* anc)
// {
// 	if (hasAncestor(anc) || anc == nullptr)
// 		return nullptr;
// 
// 	anc->addDescendant(this);
// 
// 	mAncestors.push_back(anc);
// 
// 	if (mSceneGraph) {
// 		mSceneGraph->markQueueUpdateRequired();
// 	}
// 
// 	return anc;
// }
// 
// bool Node::hasAncestor(Node* anc)
// {
// 	auto it = find(mAncestors.begin(), mAncestors.end(), anc);
// 
// 	return it == mAncestors.end() ? false : true;
// }

void Node::preUpdateStates()
{

}

void Node::updateStates()
{
	this->animationPipeline()->update();
}

void Node::update()
{
	if (this->validateInputs())
	{
		this->preUpdateStates();

		if (mPhysicsEnabled)
			this->updateStates();

		this->postUpdateStates();

		this->updateTopology();
	}
}

void Node::reset()
{
	if (this->validateInputs()) {
		this->resetStates();

		//When the node is reset, call tick() to force updating all modules
		this->tick();
	}
}

NBoundingBox Node::boundingBox()
{
	return NBoundingBox();
}

void Node::postUpdateStates()
{

}

void Node::updateGraphicsContext()
{
	this->graphicsPipeline()->update();
}

void Node::resetStates()
{
	this->stateElapsedTime()->setValue(0.0f);
	this->stateFrameNumber()->setValue(0);
}

bool Node::validateInputs()
{
	//If any input field is empty, return false;
	for(auto f_in : fields_input)
	{
		if (!f_in->isOptional() && f_in->isEmpty())
		{
			std::string errMsg = std::string("The field ") + f_in->getObjectName() +
				std::string(" in Node ") + this->getClassInfo()->getClassName() + std::string(" is not set!");

			Log::sendMessage(Log::Info, errMsg);
			return false;
		}
	}

	return true;
}

void Node::tick()
{
	std::vector<FBase*>& fields = this->getAllFields();
	for(FBase * var : fields)
	{
		if (var != nullptr) {
			if (var->getFieldType() == FieldTypeEnum::State || var->getFieldType() == FieldTypeEnum::Out)
			{
				var->tick();
			}
		}
	}
}

// std::shared_ptr<DeviceContext> Node::getContext()
// {
// 	if (m_context == nullptr)
// 	{
// 		m_context = TypeInfo::New<DeviceContext>();
// 		m_context->setParent(this);
// 		addModule(m_context);
// 	}
// 	return m_context;
// }
// 
// void Node::setContext(std::shared_ptr<DeviceContext> context)
// {
// 	if (m_context != nullptr)
// 	{
// 		deleteModule(m_context);
// 	}
// 
// 	m_context = context; 
// 	addModule(m_context);
// }

std::shared_ptr<AnimationPipeline> Node::animationPipeline()
{
	if (m_animation_pipeline == nullptr)
	{
		m_animation_pipeline = std::make_shared<AnimationPipeline>(this);
	}
	return m_animation_pipeline;
}

std::shared_ptr<GraphicsPipeline> Node::graphicsPipeline()
{
	if (m_render_pipeline == nullptr)
	{
		m_render_pipeline = std::make_shared<GraphicsPipeline>(this);
	}
	return m_render_pipeline;
}

/*
std::shared_ptr<MechanicalState> Node::getMechanicalState()
{
	if (m_mechanical_state == nullptr)
	{
		m_mechanical_state = TypeInfo::New<MechanicalState>();
		m_mechanical_state->setParent(this);
	}
	return m_mechanical_state;
}*/
/*
bool Node::addModule(std::string name, Module* module)
{
	if (getContext() == nullptr || module == NULL)
	{
		std::cout << "Context or module does not exist!" << std::endl;
		return false;
	}

	std::map<std::string, Module*>::iterator found = m_modules.find(name);
	if (found != m_modules.end())
	{
		std::cout << "Module name already exists!" << std::endl;
		return false;
	}
	else
	{
		m_modules[name] = module;
		m_module_list.push_back(module);

//		module->insertToNode(this);
	}

	return true;
}
*/
bool Node::addModule(std::shared_ptr<Module> module)
{
	bool ret = true;
	ret &= addToModuleList(module);

	return ret;
}

void Node::initialize()
{
	this->resetStates();

	this->animationPipeline()->updateExecutionQueue();
	this->graphicsPipeline()->updateExecutionQueue();
}

bool Node::deleteModule(std::shared_ptr<Module> module)
{
	bool ret = true;

	ret &= deleteFromModuleList(module);
		
	return ret;
}

// void Node::doTraverseBottomUp(Action* act)
// {
// 	act->start(this);
// 	auto iter = mAncestors.begin();
// 	for (; iter != mAncestors.end(); iter++)
// 	{
// 		(*iter)->doTraverseBottomUp(act);
// 	}
// 
// 	act->process(this);
// 
// 	act->end(this);
// }
// 
// void Node::doTraverseTopDown(Action* act)
// {
// 	act->start(this);
// 	act->process(this);
// 
// 	auto iter = mAncestors.begin();
// 	for (; iter != mAncestors.end(); iter++)
// 	{
// 		(*iter)->doTraverseTopDown(act);
// 	}
// 
// 	act->end(this);
// }


std::string FormatConnectionInfo(Node* node, NodePort* port, bool connecting, bool succeeded)
{
	Node* pOut = port != nullptr ? port->getParent() : nullptr;

	std::string capIn = node->caption();
	std::string capOut = pOut != nullptr ? pOut->caption() : "";

	std::string nameIn = node->getName();
	std::string nameOut = port != nullptr ? port->getPortName() : "";

	if (connecting)
	{
		std::string message1 = capIn + ":" + nameIn + " is connected to " + capOut + ":" + nameOut;
		std::string message2 = capIn + ":" + nameIn + " cannot be connected to " + capOut + ":" + nameOut;
		return succeeded ? message1 : message2;
	}
	else
	{
		std::string message1 = capIn + ":" + nameIn + " is disconnected from " + capOut + ":" + nameOut;
		std::string message2 = capIn + ":" + nameIn + " cannot be disconnected from " + capOut + ":" + nameOut;
		return succeeded ? message1 : message2;
	}
}

bool Node::appendExportNode(NodePort* nodePort)
{
	auto it = find(mExportNodes.begin(), mExportNodes.end(), nodePort);
	if (it != mExportNodes.end()) {
		Log::sendMessage(Log::Info, FormatConnectionInfo(this, nodePort, true, false));
		return false;
	}

	mExportNodes.push_back(nodePort);

	Log::sendMessage(Log::Info, FormatConnectionInfo(this, nodePort, true, true));
	return nodePort->addNode(this);
}

bool Node::removeExportNode(NodePort* nodePort)
{
	//TODO: this is a hack, otherwise the app will crash
	if (mExportNodes.size() == 0) {
		return false;
	}

	auto it = find(mExportNodes.begin(), mExportNodes.end(), nodePort);
	if (it == mExportNodes.end()) {
		Log::sendMessage(Log::Info, FormatConnectionInfo(this, nodePort, false, false));
		return false;
	}

	mExportNodes.erase(it);

	Log::sendMessage(Log::Info, FormatConnectionInfo(this, nodePort, false, true));
	return nodePort->removeNode(this);
}

void Node::updateTopology()
{

}

// void Node::traverseBottomUp(Action* act)
// {
// 	doTraverseBottomUp(act);
// }
// 
// void Node::traverseTopDown(Action* act)
// {
// 	doTraverseTopDown(act);
// }

bool Node::connect(NodePort* nPort)
{
	if (!canExported())
	{
		Log::sendMessage(Log::Error, this->getClassInfo()->getClassName() + "can not be exported");
		return false;
	}

	nPort->notify();

	return this->appendExportNode(nPort);
}

bool Node::disconnect(NodePort* nPort)
{
	return this->removeExportNode(nPort);
}

bool Node::attachField(FBase* field, std::string name, std::string desc, bool autoDestroy /*= true*/)
{
	field->setParent(this);
	field->setObjectName(name);
	field->setDescription(desc);
	field->setAutoDestroy(autoDestroy);

	bool ret = false;
	
	auto fType = field->getFieldType();
	switch (field->getFieldType())
	{
	case FieldTypeEnum::State:
		ret = this->addField(field);
		break;

	case FieldTypeEnum::Param:
		ret = addParameter(field);
		break;

	case FieldTypeEnum::In:
		ret = addInputField(field);
		break;

	case FieldTypeEnum::Out:
		ret = addOutputField(field);
		break;

	default:
		break;
	}
	

	if (!ret)
	{
		Log::sendMessage(Log::Error, std::string("The field ") + name + std::string(" already exists!"));
	}
	return ret;
}

uint Node::sizeOfImportNodes() const
{
	uint n = 0;
	for(auto port : mImportNodes)
	{
		n += port->getNodes().size();
	}

	return n;
}

// Node* Node::addDescendant(Node* descent)
// {
// 	if (hasDescendant(descent) || descent == nullptr)
// 		return descent;
// 
// 	mDescendants.push_back(descent);
// 	return descent;
// }
// 
// bool Node::hasDescendant(Node* descent)
// {
// 	auto it = std::find(mDescendants.begin(), mDescendants.end(), descent);
// 	return it == mDescendants.end() ? false : true;
// }
// 
// void Node::removeDescendant(Node* descent)
// {
// 	auto iter = mDescendants.begin();
// 	for (; iter != mDescendants.end(); )
// 	{
// 		if (*iter == descent)
// 		{
// 			mDescendants.erase(iter++);
// 		}
// 		else
// 		{
// 			++iter;
// 		}
// 	}
// }

bool Node::addNodePort(NodePort* port)
{
	mImportNodes.push_back(port);

	return true;
}

// void Node::setAsCurrentContext()
// {
// 	getContext()->enable();
// }

// void Node::setTopologyModule(std::shared_ptr<TopologyModule> topology)
// {
// 	if (m_topology != nullptr)
// 	{
// 		deleteModule(m_topology);
// 	}
// 	m_topology = topology;
// 	addModule(topology);
// }
// 
// void Node::setNumericalModel(std::shared_ptr<NumericalModel> numerical)
// {
// 	if (m_numerical_model != nullptr)
// 	{
// 		deleteModule(m_numerical_model);
// 	}
// 	m_numerical_model = numerical;
// 	addModule(numerical);
// }
// 
// void Node::setCollidableObject(std::shared_ptr<CollidableObject> collidable)
// {
// 	if (m_collidable_object != nullptr)
// 	{
// 		deleteModule(m_collidable_object);
// 	}
// 	m_collidable_object = collidable;
// 	addModule(collidable);
// }

std::shared_ptr<Module> Node::getModule(std::string name)
{
	std::shared_ptr<Module> base = nullptr;
	std::list<std::shared_ptr<Module>>::iterator iter;
	for (iter = m_module_list.begin(); iter != m_module_list.end(); iter++)
	{
		if ((*iter)->getName() == name)
		{
			base = *iter;
			break;
		}
	}
	return base;
}

bool Node::hasModule(std::string name)
{
	if (getModule(name) == nullptr)
		return false;

	return true;
}

/*Module* Node::getModule(std::string name)
{
	std::map<std::string, Module*>::iterator result = m_modules.find(name);
	if (result == m_modules.end())
	{
		return NULL;
	}

	return result->second;
}*/


bool Node::addToModuleList(std::shared_ptr<Module> module)
{
	auto found = std::find(m_module_list.begin(), m_module_list.end(), module);
	if (found == m_module_list.end())
	{
		m_module_list.push_back(module);
		module->setParentNode(this);
		return true;
	}

	return false;
}

bool Node::deleteFromModuleList(std::shared_ptr<Module> module)
{
	auto found = std::find(m_module_list.begin(), m_module_list.end(), module);
	if (found != m_module_list.end())
	{
		m_module_list.erase(found);
		return true;
	}

	return true;
}

}