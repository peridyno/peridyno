#include "Node.h"
#include "Action.h"

#include "SceneGraph.h"
#include "Timer.h"

#include <sstream>
#include <iomanip>

namespace dyno
{
Node::Node()
	: OBase()
	, m_node_name("default")
	, mDt(0.016f)
{
}


Node::~Node()
{
	mModuleList.clear();

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

bool Node::isAutoHidden()
{
	return mAutoHidden;
}

void Node::setAutoSync(bool con)
{
	mAutoSync = con;
}

void Node::setAutoHidden(bool con)
{
	mAutoHidden = con;
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
}

float Node::getDt()
{
	return mDt;
}

void Node::setDt(Real dt)
{
	mDt = dt;
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
	if (!this->validateInputs()) {
		return;
	}

	if (this->requireUpdate())
	{
		mSyncRenderAndSim.lock();

		this->preUpdateStates();

		if (mPhysicsEnabled)
			this->updateStates();

		this->postUpdateStates();

		this->updateTopology();

		//reset parameters
		for (auto param : fields_param)
		{
			param->tack();
		}

		//reset input fields
		for (auto f_in : fields_input)
		{
			f_in->tack();
		}

		//tag all output fields as modifed
		for (auto f_out : fields_output)
		{
			f_out->tick();
		}

		mSyncRenderAndSim.unlock();
	}


}

void Node::reset()
{
	if (this->validateInputs()) {

		CTimer timer;

		auto scn = this->getSceneGraph();
		if (scn != nullptr && scn->isNodeInfoPrintable()) {
			timer.start();
		}

		this->stateElapsedTime()->setValue(0.0f);
		this->stateFrameNumber()->setValue(0);

		this->resetStates();

		//When the node is reset, call tick() to force updating all modules
		this->tick();

		if (scn != nullptr && scn->isNodeInfoPrintable()) {
			timer.stop();

			std::stringstream name;
			std::stringstream ss;
			name << std::setw(40) << this->getClassInfo()->getClassName();
			ss << std::setprecision(10) << timer.getElapsedTime();

			std::string info = "Node: \t" + name.str() + ": \t " + ss.str() + "ms \n";
			Log::sendMessage(Log::Info, info);
		}
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
	if (mRenderingEnabled)
	{
		if (mSyncRenderAndSim.try_lock()) {
			this->graphicsPipeline()->update();
			mSyncRenderAndSim.unlock();
		}
	}
}

void Node::resetStates()
{
	this->resetPipeline()->update();
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

bool Node::requireUpdate()
{
	//TODO: improve the following rules
	if (mForceUpdate)
		return true;

	//check input fields
	bool modified = false;

	if (mImportNodes.size() > 0)
	{
		return true;
	}
	 

	for (auto f_in : fields_input)
	{
		modified |= f_in->isModified();
	}

	//check control fields
	for (auto var : fields_param)
	{
		modified |= var->isModified();
	}

	return modified;
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

std::shared_ptr<Pipeline> Node::resetPipeline()
{
	if (mResetPipeline == nullptr)
	{
		mResetPipeline = std::make_shared<AnimationPipeline>(this);
	}
	return mResetPipeline;
}

std::shared_ptr<AnimationPipeline> Node::animationPipeline()
{
	if (mAnimationPipeline == nullptr)
	{
		mAnimationPipeline = std::make_shared<AnimationPipeline>(this);
	}
	return mAnimationPipeline;
}

std::shared_ptr<GraphicsPipeline> Node::graphicsPipeline()
{
	if (mGraphicsPipeline == nullptr)
	{
		mGraphicsPipeline = std::make_shared<GraphicsPipeline>(this);
	}
	return mGraphicsPipeline;
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

	//Always show the last node
	if (mAutoHidden)
		this->setVisible(false);

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

	//Recover the visibility
	if (mAutoHidden)
		this->setVisible(true);

	Log::sendMessage(Log::Info, FormatConnectionInfo(this, nodePort, false, true));
	return nodePort->removeNode(this);
}

void Node::updateTopology()
{

}

bool Node::connect(NodePort* nPort)
{
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

void Node::setForceUpdate(bool b)
{
	mForceUpdate = b;
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
	for (iter = mModuleList.begin(); iter != mModuleList.end(); iter++)
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
	auto found = std::find(mModuleList.begin(), mModuleList.end(), module);
	if (found == mModuleList.end())
	{
		mModuleList.push_back(module);
		module->setParentNode(this);
		return true;
	}

	return false;
}

bool Node::deleteFromModuleList(std::shared_ptr<Module> module)
{
	auto found = std::find(mModuleList.begin(), mModuleList.end(), module);
	if (found != mModuleList.end())
	{
		mModuleList.erase(found);
		return true;
	}

	return true;
}

}