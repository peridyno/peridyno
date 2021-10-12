#include "Node.h"
#include "NodeIterator.h"

#include "Action.h"


namespace dyno
{
IMPLEMENT_CLASS(Node)

Node::Node(std::string name)
	: OBase()
	, m_node_name(name)
	, m_dt(0.001f)
	, m_mass(1.0f)
{
	this->varScale()->setValue(Vec3f(1, 1, 1));
	this->varScale()->setMin(0.01);
	this->varScale()->setMax(100.0f);
}


Node::~Node()
{
	m_render_list.clear();
	m_module_list.clear();
}

void Node::setName(std::string name)
{
	m_node_name = name;
}

std::string Node::getName()
{
	return m_node_name;
}


Node* Node::getAncestor(std::string name)
{
	for (auto it = mAncestors.begin(); it != mAncestors.end(); ++it)
	{
		if ((*it)->getName() == name)
			return it->get();
	}
	return NULL;
}

bool Node::isControllable()
{
	return m_controllable;
}

void Node::setControllable(bool con)
{
	m_controllable = con;
}

bool Node::isActive()
{
	return this->varActive()->getData();
}

void Node::setActive(bool active)
{
	this->varActive()->setValue(active);
}

bool Node::isVisible()
{
	return this->varVisible()->getData();
}

void Node::setVisible(bool visible)
{
	this->varVisible()->setValue(visible);
}

float Node::getDt()
{
	return m_dt;
}

void Node::setDt(Real dt)
{
	m_dt = dt;
}

void Node::setMass(Real mass)
{
	m_mass = mass;
}

Real Node::getMass()
{
	return m_mass;
}

std::shared_ptr<Node> Node::addAncestor(std::shared_ptr<Node> anc)
{
	if (hasAncestor(anc) || anc == nullptr)
		return nullptr;

	anc->addDescendant(this);

	mAncestors.push_back(anc);
	return anc;
}

bool Node::hasAncestor(std::shared_ptr<Node> anc)
{
	auto it = find(mAncestors.begin(), mAncestors.end(), anc);

	return it == mAncestors.end() ? false : true;
}

// NodeIterator Node::begin()
// {
// 	return NodeIterator(this);
// }
// 
// NodeIterator Node::end()
// {
// 	return NodeIterator();
// }

void Node::removeAncestor(std::shared_ptr<Node> anc)
{
	auto iter = mAncestors.begin();
	for (; iter != mAncestors.end(); )
	{
		if (*iter == anc)
		{
			anc->removeDescendant(this);
			mAncestors.erase(iter++);
		}
		else
		{
			++iter;
		}
	}
}

void Node::removeAllAncestors()
{
	auto iter = mAncestors.begin();
	for (; iter != mAncestors.end(); )
	{
		(*iter)->removeDescendant(this);
		mAncestors.erase(iter++);
	}
}

void Node::preUpdateStates()
{

}

void Node::updateStates()
{
	this->animationPipeline()->update();
}

void Node::update()
{
	this->preUpdateStates();

	this->updateStates();

	this->postUpdateStates();

	this->updateTopology();
}

void Node::reset()
{
	this->resetStates();
}

void Node::postUpdateStates()
{

}

void Node::resetStates()
{

}

std::shared_ptr<DeviceContext> Node::getContext()
{
	if (m_context == nullptr)
	{
		m_context = TypeInfo::New<DeviceContext>();
		m_context->setParent(this);
		addModule(m_context);
	}
	return m_context;
}

void Node::setContext(std::shared_ptr<DeviceContext> context)
{
	if (m_context != nullptr)
	{
		deleteModule(m_context);
	}

	m_context = context; 
	addModule(m_context);
}

std::unique_ptr<AnimationPipeline>& Node::animationPipeline()
{
	if (m_animation_pipeline == nullptr)
	{
		m_animation_pipeline = std::make_unique<AnimationPipeline>(this);
	}
	return m_animation_pipeline;
}

std::unique_ptr<GraphicsPipeline>& Node::graphicsPipeline()
{
	if (m_render_pipeline == nullptr)
	{
		m_render_pipeline = std::make_unique<GraphicsPipeline>(this);
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

	std::string mType = module->getModuleType();
	if (std::string("TopologyModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::cast<TopologyModule>(module);
		m_topology = downModule;
	}
	else if (std::string("NumericalModel").compare(mType) == 0)
	{
		auto downModule = TypeInfo::cast<NumericalModel>(module);
		m_numerical_model = downModule;
	}
	else if (std::string("NumericalIntegrator").compare(mType) == 0)
	{
		auto downModule = TypeInfo::cast<NumericalIntegrator>(module);
		m_numerical_integrator = downModule;
	}
	else if (std::string("ForceModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::cast<ForceModule>(module);
		this->addToForceModuleList(downModule);
	}
	else if (std::string("ConstraintModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::cast<ConstraintModule>(module);
		this->addToConstraintModuleList(downModule);
	}
	else if (std::string("ComputeModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::cast<ComputeModule>(module);
		this->addToComputeModuleList(downModule);
	}
	else if (std::string("CollisionModel").compare(mType) == 0)
	{
		auto downModule = TypeInfo::cast<CollisionModel>(module);
		this->addToCollisionModelList(downModule);
	}
	else if (std::string("VisualModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::cast<VisualModule>(module);
		this->addToVisualModuleList(downModule);
	}
	else if (std::string("TopologyMapping").compare(mType) == 0)
	{
		auto downModule = TypeInfo::cast<TopologyMapping>(module);
		this->addToTopologyMappingList(downModule);
	}

	return ret;
}

void Node::initialize()
{
	this->resetStates();
}

bool Node::deleteModule(std::shared_ptr<Module> module)
{
	bool ret = true;

	ret &= deleteFromModuleList(module);

	std::string mType = module->getModuleType();

	if (std::string("TopologyModule").compare(mType) == 0)
	{
		m_topology = nullptr;
	}
	else if (std::string("NumericalModel").compare(mType) == 0)
	{
		m_numerical_model = nullptr;
	}
	else if (std::string("NumericalIntegrator").compare(mType) == 0)
	{
		m_numerical_integrator = nullptr;
	}
	else if (std::string("ForceModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::cast<ForceModule>(module);
		this->deleteFromForceModuleList(downModule);
	}
	else if (std::string("ConstraintModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::cast<ConstraintModule>(module);
		this->deleteFromConstraintModuleList(downModule);
	}
	else if (std::string("ComputeModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::cast<ComputeModule>(module);
		this->deleteFromComputeModuleList(downModule);
	}
	else if (std::string("CollisionModel").compare(mType) == 0)
	{
		auto downModule = TypeInfo::cast<CollisionModel>(module);
		this->deleteFromCollisionModelList(downModule);
	}
	else if (std::string("VisualModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::cast<VisualModule>(module);
		this->deleteFromVisualModuleList(downModule);
	}
	else if (std::string("TopologyMapping").compare(mType) == 0)
	{
		auto downModule = TypeInfo::cast<TopologyMapping>(module);
		this->deleteFromTopologyMappingList(downModule);
	}
		
	return ret;
}

void Node::doTraverseBottomUp(Action* act)
{
	act->start(this);
	auto iter = mAncestors.begin();
	for (; iter != mAncestors.end(); iter++)
	{
		(*iter)->traverseBottomUp(act);
	}

	act->process(this);

	act->end(this);
}

void Node::doTraverseTopDown(Action* act)
{
	act->start(this);
	act->process(this);

	auto iter = mAncestors.begin();
	for (; iter != mAncestors.end(); iter++)
	{
		(*iter)->doTraverseTopDown(act);
	}

	act->end(this);
}

void Node::updateTopology()
{

}

void Node::traverseBottomUp(Action* act)
{
	doTraverseBottomUp(act);
}

void Node::traverseTopDown(Action* act)
{
	doTraverseTopDown(act);
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
	case FieldTypeEnum::Current:
		ret = this->addField(field);
		break;

	case FieldTypeEnum::Param:
		ret = this->addField(field);

	default:
		break;
	}
	

	if (!ret)
	{
		Log::sendMessage(Log::Error, std::string("The field ") + name + std::string(" already exists!"));
	}
	return ret;
}

Node* Node::addDescendant(Node* descent)
{
	if (hasDescendant(descent) || descent == nullptr)
		return descent;

	mDescendants.push_back(descent);
	return descent;
}

bool Node::hasDescendant(Node* descent)
{
	auto it = std::find(mDescendants.begin(), mDescendants.end(), descent);
	return it == mDescendants.end() ? false : true;
}

void Node::removeDescendant(Node* descent)
{
	auto iter = mDescendants.begin();
	for (; iter != mDescendants.end(); )
	{
		if (*iter == descent)
		{
			mDescendants.erase(iter++);
		}
		else
		{
			++iter;
		}
	}
}

bool Node::addNodePort(NodePort* port)
{
	mNodePorts.push_back(port);

	return true;
}

void Node::setAsCurrentContext()
{
	getContext()->enable();
}

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
		module->setParent(this);
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