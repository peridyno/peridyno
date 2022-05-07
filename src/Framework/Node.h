/**
 * Copyright 2017-2021 Xiaowei He
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "OBase.h"
#include "Field.h"
#include "Platform.h"
#include "DeclarePort.h"
#include "NodePort.h"

#include "Module/TopologyModule.h"
//#include "Module/DeviceContext.h"
#include "Module/CustomModule.h"
#include "Module/TopologyMapping.h"
#include "Module/AnimationPipeline.h"
#include "Module/GraphicsPipeline.h"

namespace dyno
{
	class Action;
	class SceneGraph;

	class Node : public OBase
	{
	public:

		template<class T>
		using SPtr = std::shared_ptr<T>;

		Node(std::string name = "default");
		~Node() override;

		void setName(std::string name);
		std::string getName();

		virtual std::string getNodeType();

		bool isControllable();

		void setControllable(bool con);


		/// Check the state of dynamics
		virtual bool isActive();

		/// Set the state of dynamics
		virtual void setActive(bool active);

		/// Check the visibility of context
		virtual bool isVisible();

		/// Set the visibility of context
		virtual void setVisible(bool visible);

		/// Simulation timestep
		virtual Real getDt();

		void setDt(Real dt);

		// 	Iterator begin();
		// 	Iterator end();

		void setSceneGraph(SceneGraph* scn);
		SceneGraph* getSceneGraph();

		/**
		 * @brief Add a ancestor
		 *
		 * @param Ancestor
		 * @return std::shared_ptr<Node>
		 */
		Node* addAncestor(Node* anc);

		bool hasAncestor(Node* anc);

		std::vector<NodePort*>& getImportNodes() { return mImportNodes; }
		std::vector<NodePort*>& getExportNodes() { return mExportNodes; }

		/**
		 * @brief Return all descendants
		 *
		 * @return std::list<Node*> descendant list
		 */
		std::list<Node*>& getDescendants() {return mDescendants; }

// 		std::shared_ptr<DeviceContext> getContext();
// 		void setContext(std::shared_ptr<DeviceContext> context);
//		virtual void setAsCurrentContext();

		/**
		 * @brief Add a module to m_module_list and other special module lists
		 *
		 * @param module 	module should be created before calling this function
		 * @return true 	return true if successfully added
		 * @return false 	return false if module already exists
		 */
		bool addModule(std::shared_ptr<Module> module);
		bool deleteModule(std::shared_ptr<Module> module);

		/**
		 * @brief Add a speical kind of module
		 *
		 * @tparam TModule 	Module type
		 * @param tModule 	Added module
		 * @return true
		 * @return false
		 */
		template<class TModule>
		bool addModule(std::shared_ptr<TModule> tModule)
		{
			std::shared_ptr<Module> module = std::dynamic_pointer_cast<Module>(tModule);
			return addModule(module);
		}
		template<class TModule>
		bool deleteModule(std::shared_ptr<TModule> tModule)
		{
			std::shared_ptr<Module> module = std::dynamic_pointer_cast<Module>(tModule);
			return deleteModule(module);
		}

		std::list<std::shared_ptr<Module>>& getModuleList() { return m_module_list; }

		bool hasModule(std::string name);

		/**
		 * @brief Get a module by its name
		 *
		 * @param name 	Module name
		 * @return std::shared_ptr<Module>	return nullptr is no module is found, otherwise return the first found module
		 */
		std::shared_ptr<Module> getModule(std::string name);

		/**
		 * @brief Get the Module by the module class name
		 *
		 * @tparam TModule 	Module class name
		 * @return std::shared_ptr<TModule> return nullptr is no module is found, otherwise return the first found module
		 */
		template<class TModule>
		std::shared_ptr<TModule> getModule()
		{
			TModule* tmp = new TModule;
			std::shared_ptr<Module> base;
			std::list<std::shared_ptr<Module>>::iterator iter;
			for (iter = m_module_list.begin(); iter != m_module_list.end(); iter++)
			{
				if ((*iter)->getClassInfo() == tmp->getClassInfo())
				{
					base = *iter;
					break;
				}
			}
			delete tmp;
			return TypeInfo::cast<TModule>(base);
		}

		template<class TModule>
		std::shared_ptr<TModule> getModule(std::string name)
		{
			std::shared_ptr<Module> base = getModule(name);

			return TypeInfo::cast<TModule>(base);
		}

#define NODE_SET_SPECIAL_MODULE_BY_TEMPLATE( CLASSNAME )														\
	template<class TModule>																						\
	std::shared_ptr<TModule> set##CLASSNAME(std::string name) {													\
		if (hasModule(name))																					\
		{																										\
			Log::sendMessage(Log::Error, std::string("Module ") + name + std::string(" already exists!"));		\
			return nullptr;																						\
		}																										\
																												\
		std::shared_ptr<TModule> module = std::make_shared<TModule>();											\
		module->setName(name);																					\
		this->set##CLASSNAME(module);																			\
																												\
		return module;																							\
	}

#define NODE_SET_SPECIAL_MODULE( CLASSNAME, MODULENAME )						\
	virtual void set##CLASSNAME( std::shared_ptr<CLASSNAME> module) {			\
		if(MODULENAME != nullptr)	deleteFromModuleList(MODULENAME);					\
		MODULENAME = module;													\
		addToModuleList(module);														\
	}

		std::shared_ptr<AnimationPipeline>		animationPipeline();
		std::shared_ptr<GraphicsPipeline>		graphicsPipeline();

		template<class TModule>
		std::shared_ptr<TModule> addModule(std::string name)
		{
			if (hasModule(name))
			{
				Log::sendMessage(Log::Error, std::string("Module ") + name + std::string(" already exists!"));
				return nullptr;
			}
			std::shared_ptr<TModule> module = std::make_shared<TModule>();
			module->setName(name);
			this->addModule(module);

			return module;
		}

#define NODE_CREATE_SPECIAL_MODULE(CLASSNAME) \
	template<class TModule>									\
		std::shared_ptr<TModule> add##CLASSNAME(std::string name)	\
		{																	\
			if (hasModule(name))											\
			{																\
				Log::sendMessage(Log::Error, std::string("Module ") + name + std::string(" already exists!"));	\
				return nullptr;																					\
			}																									\
			std::shared_ptr<TModule> module = std::make_shared<TModule>();										\
			module->setName(name);																				\
			this->add##CLASSNAME(module);																		\
																												\
			return module;																						\
		}																										

#define NODE_ADD_SPECIAL_MODULE( CLASSNAME, SEQUENCENAME ) \
	virtual void add##CLASSNAME( std::shared_ptr<CLASSNAME> module) { SEQUENCENAME.push_back(module); addToModuleList(module);} \
	virtual void delete##CLASSNAME( std::shared_ptr<CLASSNAME> module) { SEQUENCENAME.remove(module); deleteFromModuleList(module); } \
	std::list<std::shared_ptr<CLASSNAME>>& get##CLASSNAME##List(){ return SEQUENCENAME;}

			NODE_ADD_SPECIAL_MODULE(TopologyMapping, m_topology_mapping_list)
			NODE_ADD_SPECIAL_MODULE(CustomModule, m_custom_list)

			NODE_CREATE_SPECIAL_MODULE(TopologyMapping)
			NODE_CREATE_SPECIAL_MODULE(CustomModule)

			
		/**
		 * @brief Initialize all states, called before the node is first updated.
		 */
		void initialize();
		
		/**
		 * @brief Called every time interval.
		 */
		void update();

		void reset();

		/**
		 * @brief Depth-first tree traversal
		 *
		 * @param act 	Operation on the node
		 */
		void traverseBottomUp(Action* act);
		template<class Act, class ... Args>
		void traverseBottomUp(Args&& ... args) {
			Act action(std::forward<Args>(args)...);
			doTraverseBottomUp(&action);
		}

		/**
		 * @brief Breadth-first tree traversal
		 *
		 * @param act 	Operation on the node
		 */
		void traverseTopDown(Action* act);
		template<class Act, class ... Args>
		void traverseTopDown(Args&& ... args) {
			Act action(std::forward<Args>(args)...);
			doTraverseTopDown(&action);
		}
		
		bool connect(NodePort* nPort);
		bool disconnect(NodePort* nPort);

		/**
		 * @brief Attach a field to Node
		 *
		 * @param field Field pointer
		 * @param name Field name
		 * @param desc Field description
		 * @param autoDestroy The field will be destroyed by Base if true, otherwise, the field should be explicitly destroyed by its creator.
		 *
		 * @return Return false if the name conflicts with exists fields' names
		 */
		bool attachField(FBase* field, std::string name, std::string desc, bool autoDestroy = true) override;

		std::vector<NodePort*>& getAllNodePorts() { return mImportNodes; }

		uint sizeOfNodePorts() const { return (uint)mImportNodes.size(); }
		uint sizeOfAncestors() const { return (uint)mAncestors.size(); }
		uint sizeofDescendants() const { return (uint)mDescendants.size(); }
	
	protected:
		virtual void doTraverseBottomUp(Action* act);
		virtual void doTraverseTopDown(Action* act);

		bool appendExportNode(NodePort* nodePort);
		bool removeExportNode(NodePort* nodePort);

		virtual void preUpdateStates();
		virtual void updateStates();
		virtual void postUpdateStates();

		virtual void updateTopology();

		virtual void resetStates();

		virtual bool validateInputs();

	private:
		/**
		 * @brief Add a descendant
		 *
		 * @param descendant
		 * @return std::shared_ptr<Node>
		 */
		Node* addDescendant(Node* descent);

		bool hasDescendant(Node* descent);

		void removeDescendant(Node* descent);

		bool addNodePort(NodePort* port);

		bool addToModuleList(std::shared_ptr<Module> module);
		bool deleteFromModuleList(std::shared_ptr<Module> module);

#define NODE_ADD_SPECIAL_MODULE_LIST( CLASSNAME, SEQUENCENAME ) \
	virtual void addTo##CLASSNAME##List( std::shared_ptr<CLASSNAME> module) { SEQUENCENAME.push_back(module); } \
	virtual void deleteFrom##CLASSNAME##List( std::shared_ptr<CLASSNAME> module) { SEQUENCENAME.remove(module); } \

			NODE_ADD_SPECIAL_MODULE_LIST(TopologyMapping, m_topology_mapping_list)
			NODE_ADD_SPECIAL_MODULE_LIST(CustomModule, m_custom_list)


	public:
		DEF_VAR(Real, TimeStep, 0, "Time step size");
		DEF_VAR(Real, ElapsedTime, 0, "Elapsed Time");

		DEF_VAR(Vec3f, Location, 0, "Node location");
		DEF_VAR(Vec3f, Rotation, 0, "Node rotation");
		DEF_VAR(Vec3f, Scale, 0, "Node scale");

		std::string m_node_name;

		/**
		 * @brief Dynamics indicator
		 * true: Dynamics is turn on
		 * false: Dynamics is turned off
		 */
		DEF_VAR(bool, Active, true, "Indicating whether the simulation is on for this node!");


		/**
		 * @brief Visibility
		 * true: the node is visible
		 * false: the node is invisible
		 */
		DEF_VAR(bool, Visible, true, "Indicating whether the node is visible!");

		DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");

	private:
		bool m_controllable = true;

		/**
		 * @brief Time step size
		 *
		 */
		Real m_dt;
		bool m_initalized;

		Real m_mass;

		/**
		 * @brief A module list containing all modules
		 *
		 */
		std::list<std::shared_ptr<Module>> m_module_list;

		//m_topology will be deprecated in the next version
		std::shared_ptr<TopologyModule> m_topology;
		/**
		 * @brief Pointer of a specific module
		 *
		 */
		std::shared_ptr<AnimationPipeline> m_animation_pipeline;
		std::shared_ptr<GraphicsPipeline> m_render_pipeline;

		/**
		 * @brief A module list containg specific modules
		 *
		 */
		std::list<std::shared_ptr<TopologyMapping>> m_topology_mapping_list;
		std::list<std::shared_ptr<CustomModule>> m_custom_list;

		//std::shared_ptr<DeviceContext> m_context;

		std::list<Node*> mAncestors;

		/**
		 * @brief Storing pointers to descendants
		 */
		std::list<Node*> mDescendants;

		std::vector<NodePort*> mImportNodes;

		std::vector<NodePort*> mExportNodes;

		SceneGraph* mSceneGraph = nullptr;

		friend class NodePort;
	};
}
