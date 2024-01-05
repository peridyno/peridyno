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
#include "Module/CustomModule.h"
#include "Module/TopologyMapping.h"
#include "Module/AnimationPipeline.h"
#include "Module/GraphicsPipeline.h"

namespace dyno
{
	class Action;
	class SceneGraph;

	struct NBoundingBox
	{
		NBoundingBox() {
			lower = Vec3f(-1);
			upper = Vec3f(1);
		}

		NBoundingBox(Vec3f lo, Vec3f hi) {
			lower = lo;
			upper = hi;
		}

		Vec3f lower = Vec3f(-1);
		Vec3f upper = Vec3f(1);

		NBoundingBox& join(const NBoundingBox box) {
			lower = lower.minimum(box.lower);
			upper = upper.maximum(box.upper);

			return *this;
		}

		NBoundingBox& intersect(const NBoundingBox box) {
			lower = lower.maximum(box.lower);
			upper = upper.minimum(box.upper);

			return *this;
		}

		float maxLength() {
			return std::max(upper[0] - lower[0], std::max(upper[1] - lower[1], upper[2] - lower[2]));
		}
	};

	class Node : public OBase
	{
	public:

		template<class T>
		using SPtr = std::shared_ptr<T>;

		Node();
		~Node() override;

		void setName(std::string name);
		std::string getName() override;

		virtual std::string getNodeType();

		bool isAutoSync();

		/**
		 * @brief Whether the node can be automatically synchronized when its ancestor is updated
		 *
		 * @param if true, the node can be synchronized automatically, otherwise not
		 */
		void setAutoSync(bool con);

		bool canExported();

		/**
		 * @brief To allow exporting the node
		 *
		 * @param if true, the node can be exported, otherwise not
		 */
		void allowExported(bool ex);

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

		void setSceneGraph(SceneGraph* scn);
		SceneGraph* getSceneGraph();

		std::vector<NodePort*>& getImportNodes() { return mImportNodes; }
		std::vector<NodePort*>& getExportNodes() { return mExportNodes; }

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
	
		/**
		 * @brief Initialize all states, called before the node is first updated.
		 */
		void initialize();
		
		/**
		 * @brief Called every time interval.
		 */
		void update();

		void updateGraphicsContext();

		void reset();

		virtual NBoundingBox boundingBox();
// 		/**
// 		 * @brief Depth-first tree traversal
// 		 *
// 		 * @param act 	Operation on the node
// 		 */
// 		void traverseBottomUp(Action* act);
// 		template<class Act, class ... Args>
// 		void traverseBottomUp(Args&& ... args) {
// 			Act action(std::forward<Args>(args)...);
// 			doTraverseBottomUp(&action);
// 		}
// 
// 		/**
// 		 * @brief Breadth-first tree traversal
// 		 *
// 		 * @param act 	Operation on the node
// 		 */
// 		void traverseTopDown(Action* act);
// 		template<class Act, class ... Args>
// 		void traverseTopDown(Args&& ... args) {
// 			Act action(std::forward<Args>(args)...);
// 			doTraverseTopDown(&action);
// 		}
		
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

		uint sizeOfNodePorts() { return (uint)mImportNodes.size(); }

		uint sizeOfImportNodes() const;
		uint sizeOfExportNodes() const { return (uint)mExportNodes.size(); }

	protected:
// 		virtual void doTraverseBottomUp(Action* act);
// 		virtual void doTraverseTopDown(Action* act);

		bool appendExportNode(NodePort* nodePort);
		bool removeExportNode(NodePort* nodePort);

		virtual void preUpdateStates();
		virtual void updateStates();
		virtual void postUpdateStates();

		virtual void updateTopology();

		virtual void resetStates();

		virtual bool validateInputs();

		/**
		 * @brief notify all state and output fields are updated
		 */
		void tick();

	private:
		bool addNodePort(NodePort* port);

		bool addToModuleList(std::shared_ptr<Module> module);
		bool deleteFromModuleList(std::shared_ptr<Module> module);

	public:
		std::string m_node_name;

		DEF_VAR_STATE(Real, ElapsedTime, 0, "Elapsed Time");
		DEF_VAR_STATE(Real, TimeStep, Real(0.033), "Time step size");

		DEF_VAR_STATE(uint, FrameNumber, 0, "Frame number");

	private:
		/**
		 * @brief A parameter to control whether the node can be updated automatically by its ancestor
		 */
		bool mAutoSync = false;

		bool mPhysicsEnabled = true;

		bool mRenderingEnabled = true;

		bool mExported = true;

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

		/**
		 * @brief Pointer of the pipeline
		 *
		 */
		std::shared_ptr<AnimationPipeline> m_animation_pipeline;
		std::shared_ptr<GraphicsPipeline> m_render_pipeline;

// 		//std::shared_ptr<DeviceContext> m_context;
// 
		std::vector<NodePort*> mImportNodes;

		std::vector<NodePort*> mExportNodes;

		SceneGraph* mSceneGraph = nullptr;

		friend class NodePort;
	};
}
