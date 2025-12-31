#include "MaterialManager.h"
#include "DirectedAcyclicGraph.h"

namespace dyno
{
	std::vector<MaterialManagerObserver*> dyno::MaterialManager::mMaterialListObservers;
	std::vector<std::shared_ptr<MaterialLoaderModule>> MaterialManager::mMaterialLoaderModules;


	IMPLEMENT_CLASS(BreakMaterial)
	void BreakMaterial::initial() 
	{
		auto fieldChange = std::make_shared<FCallBackFunc>(std::bind(&BreakMaterial::onFieldChanged, this));
		this->inMaterial()->attach(fieldChange);
	}

	BreakMaterial::BreakMaterial(std::shared_ptr<BreakMaterial> other) 
	{
		auto matSource = other->inMaterial()->getSource();
		if (matSource) 
		{
			matSource->connect(this->inMaterial());
		}
		this->initial();
	}

	void BreakMaterial::onFieldChanged()
	{
		if (this->inMaterial()->isEmpty())
			return;

		auto inMat = this->inMaterial()->getDataPtr();

		this->outColor()->setValue(inMat->baseColor);
		this->outRoughness()->setValue(inMat->roughness);
		this->outMetallic()->setValue(inMat->metallic);
		this->outAlpha()->setValue(inMat->alpha);
		this->outBumpScale()->setValue(inMat->bumpScale);
		this->outEmissiveIntensity()->setValue(inMat->emissiveIntensity);

		if (inMat->texColor.size()) 
		{
			if (this->outTexColor()->isEmpty())
				this->outTexColor()->allocate();
			this->outTexColor()->assign(inMat->texColor);
		}
		if (inMat->texBump.size())
		{
			if (this->outTexBump()->isEmpty())
				this->outTexBump()->allocate();
			this->outTexBump()->assign(inMat->texBump);
		}
		if (inMat->texORM.size())
		{
			if (this->outTexORM()->isEmpty())
				this->outTexORM()->allocate();
			this->outTexORM()->assign(inMat->texORM);
		}
		if (inMat->texAlpha.size())
		{
			if (this->outTexAlpha()->isEmpty())
				this->outTexAlpha()->allocate();
			this->outTexAlpha()->assign(inMat->texAlpha);
		}
		if (inMat->texEmissive.size())
		{
			if (this->outTexEmissive()->isEmpty())
				this->outTexEmissive()->allocate();
			this->outTexEmissive()->assign(inMat->texEmissive);
		}
	}

	std::shared_ptr<MaterialManagedModule> BreakMaterial::clone() const
	{
		std::shared_ptr<MaterialManagedModule> materialPtr = MaterialManager::getMaterialManagedModule(this->getName());
		if (!materialPtr)
		{
			printf("Error: newGrayscaleCorrect::clone() Failed!! \n");
			return nullptr;
		}

		std::shared_ptr<BreakMaterial> breakMat = std::dynamic_pointer_cast<BreakMaterial>(materialPtr);
		if (!breakMat)
		{
			printf("Error: newGrayscaleCorrect::clone() Cast Failed!!  \n");
			return nullptr;
		}

		std::shared_ptr<BreakMaterial> newBreakMat(new BreakMaterial(breakMat));
		newBreakMat->setName(MaterialManager::generateUniqueMaterialName(breakMat->getName()));
		return newBreakMat;
	}

	IMPLEMENT_CLASS(MaterialLoaderModule);

	void CustomMaterial::updateVar2Out() 
	{
		auto varBaseColor = this->varBaseColor()->getValue();
		auto baseColor = this->inBaseColor()->isEmpty() ? varBaseColor : this->inBaseColor()->getValue();
		this->outMaterial()->getDataPtr()->baseColor = baseColor;
		this->outMaterial()->getDataPtr()->metallic = this->inMetallic()->isEmpty() ? this->varMetallic()->getValue():this->inMetallic()->getValue();
		this->outMaterial()->getDataPtr()->roughness = this->inRoughness()->isEmpty() ? this->varRoughness()->getValue() : this->inRoughness()->getValue();
		this->outMaterial()->getDataPtr()->alpha = this->inAlpha()->isEmpty() ? this->varAlpha()->getValue() : this->inAlpha()->getValue();
		this->outMaterial()->getDataPtr()->bumpScale = this->inBumpScale()->isEmpty() ? this->varBumpScale()->getValue() : this->inBumpScale()->getValue();
		this->outMaterial()->getDataPtr()->emissiveIntensity = this->inEmissiveIntensity()->isEmpty() ? this->varEmissiveIntensity()->getValue() : this->inEmissiveIntensity()->getValue();
	}

	CustomMaterial::CustomMaterial() :MaterialManagedModule()
	{
		this->outMaterial()->setDataPtr(std::make_shared<Material>());
		initialVar();
		updateVar2Out();
	}

	CustomMaterial::CustomMaterial(const std::string& name) :MaterialManagedModule()
	{
		this->outMaterial()->setDataPtr(std::make_shared<Material>());
		initialVar();
		updateVar2Out();

		this->mName = name;
	}

	void CustomMaterial::updateImpl()
	{
		if (!this->inTexColor()->isEmpty()) 
		{
			this->outMaterial()->getDataPtr()->texColor.assign(this->inTexColor()->getData());
		}
		if (!this->inTexAlpha()->isEmpty())
		{
			this->outMaterial()->getDataPtr()->texAlpha.assign(this->inTexAlpha()->getData());
		}
		if (!this->inTexBump()->isEmpty())
		{
			this->outMaterial()->getDataPtr()->texBump.assign(this->inTexBump()->getData());
		}
		if (!this->inTexORM()->isEmpty())
		{
			this->outMaterial()->getDataPtr()->texORM.assign(this->inTexORM()->getData());
		}
		if (!this->inTexEmissiveColor()->isEmpty())
		{
			this->outMaterial()->getDataPtr()->texEmissive.assign(this->inTexEmissiveColor()->getData());
		}
		updateVar2Out();
		this->updateAssigner();
	}

	void CustomMaterial::addAssigner(std::shared_ptr<Module> assigner)
	{
		mAssigner.insert(assigner);
	}
	          
	void CustomMaterial::removeAssigner(std::shared_ptr<Module> assigner)
	{
		mAssigner.erase(assigner);
	}

	void CustomMaterial::updateAssigner()
	{
		for (auto& it : mAssigner)
		{
			if (it)
			{
				it->varForceUpdate()->setValue(true);
				it->update();
				it->varForceUpdate()->setValue(false);
			}
		}
	}

	std::shared_ptr<MaterialPipeline> CustomMaterial::materialPipeline()
	{
		if (mMaterialPipeline == nullptr)
		{
			mMaterialPipeline = std::make_shared<MaterialPipeline>(this);
		}
		return mMaterialPipeline;
	}

	std::vector<std::shared_ptr<Module>> CustomMaterial::piplineModules()
	{
		std::vector<std::shared_ptr<Module>> outModules;
		if (mMaterialPipeline)
		{
			auto modules = mMaterialPipeline->allModules();
			for (auto it : modules)
				outModules.push_back(it.second);
		}
		return outModules;
	}

	std::string CustomMaterial::pushMaterialManagedModule(std::shared_ptr<MaterialManagedModule> managedModule)
	{
		return MaterialManager::pushMaterialManagedModule(managedModule);
	}

	CustomMaterial::CustomMaterial(const std::shared_ptr<MaterialLoaderModule>& MaterialLoaderPtr, std::shared_ptr<BreakMaterial>& BreakMaterialModule, std::string Name) :MaterialManagedModule()
	{
		this->outMaterial()->setDataPtr(std::make_shared<Material>());

		BreakMaterialModule = std::make_shared<BreakMaterial>();
		MaterialLoaderPtr->outMaterial()->connect(BreakMaterialModule->inMaterial());
		BreakMaterialModule->outAlpha()->connect(this->inAlpha());
		BreakMaterialModule->outBumpScale()->connect(this->inBumpScale());
		BreakMaterialModule->outColor()->connect(this->inBaseColor());
		BreakMaterialModule->outEmissiveIntensity()->connect(this->inEmissiveIntensity());
		BreakMaterialModule->outMetallic()->connect(this->inMetallic());
		BreakMaterialModule->outRoughness()->connect(this->inRoughness());
		BreakMaterialModule->outTexAlpha()->connect(this->inTexAlpha());
		BreakMaterialModule->outTexBump()->connect(this->inTexBump());
		BreakMaterialModule->outTexColor()->connect(this->inTexColor());
		BreakMaterialModule->outTexEmissive()->connect(this->inTexEmissiveColor());
		BreakMaterialModule->outTexORM()->connect(this->inTexORM());

		this->materialPipeline()->pushModule(BreakMaterialModule);
		this->materialPipeline()->pushModule(MaterialLoaderPtr);

		MaterialManager::pushMaterialManagedModule(BreakMaterialModule);
		MaterialManager::pushMaterialManagedModule(MaterialLoaderPtr);

		initialVar();
		updateVar2Out();

		this->mName = Name;
	}

	IMPLEMENT_CLASS(CustomMaterial);

	std::map<std::string, std::shared_ptr<CustomMaterial>>& MaterialManager::materials() {
		static std::map<std::string, std::shared_ptr<CustomMaterial>> s_materials;
		return s_materials;
	}

	std::map<std::string, std::shared_ptr<MaterialManagedModule>>& MaterialManager::materialManagedModules() {
		static std::map<std::string, std::shared_ptr<MaterialManagedModule>> s_matModules;
		return s_matModules;
	}

	std::map<std::string, int>& MaterialManager::nameCount()
	{
		static std::map<std::string, int> s_nameCount;
		return s_nameCount;
	}

	void MaterialManager::addMaterialListObserver(MaterialManagerObserver* observer)
	{
		mMaterialListObservers.push_back(observer);
	}


	void MaterialManager::removeMaterialListObserver(MaterialManagerObserver* observer)
	{
		mMaterialListObservers.erase(std::remove(mMaterialListObservers.begin(), mMaterialListObservers.end(), observer), mMaterialListObservers.end());
	}

	void MaterialManager::callMaterialManagerObservers(std::shared_ptr<MaterialManagedModule> mat) {
		for (auto obs : mMaterialListObservers) {
			obs->onMaterialListChanged(mat);
		}
	}

	void MaterialManager::printAllMaterials() {
		std::cout << "Materials list : " << std::endl;
		for (const auto& pair : materials()) {
			std::cout << " - " << pair.first << std::endl;
		}
	}

	void MaterialManager::printAllManagedModules()
	{
		std::cout << "ManagedModules list : " << std::endl;
		for (const auto& pair :materialManagedModules()) {
			std::cout << " - " << pair.first << std::endl;
		}
	}

	std::shared_ptr<CustomMaterial> MaterialManager::getMaterial(const std::string& name) {
		auto it = materials().find(name);
		if (it != materials().end())
			return it->second;
		return nullptr;
	}

	std::shared_ptr<MaterialManagedModule> MaterialManager::getMaterialManagedModule(const std::string& name)
	{
		auto it = materialManagedModules().find(name);
		if (it != materialManagedModules().end())
			return it->second;
		return nullptr;
	}


	std::shared_ptr<MaterialManagedModule> MaterialManager::copyMaterialManagedModule(std::shared_ptr<MaterialManagedModule> matModule)
	{
		auto ptr = matModule->clone();
		if (ptr)
			MaterialManager::pushMaterialManagedModule(ptr);
		else
			printf("MaterialManager::copyMaterialManagedModule:: Failed!!\n");

		return ptr;
	}

	std::shared_ptr<CustomMaterial> MaterialManager::createCustomMaterial(std::string name)
	{
		std::string uniqueName = MaterialManager::generateUniqueMaterialName(name);
		std::shared_ptr<CustomMaterial> customMat = std::make_shared<CustomMaterial>(uniqueName);

		addMaterial(customMat);
		customMat->materialPipeline()->pushModule(customMat);

		return customMat;
	}

	std::shared_ptr<CustomMaterial> MaterialManager::createCustomMaterial(const std::shared_ptr<MaterialLoaderModule>& MaterialLoaderPtr, std::shared_ptr<BreakMaterial>& BreakMaterialModule, std::string Name)
	{
		std::string uniqueName = MaterialManager::generateUniqueMaterialName(Name);
		std::shared_ptr<CustomMaterial> customMat = std::make_shared<CustomMaterial>(MaterialLoaderPtr, BreakMaterialModule, uniqueName);

		addMaterial(customMat);
		customMat->materialPipeline()->pushModule(customMat);
		MaterialManager::pushMaterialManagedModule(customMat);

		return customMat;
	}

	std::string MaterialManager::generateUniqueMaterialName(const std::string& baseName)
	{
		auto& nameCount = MaterialManager::nameCount();

		std::string uniqueName = baseName;
		if (nameCount.find(baseName) != nameCount.end()) {

			int count = nameCount[baseName];
			uniqueName = baseName + "_" + std::to_string(count);
			nameCount[baseName] = count + 1;
			nameCount[uniqueName] = 1;
		}
		else {
			nameCount[baseName] = 1;
		}

		return uniqueName;
	}

	bool MaterialManager::removeMaterialManagedModule(const std::string& name) 
	{
		auto it = materialManagedModules().find(name);
		bool success = false;
		if (it != materialManagedModules().end())
		{
			auto tryMat = std::dynamic_pointer_cast<CustomMaterial>(it->second);
			if(tryMat)
			{
				if (materials().find(name) != materials().end())
				{
					success = materials().erase(name) > 0 ;
				}
				printAllMaterials();
			}
			success = materialManagedModules().erase(name) > 0 || success;
			printAllManagedModules();
			callMaterialManagerObservers();
		}

		return success;
	}

	std::string MaterialManager::addMaterial(std::shared_ptr<CustomMaterial> CustomMaterial)
	{
		if (containsMaterial(CustomMaterial))
			return std::string("MaterialManager::addMaterial: Failed!");

		std::string baseName = CustomMaterial->getName();
		auto& mats = materials();
		std::string uniqueName = generateUniqueMaterialName(baseName);
		CustomMaterial->setName(uniqueName);
		mats[uniqueName] = CustomMaterial;
		pushMaterialManagedModule(CustomMaterial,false);
		printAllMaterials();
		callMaterialManagerObservers();
		return uniqueName;
	}

	void MaterialManager::rename(std::shared_ptr<MaterialManagedModule> ptr, const std::string& name)
	{
		auto& matModules = materialManagedModules();

		std::string lastName = ptr->getName();
		auto it = matModules.find(lastName);

		ptr->setName(name);

		if (it != matModules.end())
		{
			std::string uniqueName = generateUniqueMaterialName(name);

			matModules[uniqueName] = ptr;
			pushMaterialManagedModule(ptr, false);
			matModules.erase(it);

			auto tryMat = std::dynamic_pointer_cast<CustomMaterial>(ptr);
			if (tryMat)
			{
				auto& mats = materials();
				mats[uniqueName] = tryMat;

				auto mat = mats.find(lastName);
				if(mat!= mats.end())
					mats.erase(mat);
			}
		}
		else
		{
			pushMaterialManagedModule(ptr);
			auto tryMat = std::dynamic_pointer_cast<CustomMaterial>(ptr);
			if (tryMat)
			{
				addMaterial(tryMat);
			}
			std::cout << "Warning: The material is not in the material library. Add the material:" << ptr->getName() << "\n";
		}
		MaterialManager::printAllMaterials();
	}

	void MaterialPipeline::pushModule(std::shared_ptr<Module> m)
	{
		ObjectId id = m->objectId();
		if (allModules().find(id) != allModules().end())
			return;

		auto managedModule = std::dynamic_pointer_cast<MaterialManagedModule>(m);

		if (managedModule)
		{
			mCustomMaterial->pushMaterialManagedModule(managedModule);
		}

		setModuleUpdated(true);
		allModules()[id] = m;


	}

	void MaterialPipeline::popModule(std::shared_ptr<Module> m)
	{
		ObjectId id = m->objectId();
		if (allModules().find(id) != allModules().end())
			return;

		setModuleUpdated(true);
		allModules()[id] = m;
	}

	void MaterialPipeline::reconstructPipeline()
	{
		
		ObjectId baseId = Object::baseId();
		auto& moduleList = ModuleList();
		moduleList.clear();

		auto& materialAllModules = allModules();

		std::queue<Module*> moduleQueue;
		std::set<ObjectId> moduleSet;

		DirectedAcyclicGraph graph;

		auto retrieveModules = [&](ObjectId id, std::vector<FBase*>& fields) {
			for (auto f : fields) {
				auto& sinks = f->getSinks();
				for (auto sink : sinks)
				{
					Module* module = dynamic_cast<Module*>(sink->parent());
					if (module != nullptr)
					{
						ObjectId oId = module->objectId();
						graph.addEdge(id, oId);

						if (moduleSet.find(oId) == moduleSet.end() && materialAllModules.count(oId) > 0)
						{
							moduleSet.insert(oId);
							moduleQueue.push(module);
						}
					}
				}
			}
		};

		auto flushQueue = [&]()
		{
			while (!moduleQueue.empty())
			{
				Module* m = moduleQueue.front();

				auto& outFields = m->getOutputFields();
				retrieveModules(m->objectId(), outFields);

				moduleQueue.pop();
			}
		};

		flushQueue();

		for (auto m : materialAllModules) {
			ObjectId oId = m.second->objectId();

			//Create connection between fields
			if (moduleSet.find(oId) == moduleSet.end())
			{
				moduleSet.insert(oId);
				moduleQueue.push(m.second.get());

				flushQueue();
			}

			//Create connection between modules
			auto exports = m.second->getExportModules();
			for (auto exp : exports)
			{
				auto eId = exp->getParent()->objectId();
				if (materialAllModules.count(eId) > 0)
				{
					graph.addEdge(oId, eId);
				}
			}
		}

		auto& ids = graph.topologicalSort();

		for (auto id : ids)
		{
			if (materialAllModules.count(id) > 0)
			{
				moduleList.push_back(materialAllModules[id]);
			}
		}

		moduleSet.clear();

		setModuleUpdated(true);

	}

	void MaterialPipeline::updateMaterialPipline()
	{
		this->updateExecutionQueue();
		updateImpl();

	}

	void MaterialManager::traverseForward(MaterialAction& matAction)
	{
		for (auto it : materials())
		{
			matAction.process(it.second);		
		}
	}

	std::string MaterialManager::pushMaterialManagedModule(std::shared_ptr<MaterialManagedModule> MatModule, bool checkName)
	{
		if (containsModule(MatModule))
			return std::string("MaterialManager::pushMaterialManagedModule : Failed!");
		std::string baseName = MatModule->getName();
		auto& matModules = materialManagedModules();

		if (checkName) 
			baseName = generateUniqueMaterialName(baseName);

		MatModule->setName(baseName);
		matModules[baseName] = MatModule;
		printAllManagedModules();
		callMaterialManagerObservers();
		return baseName;
	}

	void MaterialManager::onKeyboardEvent(PKeyboardEvent event)
	{

		class MatKeyboardEventAct : public MaterialAction
		{
		public:
			MatKeyboardEventAct(PKeyboardEvent event) { mKeyboardEvent = event; }
			~MatKeyboardEventAct() override {}


			void process(std::shared_ptr<CustomMaterial> customMaterial) override
			{

				for (auto iter : customMaterial->materialPipeline()->activeModules())
				{
					auto m = dynamic_cast<KeyboardInputModule*>(iter.get());
					if (m)
					{
						m->enqueueEvent(mKeyboardEvent);
						m->update();
					}
				}
			}

			PKeyboardEvent mKeyboardEvent;
		};

		MatKeyboardEventAct eventAct(event);
		MaterialManager::traverseForward(eventAct);
	}

	bool MaterialManager::containsMaterial(const std::shared_ptr<CustomMaterial>& mat)
	{
		auto& mats = materials();
		for (const auto& pair : mats)
		{
			if (pair.second == mat)
			{
				return true;
			}
		}
		return false;
	}

	std::shared_ptr<MaterialLoaderModule> MaterialManager::createMaterialLoaderModule(std::shared_ptr<Material> mat, std::string Name) {
		auto loader = std::make_shared<MaterialLoaderModule>(mat,Name);
		mMaterialLoaderModules.push_back(loader);
		pushMaterialManagedModule(loader);
		return loader;
	}

	std::shared_ptr<Material> MaterialManager::getMaterialPtr(std::string Name)
	{
		for (auto it : mMaterialLoaderModules)
		{
			if (it->getName() == Name)
				return it->outMaterial()->getDataPtr();
		}
		return NULL;
	}

	bool MaterialManager::containsModule(const std::shared_ptr<MaterialManagedModule>& matModule)
	{
		auto& matModules = materialManagedModules();
		for (const auto& pair : matModules)
		{
			if (pair.second == matModule)
			{
				return true;
			}
		}
		return false;
	}
}