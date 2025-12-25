#include "MaterialManager.h"
#include "DirectedAcyclicGraph.h"


namespace dyno
{
	std::vector<MaterialManagerObserver*> dyno::MaterialManager::mMaterialListObservers;


	std::shared_ptr<MaterialManagedModule> Material::clone() const
	{
		std::shared_ptr<MaterialManagedModule> materialPtr = MaterialManager::getMaterialManagedModule(this->getName());
		if (!materialPtr)
		{
			printf("Error: Material::clone() Failed!! \n");
			return nullptr;
		}

		std::shared_ptr<Material> material = std::dynamic_pointer_cast<Material>(materialPtr);
		if (!material)
		{
			printf("Error: Material::clone() Cast Failed!!  \n");
			return nullptr;
		}

		std::shared_ptr<Material> newMaterial (new Material(material));

		return newMaterial;
	}

	Material::Material(const std::string& name)
		:MaterialManagedModule()
	{
		initial();
		this->mName = name;
	}

	Material::Material(std::shared_ptr<Material> other) 
		:MaterialManagedModule()
	{
		initial();
		this->mName = other->getName();
		this->outBaseColor()->setValue(other->outBaseColor()->getValue());
		this->outMetallic()->setValue(other->outMetallic()->getValue());
		this->outRoughness()->setValue(other->outRoughness()->getValue());
		this->outAlpha()->setValue(other->outAlpha()->getValue());
		this->outBumpScale()->setValue(other->outBumpScale()->getValue());
		this->outEmissiveItensity()->setValue(other->outEmissiveItensity()->getValue());

		this->outTexColor()->getDataPtr()->assign(other->outTexColor()->getData());
		this->outTexBump()->getDataPtr()->assign(other->outTexBump()->getData());
		this->outTexORM()->getDataPtr()->assign(other->outTexORM()->getData());
		this->outTexAlpha()->getDataPtr()->assign(other->outTexAlpha()->getData());
		this->outTexEmissive()->getDataPtr()->assign(other->outTexEmissive()->getData());

	}


	Material::~Material() {
		this->outTexColor()->getDataPtr()->clear();
		this->outTexBump()->getDataPtr()->clear();
		this->outTexORM()->getDataPtr()->clear();
		this->outTexAlpha()->getDataPtr()->clear();
		this->outTexEmissive()->getDataPtr()->clear();
	}
	void Material::initial()
	{
		{
			updateVar2Out();

			this->outTexAlpha()->allocate();
			this->outTexBump()->allocate();
			this->outTexColor()->allocate();
			this->outTexORM()->allocate();
			this->outTexEmissive()->allocate();

		}
	}
	void Material::updateVar2Out()
	{	
		auto baseColor = this->varBaseColor()->getValue();
		this->outBaseColor()->setValue(Vec3f(baseColor.r, baseColor.g, baseColor.b));
		this->outMetallic()->setValue(this->varMetallic()->getValue());
		this->outRoughness()->setValue(this->varRoughness()->getValue());
		this->outAlpha()->setValue(this->varAlpha()->getValue());
		this->outBumpScale()->setValue(this->varBumpScale()->getValue());		
		this->outEmissiveItensity()->setValue(this->varEmissiveIntensity()->getValue());
	}

	void Material::updateImpl()
	{
		updateVar2Out();
		updateAssigner();
	}

	IMPLEMENT_CLASS(Material);

	void CustomMaterial::initial()
	{
		Material::initial();
		this->inBaseColor()->tagOptional(true);
		this->inMetallic()->tagOptional(true);
		this->inRoughness()->tagOptional(true);
		this->inAlpha()->tagOptional(true);
		this->inBumpScale()->tagOptional(true);
		this->inTexColor()->tagOptional(true);
		this->inTexBump()->tagOptional(true);
		this->inTexORM()->tagOptional(true);
		this->inTexEmissiveColor()->tagOptional(true);
		this->inTexAlpha()->tagOptional(true);
		this->inEmissiveIntensity()->tagOptional(true);
	}

	void CustomMaterial::updateVar2Out() 
	{
		auto varBaseColor = this->varBaseColor()->getValue();
		auto baseColor = this->inBaseColor()->isEmpty() ? Vec3f(varBaseColor.r, varBaseColor.g, varBaseColor.b) : this->inBaseColor()->getValue();
		this->outBaseColor()->setValue(baseColor);
		this->outMetallic()->setValue(this->inMetallic()->isEmpty() ? this->varMetallic()->getValue():this->inMetallic()->getValue());
		this->outRoughness()->setValue(this->inRoughness()->isEmpty() ? this->varRoughness()->getValue() : this->inRoughness()->getValue());
		this->outAlpha()->setValue(this->inAlpha()->isEmpty() ? this->varAlpha()->getValue() : this->inAlpha()->getValue());
		this->outBumpScale()->setValue(this->inBumpScale()->isEmpty() ? this->varBumpScale()->getValue() : this->inBumpScale()->getValue());
		this->outEmissiveItensity()->setValue(this->inEmissiveIntensity()->isEmpty() ? this->varEmissiveIntensity()->getValue() : this->inEmissiveIntensity()->getValue());
	}

	void CustomMaterial::updateImpl()
	{
		if (!this->inTexColor()->isEmpty()) 
		{
			this->outTexColor()->assign(this->inTexColor()->getData());
		}
		if (!this->inTexAlpha()->isEmpty())
		{
			this->outTexAlpha()->assign(this->inTexAlpha()->getData());
		}
		if (!this->inTexBump()->isEmpty())
		{
			this->outTexBump()->assign(this->inTexBump()->getData());
		}
		if (!this->inTexORM()->isEmpty())
		{
			this->outTexORM()->assign(this->inTexORM()->getData());
		}
		if (!this->inTexEmissiveColor()->isEmpty())
		{
			this->outTexEmissive()->assign(this->inTexEmissiveColor()->getData());
		}
		updateVar2Out();
		this->updateAssigner();
	}

	void Material::addAssigner(std::shared_ptr<Module> assigner)
	{
		mAssigner.insert(assigner);
	}
	          
	void Material::removeAssigner(std::shared_ptr<Module> assigner)
	{
		mAssigner.erase(assigner);
	}

	void Material::updateAssigner()
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
			auto mat = MaterialManager::getMaterial(this->getName());
			if (!mat)
				return mMaterialPipeline;
			auto customMaterialPtr = std::dynamic_pointer_cast<CustomMaterial>(mat);
			if (!customMaterialPtr)
				return mMaterialPipeline;
			mMaterialPipeline = std::make_shared<MaterialPipeline>(customMaterialPtr);
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

	CustomMaterial::CustomMaterial(std::shared_ptr<Material> sourceMaterial, const std::string& name)
		:Material()
	{
		this->mName = name;

		if (sourceMaterial) 
		{
			sourceMaterial->outAlpha()->connect(this->inAlpha());
			sourceMaterial->outBaseColor()->connect(this->inBaseColor());
			sourceMaterial->outMetallic()->connect(this->inMetallic());
			sourceMaterial->outRoughness()->connect(this->inRoughness());
			sourceMaterial->outBumpScale()->connect(this->inBumpScale());
			sourceMaterial->outEmissiveItensity()->connect(this->inEmissiveIntensity());

			sourceMaterial->outTexAlpha()->connect(this->inTexAlpha());
			sourceMaterial->outTexBump()->connect(this->inTexBump());
			sourceMaterial->outTexORM()->connect(this->inTexORM());
			sourceMaterial->outTexColor()->connect(this->inTexColor());
			sourceMaterial->outTexEmissive()->connect(this->inTexEmissiveColor());

			this->outBaseColor()->setValue(sourceMaterial->outBaseColor()->getValue());
			this->outMetallic()->setValue(sourceMaterial->outMetallic()->getValue());
			this->outRoughness()->setValue(sourceMaterial->outRoughness()->getValue());
			this->outAlpha()->setValue(sourceMaterial->outAlpha()->getValue());
			this->outBumpScale()->setValue(sourceMaterial->outBumpScale()->getValue());
			this->outEmissiveItensity()->setValue(sourceMaterial->outEmissiveItensity()->getValue());

			this->varBaseColor()->setValue(Color(sourceMaterial->outBaseColor()->getValue().x, sourceMaterial->outBaseColor()->getValue().y, sourceMaterial->outBaseColor()->getValue().z));
			this->varMetallic()->setValue(sourceMaterial->outMetallic()->getValue());
			this->varRoughness()->setValue(sourceMaterial->outRoughness()->getValue());
			this->varAlpha()->setValue(sourceMaterial->outAlpha()->getValue());
			this->varBumpScale()->setValue(sourceMaterial->outBumpScale()->getValue());
			this->varEmissiveIntensity()->setValue(sourceMaterial->outEmissiveItensity()->getValue());

			this->outTexColor()->getDataPtr()->assign(sourceMaterial->outTexColor()->getData());
			this->outTexBump()->getDataPtr()->assign(sourceMaterial->outTexBump()->getData());
			this->outTexORM()->getDataPtr()->assign(sourceMaterial->outTexORM()->getData());
			this->outTexEmissive()->getDataPtr()->assign(sourceMaterial->outTexEmissive()->getData());
			this->outTexAlpha()->getDataPtr()->assign(sourceMaterial->outTexAlpha()->getData());
		}	
	}
	IMPLEMENT_CLASS(CustomMaterial);

	std::map<std::string, std::shared_ptr<Material>>& MaterialManager::materials() {
		static std::map<std::string, std::shared_ptr<Material>> s_materials;
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

	void MaterialManager::callMaterialManagerObservers(std::shared_ptr<Material> mat) {
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

	std::shared_ptr<Material> MaterialManager::getMaterial(const std::string& name) {
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


	std::shared_ptr<Material> MaterialManager::NewMaterial()
	{
		std::shared_ptr<Material> mat = std::shared_ptr<Material>(new Material(std::string("Material")));
		addMaterial(mat);
		return mat;
	}

	std::shared_ptr<Material> MaterialManager::NewMaterial(std::string name)
	{
		std::shared_ptr<Material> mat = std::shared_ptr<Material>(new Material(name));
		addMaterial(mat);
		return mat;
	}
	std::shared_ptr<Material> MaterialManager::NewMaterial(const Material& other)
	{
		std::shared_ptr<Material> mat = std::shared_ptr<Material>(new Material(other));
		addMaterial(mat);
		return mat;
	}

	std::shared_ptr<CustomMaterial> MaterialManager::createCustomMaterial(std::shared_ptr<Material> sourceMaterial, std::string name)
	{
		std::string tempName;
		if (sourceMaterial)
			tempName = sourceMaterial->getName() + "_Custom";
		else
			tempName = name + "_Custom";
		std::string uniqueName = MaterialManager::generateUniqueMaterialName(tempName);
		std::shared_ptr<CustomMaterial> customMat;
		if (sourceMaterial)
			customMat = std::shared_ptr<CustomMaterial>(new CustomMaterial(sourceMaterial, uniqueName));
		else
			customMat = std::shared_ptr<CustomMaterial>(new CustomMaterial(uniqueName));

		addMaterial(customMat);

		customMat->materialPipeline()->pushModule(customMat);
		if(sourceMaterial)
			customMat->materialPipeline()->pushModule(sourceMaterial);

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
			auto tryMat = std::dynamic_pointer_cast<Material>(it->second);
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

	std::string MaterialManager::addMaterial(std::shared_ptr<Material> material)
	{
		if (containsMaterial(material))
			return std::string("MaterialManager::addMaterial: Failed!");

		std::string baseName = material->getName();
		auto& mats = materials();
		std::string uniqueName = generateUniqueMaterialName(baseName);
		material->setName(uniqueName);
		mats[uniqueName] = material;
		pushMaterialManagedModule(material,false);
		printAllMaterials();
		callMaterialManagerObservers();
		return uniqueName;
	}


	void MaterialManager::rename(std::shared_ptr<Material> ptr, const std::string& name)
	{
		auto& s_materials = MaterialManager::materials();
		auto it = s_materials.find(ptr->getName());

		ptr->setName(name);

		if (it != s_materials.end())
		{
			addMaterial(ptr);
			s_materials.erase(it);
		}
		else
		{
			addMaterial(ptr);
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
			auto customMaterialPtr = std::dynamic_pointer_cast<CustomMaterial>(it.second);
			if (customMaterialPtr)
			{
				matAction.process(customMaterialPtr);
			}
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

	bool MaterialManager::containsMaterial(const std::shared_ptr<Material>& mat)
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