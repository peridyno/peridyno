#pragma once

#include <functional>
#include <memory>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "WtNodeDataModel.h"

using SharedNodeData = std::shared_ptr<WtNodeData>;
using TypeConverter = std::function<SharedNodeData(SharedNodeData)>;
using TypeConverterId = std::pair<NodeDataType, NodeDataType>;

inline bool operator<(NodeDataType const& d1, NodeDataType const& d2)
{
	return d1.id < d2.id;
}

class WtDataModelRegistry
{
public:
	using RegistryItemPtr = std::unique_ptr<WtNodeDataModel>;
	using RegistryItemCreator = std::function<RegistryItemPtr()>;
	using RegisteredModelCreatorsMap = std::unordered_map<std::string, RegistryItemCreator>;
	using RegisteredModelsCategoryMap = std::unordered_map<std::string, std::string>;
	using CategoriesSet = std::set<std::string>;
	using RegisteredTypeConvertersMap = std::map<TypeConverterId, TypeConverter>;

	WtDataModelRegistry() = default;
	~WtDataModelRegistry() = default;

	WtDataModelRegistry(WtDataModelRegistry const&) = delete;
	WtDataModelRegistry(WtDataModelRegistry&&) = default;

	WtDataModelRegistry& operator=(WtDataModelRegistry const&) = delete;
	WtDataModelRegistry& operator=(WtDataModelRegistry&&) = default;

public:
	template<typename ModelType>
	void registerModel(RegistryItemCreator creator, std::string const& category = "Nodes")
	{
		const std::string name = computeName<ModelType>(HasStaticMethodName<ModelType>{}, creator);
		if (!_registeredItemCreators.count(name))
		{
			_registeredItemCreators[name] = std::move(creator);
			_categories.insert(category);
			_registeredModelsCategory[name] = category;
		}
	}

	template<typename ModelType>
	void registerModel(std::string const& category = "Nodes")
	{
		RegistryItemCreator creator = []() { return std::make_unique<ModelType>(); };
		registerModel<ModelType>(std::move(creator), category);
	}

	template<typename ModelType>
	void registerModel(std::string const& category,
		RegistryItemCreator creator)
	{
		registerModel<ModelType>(std::move(creator), category);
	}

	template <typename ModelCreator>
	void registerModel(ModelCreator&& creator, std::string const& category = "Nodes")
	{
		using ModelType = compute_model_type_t<decltype(creator())>;
		registerModel<ModelType>(std::forward<ModelCreator>(creator), category);
	}

	template <typename ModelCreator>
	void registerModel(std::string const& category, ModelCreator&& creator)
	{
		registerModel(std::forward<ModelCreator>(creator), category);
	}

	void registerTypeConverter(TypeConverterId const& id,
		TypeConverter typeConverter)
	{
		_registeredTypeConverters[id] = std::move(typeConverter);
	}

	std::unique_ptr<WtNodeDataModel>create(std::string const& modelName);

	RegisteredModelCreatorsMap const& registeredModelCreators() const;

	RegisteredModelsCategoryMap const& registeredModelsCategoryAssociation() const;

	CategoriesSet const& categories() const;

	TypeConverter getTypeConverter(NodeDataType const& d1, NodeDataType const& d2) const;

private:
	RegisteredModelsCategoryMap _registeredModelsCategory;

	CategoriesSet _categories;

	RegisteredModelCreatorsMap _registeredItemCreators;

	RegisteredTypeConvertersMap _registeredTypeConverters;

private:
	template <typename T, typename = void>
	struct HasStaticMethodName : std::false_type {};

	template <typename T>
	struct HasStaticMethodName<T, typename std::enable_if<std::is_same<decltype(T::Name()), std::string>::value>::type>
		: std::true_type {};

	template <typename ModelType>
	static std::string computeName(std::true_type, RegistryItemCreator const&)
	{
		return ModelType::Name();
	}

	template<typename ModelType>
	static std::string computeName(std::false_type, RegistryItemCreator const& creator)
	{
		return creator()->name();
	}

	template <typename T>
	struct UnwrapUniquePtr
	{
		// Assert always fires, but the compiler doesn't know this:
		static_assert(!std::is_same<T, T>::value,
			"The ModelCreator must return a std::unique_ptr<T>, where T "
			"inherits from WtNodeDataModel");
	};

	template <typename T>
	struct UnwrapUniquePtr<std::unique_ptr<T>>
	{
		static_assert(std::is_base_of<WtNodeDataModel, T>::value,
			"The ModelCreator must return a std::unique_ptr<T>, where T "
			"inherits from WtNodeDataModel");
		using type = T;
	};

	template <typename CreatorResult>
	using compute_model_type_t = typename UnwrapUniquePtr<CreatorResult>::type;
};