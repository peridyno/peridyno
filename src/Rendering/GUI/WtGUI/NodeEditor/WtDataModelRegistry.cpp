#include "WtDataModelRegistry.h"

std::unique_ptr<WtNodeDataModel> WtDataModelRegistry::create(std::string const& modelName)
{
	auto it = _registeredItemCreators.find(modelName);
	if (it != _registeredItemCreators.end())
	{
		return it->second();
	}

	return nullptr;
}

WtDataModelRegistry::RegisteredModelCreatorsMap const& WtDataModelRegistry::registeredModelCreators() const
{
	return _registeredItemCreators;
}

WtDataModelRegistry::RegisteredModelsCategoryMap const& WtDataModelRegistry::registeredModelsCategoryAssociation() const
{
	return _registeredModelsCategory;
}

WtDataModelRegistry::CategoriesSet const& WtDataModelRegistry::categories() const
{
	return _categories;
}

TypeConverter WtDataModelRegistry::getTypeConverter(NodeDataType const& d1, NodeDataType const& d2) const
{
	TypeConverterId converterId = std::make_pair(d1, d2);

	auto it = _registeredTypeConverters.find(converterId);

	if (it != _registeredTypeConverters.end())
	{
		return it->second;
	}
	return TypeConverter{};
}