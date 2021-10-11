#include "DataModelRegistry.h"

#include <QtCore/QFile>
#include <QtWidgets/QMessageBox>

using QtNodes::DataModelRegistry;
using QtNodes::QtBlockDataModel;
using QtNodes::BlockDataType;
using QtNodes::TypeConverter;

std::unique_ptr<QtBlockDataModel>
DataModelRegistry::
create(QString const &modelName)
{
  auto it = _registeredItemCreators.find(modelName);

  if (it != _registeredItemCreators.end())
  {
    return it->second();
  }

  return nullptr;
}


DataModelRegistry::RegisteredModelCreatorsMap const &
DataModelRegistry::
registeredModelCreators() const
{
  return _registeredItemCreators;
}


DataModelRegistry::RegisteredModelsCategoryMap const &
DataModelRegistry::
registeredModelsCategoryAssociation() const
{
  return _registeredModelsCategory;
}


DataModelRegistry::CategoriesSet const &
DataModelRegistry::
categories() const
{
  return _categories;
}


TypeConverter
DataModelRegistry::
getTypeConverter(BlockDataType const & d1,
                 BlockDataType const & d2) const
{
  TypeConverterId converterId = std::make_pair(d1, d2);

  auto it = _registeredTypeConverters.find(converterId);

  if (it != _registeredTypeConverters.end())
  {
    return it->second;
  }

  return TypeConverter{};
}
