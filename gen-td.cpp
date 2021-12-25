#include "google/protobuf/compiler/cpp/cpp_helpers.h"
#include "inja.hpp"
#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/compiler/plugin.h>
#include <google/protobuf/descriptor.h>

#undef NDEBUG
using namespace google::protobuf::compiler;
using namespace google::protobuf;
using inja::json;

class ODSDefinition {
public:
  ODSDefinition(std::string op_type_name)
      : op_type_name(op_type_name),
        def{{"name", op_type_name},
            {"op_class_name",
             cpp::UnderscoresToCamelCase(op_type_name, true) + "Op"},
            {"input", json::array()},
            {"output", json::array()},
            {"attrs", json::array()}} {};
  ODSDefinition(ODSDefinition &&) = default;
  ODSDefinition(const ODSDefinition &) = default;
  ODSDefinition &operator=(ODSDefinition &&) = default;
  ODSDefinition &operator=(const ODSDefinition &) = default;
  ~ODSDefinition() = default;

  std::string op_type_name;
  json def;
  inja::Environment env;
  void add_input(const std::string &field) { def["input"].push_back(field); }
  void add_output(const std::string &field) { def["output"].push_back(field); }
  void add_attr(const std::string &field) { def["attrs"].push_back(field); }
  std::string serialize() {
    return env.render(
        R"(
def OneFlow_{{ op_class_name }} : OneFlow_BaseOp<"{{ name }}", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
{% if length(input) %}
  let input = (ins
## for i in input
      {{ i }}{% if not loop.is_last %},{% endif %}
## endfor
  );
{% endif %}
{% if length(output) %}
  let output = (outs
## for o in output
      {{ o }}{% if not loop.is_last %},{% endif %}
## endfor
  );
{% endif %}
{% if length(attrs) %}
  let attrs = (ins
## for attr in attrs
      {{ attr }}{% if not loop.is_last %},{% endif %}
## endfor
  );
{% endif %}
}
)",
        def);
  }
};

class MyCodeGenerator : public CodeGenerator {
public:
  MyCodeGenerator() = default;
  MyCodeGenerator(MyCodeGenerator &&) = default;
  MyCodeGenerator(const MyCodeGenerator &) = default;
  MyCodeGenerator &operator=(MyCodeGenerator &&) = default;
  MyCodeGenerator &operator=(const MyCodeGenerator &) = default;
  ~MyCodeGenerator() = default;

  bool Generate(const FileDescriptor *file, const std::string &parameter,
                GeneratorContext *generator_context,
                std::string *error) const override;
};

#define FOR_RANGE(i, end) for (size_t i = (0), __end = (end); i < __end; ++i)
#define LOG(x) std::cerr << x << "\n";
namespace {

std::string GetODSType(FieldDescriptor::Type t) {
  switch (t) {
  case FieldDescriptor::TYPE_DOUBLE:
    return "F64Attr";
  case FieldDescriptor::TYPE_FLOAT:
    return "F32Attr";
  case FieldDescriptor::TYPE_INT64:
    return "SI64Attr";
  case FieldDescriptor::TYPE_UINT64:
    return "SI64Attr";
  case FieldDescriptor::TYPE_INT32:
    return "SI32Attr";
  case FieldDescriptor::TYPE_FIXED64:
    return "";
  case FieldDescriptor::TYPE_FIXED32:
    return "";
  case FieldDescriptor::TYPE_BOOL:
    return "BoolAttr";
  case FieldDescriptor::TYPE_STRING:
    return "StrAttr";
  case FieldDescriptor::TYPE_GROUP:
    return "";
  case FieldDescriptor::TYPE_MESSAGE:
    return "";
  case FieldDescriptor::TYPE_BYTES:
    return "";
  case FieldDescriptor::TYPE_UINT32:
    return "SI32Attr";
  case FieldDescriptor::TYPE_ENUM:
    return "ENUM";
  case FieldDescriptor::TYPE_SFIXED32:
    return "";
  case FieldDescriptor::TYPE_SFIXED64:
    return "";
  case FieldDescriptor::TYPE_SINT32:
    return "";
  case FieldDescriptor::TYPE_SINT64:
    return "";
  }
  return "";
}

void ConverFields(const google::protobuf::Descriptor *d,
                  ODSDefinition &ods_def) {
  FOR_RANGE(i, d->field_count()) {
    const bool is_last = i == d->field_count() - 1;
    auto f = d->field(i);
    auto ods_t = GetODSType(f->type());
    if (f->type() == FieldDescriptor::TYPE_ENUM &&
        f->enum_type()->name() == "DataType") {
      ods_def.add_attr("DataType:$" + f->name());
    } else if (f->type() == FieldDescriptor::TYPE_MESSAGE) {
      auto t = f->message_type();
      if (t->name() == "ShapeProto") {
        ods_def.add_attr("ShapeAttr:$" + f->name());
      } else if (t->name() == "Int64List") {
        ods_def.add_input("SI64ArrayAttr:$" + f->name());
      } else if (t->name() == "LogicalBlobId") {
        ods_def.add_input("OneFlow_Tensor:$" + f->name());
      } else {
        ods_def.add_attr("[" + t->name() + "]" + ":$" + f->name());
      }
    } else if (!ods_t.empty()) {
      ods_def.add_attr(ods_t + ":$" + f->name());
    } else {
      LOG("can't handle" + std::to_string(f->type()));
      std::exit(1);
    }
  }
}

const int USER_OP_NUMBER = 199;
bool IsSystemOp(int number) { return number > 100 && number < USER_OP_NUMBER; }

bool EndsWith(const std::string &data, const std::string &suffix) {
  return data.find(suffix, data.size() - suffix.size()) != std::string::npos;
}

std::string GetBaseOp() { return "OneFlow_BaseOp"; }

std::string GetTraits() { return ""; }

bool ShouldGenBaseClass() { return false; }

} // namespace

bool MyCodeGenerator::Generate(const FileDescriptor *file,
                               const std::string &parameter,
                               GeneratorContext *generator_context,
                               std::string *error) const {
  auto OperatorConf = file->FindMessageTypeByName("OperatorConf");
  auto op_type = OperatorConf->FindOneofByName("op_type");
  for (int filed_i; filed_i < op_type->field_count(); filed_i += 1) {
    auto m = op_type->field(filed_i);
    const bool should_convert = IsSystemOp(m->number());
    if (!should_convert)
      continue;
    assert(EndsWith(m->name(), "_conf"));
    const std::string register_name = m->name().substr(0, m->name().size() - 5);
    ODSDefinition ods_def(register_name);
    ConverFields(m->message_type(), ods_def);
    std::cerr << ods_def.serialize();
  }
  return true;
}
int main(int argc, char **argv) {
  MyCodeGenerator generator;
  return google::protobuf::compiler::PluginMain(argc, argv, &generator);
}
