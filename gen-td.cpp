#include "google/protobuf/compiler/cpp/cpp_helpers.h"
#include "inja.hpp"
#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/compiler/plugin.h>
#include <google/protobuf/descriptor.h>
#include <regex>

#undef NDEBUG
using namespace google::protobuf::compiler;
using namespace google::protobuf;
using inja::json;

namespace {
json GetEntry(std::string type, const std::string field_name,
              bool is_optional) {
  json j;
  j["ods_type"] = type;
  j["field_name"] = field_name;
  j["is_optional"] = is_optional;
  return j;
}

json GetArgEntry(std::string type, const std::string field_name,
                 bool is_optional, bool is_variadic = false) {
  json j;
  j["ods_type"] = type;
  j["field_name"] = field_name;
  j["is_optional"] = is_optional;
  j["is_variadic"] = is_variadic;
  return j;
}

} // namespace
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
  ODSDefinition &ConverFields(std::string op_type_name,
                              const google::protobuf::Descriptor *d,
                              std::string field_prefix = "",
                              bool is_one_of = false);
  ~ODSDefinition() = default;

  std::string op_type_name;
  json def;
  inja::Environment env;
  void add_input(std::string type, const std::string field_name,
                 bool is_optional = false, bool is_variadic = false) {
    def["input"].push_back(GetArgEntry(type, field_name, is_optional));
  }
  void add_output(std::string type, const std::string field_name,
                  bool is_optional = false, bool is_variadic = false) {
    def["output"].push_back(GetArgEntry(type, field_name, is_optional));
  }
  void add_attr(std::string type, const std::string field_name,
                bool is_optional = false) {
    def["attrs"].push_back(GetEntry(type, field_name, is_optional));
  }
  std::string serialize() {
    return env.render(
        R"(
def OneFlow_{{ op_class_name }} : OneFlow_BaseOp<"{{ name }}", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
{% if length(input) > 0 %}  let input = (ins
## for i in input
      {% if i.is_optional %}Optional<{% endif %}{% if i.is_variadic %}Variadic<{% endif %}{{ i.ods_type }}{% if i.is_variadic %}>{% endif %}{% if i.is_optional %}>{% endif %}:${{ i.field_name }}{% if not loop.is_last %},{% endif %}
## endfor
  );
{% endif %}{% if length(output) %}  let output = (outs
## for o in output
      {% if o.is_optional %}Optional<{% endif %}{% if o.is_variadic %}Variadic<{% endif %}{{ o.ods_type }}{% if o.is_variadic %}>{% endif %}{% if o.is_optional %}>{% endif %}:${{ o.field_name }}{% if not loop.is_last %},{% endif %}
## endfor
  );
{% endif %}{% if length(attrs) %}  let attrs = (ins
## for a in attrs
      {% if a.is_optional %}Optional<{% endif %}{{ a.ods_type }}{% if a.is_optional %}>{% endif %}:${{ a.field_name }}{% if not loop.is_last %},{% endif %}
## endfor
  );
{% endif %}}
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

std::string GetODSType(const FieldDescriptor *f) {
  switch (f->type()) {
  case FieldDescriptor::TYPE_DOUBLE:
    return "F64Attr";
  case FieldDescriptor::TYPE_FLOAT:
    if (f->is_repeated())
      return "F32ArrayAttr";
    return "F32Attr";
  case FieldDescriptor::TYPE_INT64:
  case FieldDescriptor::TYPE_UINT64:
    if (f->is_repeated())
      return "SI64ArrayAttr";
    return "SI64Attr";
  case FieldDescriptor::TYPE_INT32:
    if (f->is_repeated())
      return "SI64ArrayAttr";
    return "SI32Attr";
  case FieldDescriptor::TYPE_FIXED64:
    break;
  case FieldDescriptor::TYPE_FIXED32:
    break;
  case FieldDescriptor::TYPE_BOOL:
    return "BoolAttr";
  case FieldDescriptor::TYPE_STRING:
    if (f->is_repeated())
      return "StrArrayAttr";
    return "StrAttr";
  case FieldDescriptor::TYPE_GROUP:
    break;
  case FieldDescriptor::TYPE_MESSAGE:
    break;
  case FieldDescriptor::TYPE_BYTES:
    break;
  case FieldDescriptor::TYPE_UINT32:
    return "SI32Attr";
  case FieldDescriptor::TYPE_ENUM:
    return "ENUM";
  case FieldDescriptor::TYPE_SFIXED32:
    break;
  case FieldDescriptor::TYPE_SFIXED64:
    break;
  case FieldDescriptor::TYPE_SINT32:
    break;
  case FieldDescriptor::TYPE_SINT64:
    break;
  }
  return "";
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

ODSDefinition &
ODSDefinition::ConverFields(std::string op_type_name,
                            const google::protobuf::Descriptor *d,
                            std::string field_prefix, bool is_one_of) {
  FOR_RANGE(i, d->field_count()) {
    const bool is_last = i == d->field_count() - 1;
    auto f = d->field(i);
    auto ods_t = GetODSType(f);
    if (f->containing_oneof()) {
      is_one_of = true;
    }
    auto filed_name_ = f->name();
    filed_name_ = std::regex_replace(filed_name_, std::regex("_conf"), "");
    filed_name_ = std::regex_replace(filed_name_, std::regex("_lbi"), "");
    filed_name_ = std::regex_replace(filed_name_, std::regex("lbi"), "in");
    const std::string field_name = field_prefix + filed_name_;
    bool is_optional = f->is_optional() || is_one_of;
    if (f->type() == FieldDescriptor::TYPE_ENUM &&
        f->enum_type()->name() == "DataType") {
      add_attr("DataType", field_name, is_optional);
    } else if (f->type() == FieldDescriptor::TYPE_ENUM) {
      add_attr("UndefinedEnum" + f->enum_type()->name(), field_name,
               is_optional);
    } else if (f->type() == FieldDescriptor::TYPE_MESSAGE) {
      auto t = f->message_type();
      if (t->name() == "ShapeProto") {
        add_attr("ShapeAttr", field_name, is_optional);
      } else if (t->name() == "Int64List") {
        add_attr("SI64ArrayAttr", field_name, is_optional);
      } else if (t->name() == "LogicalBlobId") {
        add_input("OneFlow_Tensor", field_name, is_optional, f->is_repeated());
      } else {
        ConverFields(op_type_name, t, field_name + "_", is_one_of);
      }
    } else if (op_type_name == "shape_elem_cnt") {
      if (f->name() == "x")
        add_input("OneFlow_Tensor", field_name, is_optional, f->is_repeated());
      if (f->name() == "y")
        add_output("OneFlow_Tensor", field_name, is_optional, f->is_repeated());
    } else if (f->name() == "out" &&
               f->type() == FieldDescriptor::TYPE_STRING) {
      add_output("OneFlow_Tensor", field_name, is_optional, f->is_repeated());
    } else if ((f->name() == "tick" || f->name() == "in") &&
               f->type() == FieldDescriptor::TYPE_STRING) {
      add_input("OneFlow_Tensor", field_name, is_optional, f->is_repeated());
    } else if (!ods_t.empty()) {
      add_attr(ods_t, field_name, is_optional);
    } else {
      LOG("can't handle" + std::to_string(f->type()));
      std::exit(1);
    }
  }
  return *this;
}

bool MyCodeGenerator::Generate(const FileDescriptor *file,
                               const std::string &parameter,
                               GeneratorContext *generator_context,
                               std::string *error) const {
  auto OperatorConf = file->FindMessageTypeByName("OperatorConf");
  auto op_type = OperatorConf->FindOneofByName("op_type");
  std::ofstream td_file("OneFlowSystemOps.td");
  assert(op_type->field_count());
  for (int filed_i = 0; filed_i < op_type->field_count(); filed_i += 1) {
    auto m = op_type->field(filed_i);
    const bool should_convert = IsSystemOp(m->number());
    if (!should_convert)
      continue;
    assert(EndsWith(m->name(), "_conf"));
    const std::string register_name = m->name().substr(0, m->name().size() - 5);
    ODSDefinition ods_def(register_name);
    td_file
        << ods_def.ConverFields(register_name, m->message_type()).serialize();
  }
  td_file.flush();
  td_file.close();
  return true;
}
int main(int argc, char **argv) {
  MyCodeGenerator generator;
  return google::protobuf::compiler::PluginMain(argc, argv, &generator);
}
