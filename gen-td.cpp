#include "google/protobuf/compiler/cpp/cpp_helpers.h"
#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/compiler/plugin.h>
#include <google/protobuf/descriptor.h>
#undef NDEBUG
using namespace google::protobuf::compiler;
using namespace google::protobuf;
class MyCodeGenerator : public CodeGenerator {
public:
  MyCodeGenerator();
  MyCodeGenerator(MyCodeGenerator &&) = default;
  MyCodeGenerator(const MyCodeGenerator &) = default;
  MyCodeGenerator &operator=(MyCodeGenerator &&) = default;
  MyCodeGenerator &operator=(const MyCodeGenerator &) = default;
  ~MyCodeGenerator();

  bool Generate(const FileDescriptor *file, const std::string &parameter,
                GeneratorContext *generator_context,
                std::string *error) const override;
};

MyCodeGenerator::MyCodeGenerator() {}

MyCodeGenerator::~MyCodeGenerator() {}

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
}

void ConverFields(const google::protobuf::Descriptor *d) {
  FOR_RANGE(i, d->field_count()) {
    const bool is_last = i == d->field_count() - 1;
    auto f = d->field(i);
    std::cerr << "    ";
    auto ods_t = GetODSType(f->type());
    if (f->type() == FieldDescriptor::TYPE_ENUM &&
        f->enum_type()->name() == "DataType") {
      std::cerr << "OneFlow_DataType";
    } else if (f->type() == FieldDescriptor::TYPE_MESSAGE) {
      auto t = f->message_type();
      if (t->name() == "ShapeProto") {
        std::cerr << "ShapeAttr";
      } else if (t->name() == "Int64List") {
        std::cerr << "SI64ArrayAttr";
      } else if (t->name() == "LogicalBlobId") {
        std::cerr << "OneFlow_Tensor";
      } else {
        std::cerr << "[" << t->name() << "]";
      }
    } else if (!ods_t.empty()) {
      std::cerr << ods_t;
    } else {
      LOG("can't handle" + std::to_string(f->type()));
      std::exit(1);
    }
    std::cerr << ":$";
    LOG(f->name() + (is_last ? "" : ","));
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
  std::cerr << parameter << "\n";
  std::cerr << *error << "\n";
  std::cerr << file->name() << "\n";
  // for (int i; i < file->message_type_count(); i++) {
  //   auto m = file->message_type(i);
  //   std::cerr << m->name() << "\n";
  // }
  auto OperatorConf = file->FindMessageTypeByName("OperatorConf");
  std::cerr << OperatorConf->name() << "\n";
  auto op_type = OperatorConf->FindOneofByName("op_type");
  for (int filed_i; filed_i < op_type->field_count(); filed_i += 1) {
    auto m = op_type->field(filed_i);
    const bool should_convert = IsSystemOp(m->number());
    if (!should_convert)
      continue;
    assert(EndsWith(m->name(), "_conf"));
    const std::string register_name = m->name().substr(0, m->name().size() - 5);
    std::cerr << (ShouldGenBaseClass() ? "class" : "def") << " OneFlow_"
              << cpp::UnderscoresToCamelCase(register_name, true)
              << "Op : " << GetBaseOp() << "<\"" << register_name
              << "\", [" + GetTraits() + "]> "; // TODO: add traits
    std::cerr << "\n";
    std::cerr << "  let attrs = (ins\n";
    ConverFields(m->message_type());
    std::cerr << "  );"
              << "\n";
  }
  return true;
}
int main(int argc, char **argv) {
  MyCodeGenerator generator;
  return google::protobuf::compiler::PluginMain(argc, argv, &generator);
}
