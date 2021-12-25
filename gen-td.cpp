#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/compiler/plugin.h>
#include <google/protobuf/descriptor.h>

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
    auto f = d->field(i);
    std::cerr << "  - ";
    auto ods_t = GetODSType(f->type());
    if (!ods_t.empty()) {
      std::cerr << ods_t << ": ";
    } else if (f->type() == FieldDescriptor::TYPE_MESSAGE) {
      auto t = f->message_type();
      std::cerr << t->name() << ": ";
    } else {
      LOG("can't handle" + std::to_string(f->type()));
      std::exit(1);
    }
    LOG(f->name());
  }
}

const int USER_OP_NUMBER = 199;
bool IsSystemOp(int number) { return number > 100 && number < USER_OP_NUMBER; }

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
    std::cerr << m->number() << "\n";
    std::cerr << m->name() << "\n";
    ConverFields(m->message_type());
  }
  return true;
}
int main(int argc, char **argv) {
  MyCodeGenerator generator;
  return google::protobuf::compiler::PluginMain(argc, argv, &generator);
}
